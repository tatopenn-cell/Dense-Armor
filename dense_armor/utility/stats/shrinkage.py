"""Online shrinkage covariance (Bodnar, Gupta and Parolya 2014).

The sample covariance of a small window is noisy: the off-diagonal
entries are estimated from few observations, and a portfolio built on
them is unstable. Linear shrinkage takes a weighted average of the
sample covariance and a target (a multiple of the identity), and the
weight is chosen to minimise the expected Frobenius loss.

Reference: Bodnar, T., Gupta, A. K. and Parolya, N. (2014). *On the
Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
Dimensional Covariance Matrix.* arXiv:1308.2608. The optimal shrinkage
estimator is equation (4.2), page 12; the two intensities are equations
(4.3) and (4.4), page 12; the target is ``Sigma_0 = (1/p) I``, page 12.

This class computes the two intensities from the running first and
second moments as the data arrive. The running moments are the ones of
:class:`dense_armor.utility.stats.dependence.RunningCovariance`, one per
pair and one per diagonal entry; the intensities are recomputed from the
moments on every call to :meth:`shrinkage`. The running computation of
the intensities is a choice of this implementation: the paper gives the
closed form on the sample covariance, not a running update, and the
closed form is what :meth:`shrinkage` evaluates.
"""

from __future__ import annotations

import numpy as np

from dense_armor.utility.stats.dependence import RunningCovariance


class OnlineShrinkageCovariance:
    """Online covariance with linear shrinkage toward ``(1/p) I``.

    Args:
        n_features: number of variables ``p``.
        feature_prefix: prefix of the keys of the input dict. The entry
            ``i`` is read from the key ``f"{feature_prefix}{i}"``.
            Default ``"v"``.

    Raises:
        ValueError: on a non-positive ``n_features``.
    """

    budget_s = 1e-3
    memory_class = "O(p^2)"

    def __init__(self, n_features: int, feature_prefix: str = "v") -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        self.n_features = int(n_features)
        self.feature_prefix = str(feature_prefix)
        self._pairs: dict[tuple[int, int], RunningCovariance] = {}
        for i in range(self.n_features):
            for j in range(i, self.n_features):
                key_i = f"{self.feature_prefix}{i}"
                key_j = f"{self.feature_prefix}{j}"
                self._pairs[(i, j)] = RunningCovariance(key_i, key_j)
        self.n_seen_: int = 0

    def learn_one(self, x: dict, y=None, t: float | None = None) -> OnlineShrinkageCovariance:
        """Update the running moments with one sample."""
        for cov in self._pairs.values():
            cov.learn_one(x, t=t)
        self.n_seen_ += 1
        return self

    def sample_covariance(self) -> np.ndarray:
        """The running sample covariance, ``p`` by ``p``."""
        p = self.n_features
        s = np.zeros((p, p), dtype=float)
        for (i, j), cov in self._pairs.items():
            s[i, j] = cov.cov
            s[j, i] = cov.cov
        return s

    def shrinkage(self) -> tuple[np.ndarray, float, float]:
        """The shrunk covariance, and the two intensities.

        The intensities are equations (4.3) and (4.4) of the paper, page
        12, with ``Sigma_0 = (1/p) I``:

            alpha* = 1 - (tr(S)^2 / n) / (tr(S^2) - tr(S)^2 / p)
            beta*  = (1 - alpha*) * tr(S)

        with ``Sigma_0 = (1/p) I``, so ``tr(S Sigma_0) = tr(S) / p`` and
        ``||Sigma_0||_F^2 = 1/p``, and the product ``beta* (1/p) I`` is
        the target term. ``n`` is the smallest number of complete pairs
        among the entries of ``S`` (a sample with a NaN is skipped by the
        pairs that use it), which equals ``n_seen_`` without NaN.

        Returns:
            ``(shrunk, alpha, beta)``: the shrunk covariance, the
            intensity ``alpha`` on the sample covariance and the
            intensity ``beta`` of equation (4.4), which multiplies the
            target ``(1/p) I``.
        """
        s = self.sample_covariance()
        n = max(min(cov.n_ for cov in self._pairs.values()), 1)
        p = self.n_features
        tr_s = float(np.trace(s))
        tr_s2 = float(np.sum(s * s))
        denom = tr_s2 - tr_s * tr_s / p
        if denom <= 0.0:
            return s.copy(), 1.0, 0.0
        alpha = 1.0 - (tr_s * tr_s / n) / denom
        alpha = float(min(max(alpha, 0.0), 1.0))
        beta = (1.0 - alpha) * tr_s
        shrunk = alpha * s + (beta / p) * np.eye(p)
        return shrunk, alpha, beta
