"""SONAR: SGD-based One-Class Novelty detection with Approximate RBFs.

One-Class SVM finds the maximum-margin hyperplane separating the
observed data from the origin in a reproducing kernel Hilbert space.
The paper of Suk and Kpotufe (2025) rewrites the OCSVM objective in an
empirical risk form, replaces the kernel with Random Fourier Features
(RFF) so gradient updates depend only on the current sample, adds a
strongly convex regularisation, and solves the resulting objective by
stochastic gradient descent. The resulting algorithm is called SONAR.

Objective (Suk and Kpotufe 2025, eq. 9):

    F(w, rho) = (||w||^2 + rho^2) / 2 - nu * rho
                + E_X[ max(0, rho - w^T z(x)) ]

with ``z(x)`` the RFF embedding, ``nu`` the expected outlier
proportion, ``w`` the hyperplane normal and ``rho`` its offset.

SGD update (Algorithm 1):

    Z_t     = 1{ w_{t-1}^T z(X_t) <= rho_{t-1} }
    w_t     = w_{t-1} - eta_{t-1} * (w_{t-1} - z(X_t) * Z_t)
    rho_t   = rho_{t-1} - eta_{t-1} * (rho_{t-1} - nu + Z_t)

with step sizes ``eta_t = step / t`` (``eta_t := 1/t`` in the paper).

RFF embedding (Rahimi and Recht 2007, cos-sin pair):

    z(x) = sqrt(2/d) * [sin(w_1^T x), cos(w_1^T x), ...,
                        sin(w_d^T x), cos(w_d^T x)]

with ``w_j ~ N(0, 2 * gamma * I)`` for the RBF kernel
``K(x, y) = exp(-gamma * ||x - y||^2)``.

Details not fixed by the paper, stated explicitly:

- The paper assumes the data lie on the unit sphere; that assumption
  is about the data, not a preprocessing step the paper prescribes.
  Dividing each sample by its Euclidean norm maps a large fault in
  the direction of a healthy sample onto the same point, so it cannot
  be used. This module standardises each feature with running
  statistics learned from the first ``n_init`` samples, frozen
  afterwards: a later shift in the data is measured against the
  pre-fault distribution.
- The RFF feature count is a hyper-parameter ``n_features_rff``; the
  paper's bound on the number of features (Appendix A) depends on the
  data dimension and the assumed margin and is not used here as a
  formula.
- Step size ``eta_t = step / t`` with ``step`` a user constant
  (``1.0`` recovers the paper's ``eta_t = 1/t``; the paper's own
  experiments use AdaGrad).
- Threshold: when ``threshold`` is ``None`` (default) the effective
  threshold is the ``threshold_quantile``-quantile of the last
  ``threshold_window`` scores (default 0.95 and 500); while fewer than
  ``threshold_warmup`` scores have been seen the effective threshold
  is ``1e9`` (flag nothing).

Known limitation on short one-pass streams. SONAR's SGD convergence
(Theorem 6 and Corollary 7 of the paper) requires the number of steps
to grow like ``(eps * lambda) ** -2 * log(1 / delta)`` before the
Type I error bound becomes tight. On the benchmark streams used in
this repository (a few hundred healthy samples, then a fault that
does not reappear), the boundary has not yet converged when the fault
arrives, so the score of the fault is close to the score of a healthy
point and the ROC-AUC is close to chance: the algorithm detects the
fault only if it can learn from enough healthy samples first, which
this benchmark does not provide. On a longer single-pass stream with
a few thousand healthy samples before the first fault, the boundary
has time to converge and the detector behaves as the paper describes.

References
----------
Suk, J., Kpotufe, S. (2025). An efficient variant of one-class SVM
    with lifelong online learning guarantees. arXiv:2512.11052.
    Algorithm 1 (SONAR), Proposition 2, Theorems 3, 4, 6, 8,
    Appendix A.
Rahimi, A., Recht, B. (2007). Random features for large-scale kernel
    machines. NIPS.
"""

from collections import deque
from collections.abc import Sequence

import numpy as np

from dense_armor.roles import AnomalyDetector


class OneClassSGD(AnomalyDetector):
    """One-Class SVM solved by SGD on random Fourier features.

    Args:
        n_features_rff: number of random Fourier frequencies. The RFF
            embedding has ``2 * n_features_rff`` entries.
        nu: expected outlier proportion, in ``(0, 1)``.
        step: constant ``c`` in the step size ``eta_t = c / t``. The
            default ``0.1`` keeps the first step small enough to behave
            well on a few hundred samples.
        gamma: RBF kernel parameter; the frequencies are drawn from
            ``N(0, 2 * gamma * I)``.
        n_init: number of samples collected before the standardisation
            statistics are frozen. Default 20.
        feature_keys: names of the dict keys, in order. ``None`` uses
            the sorted keys of the first sample seen.
        threshold: fixed threshold, or ``None`` for the adaptive
            quantile rule (see the module docstring).
        threshold_quantile: quantile of the recent scores used as the
            effective threshold when ``threshold`` is ``None``.
            Default 0.95.
        threshold_window: number of most recent scores kept for the
            adaptive threshold. Default 500.
        threshold_warmup: number of scores required before the
            adaptive threshold becomes effective. Default 50.
        seed: seed of the random generator.
        eps: numerical guard.

    Raises:
        ValueError: on a bad parameter.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.ocsvm import OneClassSGD
        >>> ds = OneClassSGD(n_features_rff=64, nu=0.05, n_init=20,
        ...                  seed=0)
        >>> rng = np.random.default_rng(0)
        >>> for _ in range(500):
        ...     _ = ds.learn_one({
        ...         "a": float(rng.normal(0.5, 0.05)),
        ...         "b": float(rng.normal(0.5, 0.05)),
        ...     })
        >>> s_in = ds.score_one({"a": 0.5, "b": 0.5})
        >>> s_out = ds.score_one({"a": 5.0, "b": 5.0})
        >>> isinstance(s_in, float) and isinstance(s_out, float)
        True
    """

    budget_s = 0.001
    memory_class = "O(1)"

    def __init__(
        self,
        n_features_rff: int = 128,
        nu: float = 0.05,
        step: float = 0.1,
        gamma: float = 0.5,
        n_init: int = 20,
        feature_keys: Sequence[str] | None = None,
        threshold: float | None = None,
        threshold_quantile: float = 0.95,
        threshold_window: int = 500,
        threshold_warmup: int = 50,
        seed: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        if n_features_rff < 1:
            raise ValueError(f"n_features_rff must be >= 1, got {n_features_rff}")
        if not (0.0 < nu < 1.0):
            raise ValueError(f"nu must be in (0, 1), got {nu}")
        if step <= 0.0:
            raise ValueError(f"step must be > 0, got {step}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        if not (0.0 < threshold_quantile < 1.0):
            raise ValueError(
                f"threshold_quantile must be in (0, 1), got {threshold_quantile}"
            )
        self.n_features_rff = int(n_features_rff)
        self.nu = float(nu)
        self.step = float(step)
        self.gamma = float(gamma)
        self.n_init = int(n_init)
        self.feature_keys = feature_keys
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_window = int(threshold_window)
        self.threshold_warmup = int(threshold_warmup)
        self._threshold_fixed: float | None = (
            None if threshold is None else float(threshold)
        )
        self.threshold = 1e9 if threshold is None else float(threshold)
        self.seed = seed
        self.eps = eps
        self.n_missing_ = 0
        self._ready = False
        self._keys_ready = False
        self._keys: list[str] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._omega: np.ndarray | None = None
        self._scale: float = 1.0
        self._w: np.ndarray | None = None
        self._rho: float = 0.0
        self._n: int = 0
        self._init_buf: list[np.ndarray] = []
        self._score_hist: deque = deque(maxlen=threshold_window)

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _lazy_init(self, x: dict) -> None:
        if self._keys_ready:
            return
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        self._keys_ready = True

    def _to_vec(self, x: dict) -> np.ndarray:
        return np.array([float(x[k]) for k in self._keys], dtype=float)

    def _finalise_stats(self) -> None:
        arr = np.array(self._init_buf, dtype=float)
        self._init_buf = []
        d = arr.shape[1]
        self._mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        self._std = np.where(std > self.eps, std, 1.0)
        rng = np.random.default_rng(self.seed)
        sd = float(np.sqrt(2.0 * self.gamma))
        self._omega = rng.normal(0.0, sd, size=(self.n_features_rff, d))
        self._scale = float(np.sqrt(2.0 / self.n_features_rff))
        self._w = np.zeros(2 * self.n_features_rff, dtype=float)
        self._rho = 0.0
        self._n = 0
        self._ready = True

    def _to_unit(self, x: dict) -> np.ndarray:
        v = self._to_vec(x)
        assert self._mean is not None and self._std is not None
        return (v - self._mean) / self._std

    def _rff(self, x_unit: np.ndarray) -> np.ndarray:
        assert self._omega is not None
        arg = self._omega @ x_unit
        z = np.empty(2 * self.n_features_rff, dtype=float)
        z[0::2] = np.sin(arg)
        z[1::2] = np.cos(arg)
        z *= self._scale
        return z

    def _raw_score(self, x: dict) -> float:
        try:
            x_unit = self._to_unit(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        z = self._rff(x_unit)
        assert self._w is not None
        return float(self._rho - float(self._w @ z))

    def _update_threshold(self, s: float) -> None:
        if self._threshold_fixed is not None:
            return
        self._score_hist.append(s)
        if len(self._score_hist) < self.threshold_warmup:
            self.threshold = 1e9
            return
        self.threshold = float(
            np.quantile(list(self._score_hist), self.threshold_quantile)
        )

    def learn_one(self, x: dict, t: float | None = None) -> "OneClassSGD":
        """Learn one sample with one SGD step.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        if not self._ready:
            self._lazy_init(x)
            try:
                v_r = self._to_vec(x)
            except (KeyError, TypeError, ValueError):
                self.n_missing_ += 1
                return self
            if not np.isfinite(v_r).all():
                self.n_missing_ += 1
                return self
            self._init_buf.append(v_r)
            if len(self._init_buf) >= self.n_init:
                self._finalise_stats()
            return self
        try:
            v = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return self
        try:
            x_unit = self._to_unit(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        self._update_threshold(self._raw_score(x))
        z = self._rff(x_unit)
        assert self._w is not None
        self._n += 1
        eta = self.step / self._n
        proj = float(self._w @ z)
        Z = 1.0 if proj <= self._rho else 0.0
        self._w = self._w - eta * (self._w - z * Z)
        self._rho = self._rho - eta * (self._rho - self.nu + Z)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample: higher means more anomalous.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            The signed distance ``rho - w^T z(x)`` to the boundary.
            Positive outside the normal region. ``0.0`` while the
            standardisation statistics are being collected.
        """
        if not self._ready:
            return 0.0
        return self._raw_score(x)

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
