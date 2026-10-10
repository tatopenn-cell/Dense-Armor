"""Online robust Mahalanobis distance for multichannel anomaly detection.

Anomaly detection with a robust estimate of the data covariance: the
location is the geometric median (a robust alternative to the mean),
the dispersion is the median covariation matrix (MCM), and the score
is the Mahalanobis distance of the sample from the median under the
current MCM. Both the median and the MCM are updated online with
averaged stochastic gradient.

The geometric median is initialised by the Weiszfeld algorithm
(Guillot et al. 2025, eq. 6). The MCM is initialised by the empirical
average of the centred outer products instead of the Weiszfeld
offline MCM of Section A.2: with ``n_init`` of the order of tens of
samples the average is more stable than the Weiszfeld iteration,
which for small samples can converge to a rank-one solution. The
online averaged stochastic gradient immediately refines the estimate
as soon as new samples arrive. This is a choice, not the paper's
offline initialisation. Without an offline initialisation the ASGD
updates, whose gradients are normalised to unit Frobenius norm, start
from the identity matrix and never reach the scale of the data.

Two steps of the full method are not implemented. First, Guillot et
al. (2025, Section 3.2 and eqs. 4-5) reconstruct the eigenvalues of
the true covariance from the eigenvalues of the MCM by a Robbins-Monro
scheme; that step refines the absolute scale of the distance, because
the MCM eigenvalues under-estimate the variance. Second, eq. 1 scales
the squared Mahalanobis distances by ``chi2_d(0.5) / med(D_1..n)`` so
that the median of the corrected distances matches the median of the
chi-squared distribution of order ``d``. Neither is applied here: for
flagging anomalies the MCM metric with a calibrated ``threshold`` is
already usable, and the scaling constant would only shift the number
at which the calibration is done. The threshold in this module is the
one the user calibrates, not the ``chi2_d(0.5)`` one of eq. 1.

References
----------
Guillot, A., Godichon-Baggioni, A., Robin, S. (2025). Online robust
    covariance and outlier detection. arXiv:2601.03957. Sections 3.2.1
    (online algorithm and initialisation), 3.2.2 (streaming version),
    and 3.4 (outlier detection).
Cardot, H., Cenac, P., Zitt, P.-A. (2013). Bernoulli 19(1), 18-43.
Cardot, H., Godichon-Baggioni, A. (2017). TEST 26(3), 461-480.
"""

from collections.abc import Sequence

import numpy as np

from dense_armor.roles import AnomalyDetector


class OnlineRobustMahalanobis(AnomalyDetector):
    """Online robust Mahalanobis scorer for multichannel signals.

    ``learn_one(x)`` accepts a dict of features (one per channel). For
    the first ``n_init`` samples the location and the dispersion are
    accumulated; at step ``n_init`` the geometric median and the
    median covariation matrix are estimated offline (Weiszfeld for the
    median, average of centred outer products for the MCM). From
    ``n_init + 1`` on, both are updated by averaged stochastic gradient
    (Cardot et al. 2013, Cardot and Godichon-Baggioni 2017).

    ``score_one(x)`` returns the Mahalanobis distance

        D(x) = sqrt( sum_j (1 / delta[j]) * <x - m_bar, P[:, j]>^2 )

    where ``P[:, j]`` is the j-th eigenvector of ``V_bar`` and
    ``delta[j]`` is the corresponding eigenvalue of the averaged MCM.
    ``0.0`` until the offline initialisation is complete.

    Args:
        c_gamma: numerator of the step size
            ``gamma_n = c_gamma * (n + n0) ** (-gamma_exp)`` for the
            averaged stochastic gradient updates.
        gamma_exp: exponent of the step size; must be in ``(1/2, 1)``
            for convergence (Cardot et al. 2013).
        n0: warm-up offset in the step size.
        n_init: number of samples accumulated before the offline
            initialisation. The paper uses 100 in the simulations
            (section 4.1). Must be at least ``d + 1`` where ``d`` is
            the number of features.
        max_step_frac: cap on the ASGD step length, as a fraction
            of the current Frobenius norm of ``V`` (and of the
            Euclidean norm of ``m``). The paper's step size
            ``gamma_n = c_gamma * (n + n0) ** -gamma_exp`` can, at
            ``c_gamma = 1``, move ``V`` by 1 in Frobenius norm in a
            single step, much larger than the scale of the data:
            without a cap the averaged MCM oscillates through zero
            and its eigenvalues become negative. Default ``0.1``.
            This is a numerical safeguard, not part of the paper.
        feature_keys: list of dict keys, one per channel. ``None``
            takes the sorted keys of the first dict seen.
        eps: numerical guard for divisions and degenerate eigenvalues.
            Additionally, every eigenvalue of ``V_bar`` is floored at
            ``trace(V_bar) / d * 1e-3``: during the first ASGD steps
            the unit-normalised gradient can drive ``V`` through
            zero, and the floor keeps the score bounded without
            changing the shape of the metric. This is a numerical
            safeguard, not part of the paper.
        seed: unused in this minimal version, kept for interface
            stability with the full version.
        threshold: distance above which ``is_outlier`` returns True.
            Default 7.0 (calibrated on the seeded Gaussian benchmark;
            with the MCM eigenvalues the score lives on a scale
            comparable to the chi-squared one, so the classical 7.0 is
            a reasonable starting point for a low-dimensional signal).

    Raises:
        ValueError: on a non-positive parameter or a bad
            ``feature_keys`` size.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.mahalanobis import (
        ...     OnlineRobustMahalanobis,
        ... )
        >>> rng = np.random.default_rng(0)
        >>> model = OnlineRobustMahalanobis(
        ...     feature_keys=["a", "b"], n_init=200
        ... )
        >>> for _ in range(500):
        ...     x = rng.standard_normal(2)
        ...     _ = model.learn_one(
        ...         {"a": float(x[0]), "b": float(x[1])}
        ...     )
        >>> inlier = model.score_one({"a": 0.0, "b": 0.0})
        >>> outlier = model.score_one({"a": 10.0, "b": 10.0})
        >>> outlier > inlier
        True

    References:
        Guillot, A., Godichon-Baggioni, A., Robin, S. (2025).
        arXiv:2601.03957. Sections 3.2.1, 3.2.2, 3.4.
    """

    budget_s = 0.005
    memory_class = "O(1)"

    def __init__(
        self,
        c_gamma: float = 1.0,
        gamma_exp: float = 0.7,
        n0: int = 0,
        n_init: int = 100,
        max_step_frac: float = 0.1,
        feature_keys: Sequence[str] | None = None,
        eps: float = 1e-9,
        seed: int | None = None,
        threshold: float = 7.0,
    ) -> None:
        if c_gamma <= 0:
            raise ValueError(f"c_gamma must be > 0, got {c_gamma}")
        if not (0.5 < gamma_exp < 1.0):
            raise ValueError(f"gamma_exp must be in (1/2, 1), got {gamma_exp}")
        if n0 < 0:
            raise ValueError(f"n0 must be >= 0, got {n0}")
        if n_init < 2:
            raise ValueError(f"n_init must be >= 2, got {n_init}")
        if not (0.0 < max_step_frac <= 1.0):
            raise ValueError(f"max_step_frac must be in (0, 1], got {max_step_frac}")
        self.c_gamma = float(c_gamma)
        self.gamma_exp = float(gamma_exp)
        self.n0 = int(n0)
        self.n_init = int(n_init)
        self.max_step_frac = float(max_step_frac)
        self.feature_keys = feature_keys
        self.eps = eps
        self.seed = seed
        self.threshold = float(threshold)
        self._dim: int | None = None
        self._m: np.ndarray | None = None
        self._m_bar: np.ndarray | None = None
        self._V: np.ndarray | None = None
        self._V_bar: np.ndarray | None = None
        self._delta: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._n: int = 0
        self._n_after_init: int = 0
        self._ready: bool = False
        self._keys: list[str] | None = None
        self._init_buf: list[np.ndarray] = []
        self.n_missing_: int = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _resolve_keys(self, x) -> list[str]:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        return self._keys

    def _to_vec(self, x) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x[k]) for k in keys], dtype=float)

    def _gamma(self, n: int) -> float:
        return self.c_gamma * (n + self.n0 + 1) ** (-self.gamma_exp)

    def _weiszfeld(self, pts: np.ndarray, n_iter: int = 20) -> np.ndarray:
        m = pts.mean(axis=0)
        for _ in range(n_iter):
            diff = pts - m
            d = np.linalg.norm(diff, axis=1)
            d = np.maximum(d, self.eps)
            w = 1.0 / d
            m_new = (w[:, None] * pts).sum(axis=0) / w.sum()
            if np.linalg.norm(m_new - m) < self.eps:
                m = m_new
                break
            m = m_new
        return m

    def _offline_init(self) -> None:
        pts = np.array(self._init_buf, dtype=float)
        self._init_buf = []
        d = pts.shape[1]
        if pts.shape[0] < d + 1:
            self._m = pts.mean(axis=0)
            self._V = np.eye(d)
            self._m_bar = self._m.copy()
            self._V_bar = self._V.copy()
        else:
            self._m = self._weiszfeld(pts)
            centred = pts - self._m
            self._V = (centred[:, :, None] * centred[:, None, :]).mean(axis=0)
            self._V = 0.5 * (self._V + self._V.T)
            self._m_bar = self._m.copy()
            self._V_bar = self._V.copy()
        try:
            delta, P = np.linalg.eigh(self._V_bar)
            trace = float(np.trace(self._V_bar))
            floor = max(trace / max(self._dim or 1, 1) * 1e-3, self.eps)
            self._delta = np.maximum(delta, floor)
            self._P = P
            self._ready = True
        except np.linalg.LinAlgError:
            self._ready = False
        self._n_after_init = 0

    def learn_one(self, x: dict, t: float | None = None) -> "OnlineRobustMahalanobis":
        """Learn one sample.

        Args:
            x: dict of one feature per channel.
            t: unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        try:
            y = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(y).all():
            self.n_missing_ += 1
            return self
        if self._dim is None:
            self._dim = int(y.size)
        if self._n < self.n_init:
            self._init_buf.append(y.copy())
            self._n += 1
            if self._n == self.n_init:
                self._offline_init()
            return self

        self._n_after_init += 1
        n = self._n_after_init
        gamma = self._gamma(n)

        assert self._m is not None and self._m_bar is not None
        assert self._V is not None and self._V_bar is not None

        diff = y - self._m
        nrm = float(np.linalg.norm(diff))
        if nrm > self.eps:
            step_m = min(
                gamma,
                self.max_step_frac * float(np.linalg.norm(self._m)) + self.eps,
            )
            self._m = self._m + step_m * diff / nrm
        self._m_bar = self._m_bar + (1.0 / (n + 1)) * (self._m - self._m_bar)

        outer = np.outer(y - self._m_bar, y - self._m_bar)
        dV = outer - self._V
        nrm_V = float(np.linalg.norm(dV, "fro"))
        if nrm_V > self.eps:
            step_V = min(
                gamma,
                self.max_step_frac * float(np.linalg.norm(self._V, "fro")) + self.eps,
            )
            self._V = self._V + step_V * dV / nrm_V
        self._V_bar = self._V_bar + (1.0 / (n + 1)) * (self._V - self._V_bar)

        if n % 10 == 0 or n < 10:
            try:
                delta, P = np.linalg.eigh(self._V_bar)
                trace = float(np.trace(self._V_bar))
                floor = max(
                    trace / max(self._dim or 1, 1) * 1e-3,
                    self.eps,
                )
                self._delta = np.maximum(delta, floor)
                self._P = P
                self._ready = True
            except np.linalg.LinAlgError:
                pass
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample.

        Args:
            x: dict of one feature per channel.
            t: unused.

        Returns:
            Mahalanobis distance under the current MCM. ``0.0`` before
            the offline initialisation is complete or on a non-finite
            input.
        """
        self._time_step(t)
        if not self._ready:
            return 0.0
        try:
            y = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        if not np.isfinite(y).all():
            return 0.0
        assert self._m_bar is not None
        assert self._P is not None and self._delta is not None
        diff = y - self._m_bar
        proj = self._P.T @ diff
        d2 = float(np.sum((proj**2) / self._delta))
        return float(np.sqrt(max(d2, 0.0)))

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
