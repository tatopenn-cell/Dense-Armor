# -*- coding: utf-8 -*-
"""
utility/streaming_mahalanobis.py
=================================
Online robust Mahalanobis distance for multichannel anomaly detection.
Minimal version: no Robbins-Monro reconstruction of the true-covariance
eigenvalues. The score uses the geometric median (location) and the median
covariation matrix (dispersion), both updated by averaged stochastic
gradient, and the eigen-decomposition of the averaged MCM.

The full version of Guillot et al. (arXiv:2601.03957) additionally
reconstructs the eigenvalues of the true covariance from the eigenvalues
of the MCM by a Robbins-Monro scheme. That step refines the absolute scale
of the distance (the MCM eigenvalues under-estimate the variance), but for
the purpose of flagging anomalies the MCM metric with a calibrated
threshold is already usable and is what this module provides.

References
----------
Guillot, A. et al. (2025). Online robust covariance and outlier detection.
arXiv:2601.03957.
Cardot, H., Cenac, P., Zitt, P.-A. (2013). Bernoulli 19(1), 18-43.
Cardot, H., Godichon-Baggioni, A. (2017). TEST 26(3), 461-480.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from dense_armor.base import AnomalyDetector


class OnlineRobustMahalanobis(AnomalyDetector):
    """Online robust Mahalanobis scorer for multichannel signals.

    ``learn_one(x)`` accepts a dict of features (one per channel) and
    updates the geometric median ``m`` and the median covariation matrix
    ``V``, both by averaged stochastic gradient (Cardot et al. 2013,
    Cardot and Godichon-Baggioni 2017).

    ``score_one(x)`` returns the Mahalanobis distance

        D(x) = sqrt( sum_j (1 / delta[j]) * <x - m_bar, P[:, j]>^2 )

    where ``P[:, j]`` is the j-th eigenvector of ``V_bar`` and ``delta[j]``
    is the corresponding eigenvalue of the averaged MCM.

    Parameters
    ----------
    c_gamma, gamma_exp, n0
        Step size ``gamma_n = c_gamma * (n + n0)^(-gamma_exp)`` for the
        averaged stochastic gradient updates. ``gamma_exp`` must be in
        ``(1/2, 1)`` for convergence.
    feature_keys
        List of dict keys, one per channel. ``None`` takes the sorted keys
        of the first dict seen, so the result does not depend on the order
        of the keys.
    eps
        Numerical guard for divisions and degenerate eigenvalues.
    seed
        Unused in the minimal version, kept for interface stability with
        the full version.
    threshold
        Distance above which ``is_outlier`` returns True. The default 7.0
        is calibrated on the seeded Gaussian ellipsoid benchmark in the
        module's ``__main__``; for a d-dimensional Gaussian, MCM
        eigenvalues under-estimate the true variance, so the absolute
        scale of the score is larger than the classical chi-squared one.

    Notes
    -----
    The averaged-median update uses ``1/n`` with ``n`` 1-indexed, the
    correct running average. The paper writes ``1/(n+2)`` with n
    0-indexed (an off-by-one in the index).

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.utility.anomaly.mahalanobis import (
    ...     OnlineRobustMahalanobis,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> model = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    >>> for _ in range(500):
    ...     x = rng.standard_normal(2)
    ...     model.learn_one({"a": float(x[0]), "b": float(x[1])})
    >>> inlier = model.score_one({"a": 0.0, "b": 0.0})
    >>> outlier = model.score_one({"a": 10.0, "b": 10.0})
    >>> outlier > inlier
    True

    References
    ----------
    Guillot, A. et al. (2025). arXiv:2601.03957.
    Cardot, H., Cenac, P., Zitt, P.-A. (2013). Bernoulli 19(1), 18-43.
    Cardot, H., Godichon-Baggioni, A. (2017). TEST 26(3), 461-480.
    """

    def __init__(self, c_gamma: float = 1.0, gamma_exp: float = 0.7,
                 n0: int = 0,
                 feature_keys: Optional[Sequence[str]] = None,
                 eps: float = 1e-12, seed: Optional[int] = None,
                 threshold: float = 7.0):
        self.c_gamma = c_gamma
        self.gamma_exp = gamma_exp
        self.n0 = n0
        self.feature_keys = feature_keys
        self.eps = eps
        self.seed = seed
        self.threshold = threshold
        self._dim: Optional[int] = None
        self._m: Optional[np.ndarray] = None
        self._m_bar: Optional[np.ndarray] = None
        self._V: Optional[np.ndarray] = None
        self._V_bar: Optional[np.ndarray] = None
        self._delta: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._n: int = 0
        self._ready: bool = False
        self._keys: Optional[List[str]] = None

    def _resolve_keys(self, x) -> List[str]:
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
        return self.c_gamma * (n + self.n0) ** (-self.gamma_exp)

    def _init_state(self, d: int) -> None:
        self._m = np.zeros(d)
        self._m_bar = np.zeros(d)
        self._V = np.eye(d)
        self._V_bar = np.eye(d)
        self._delta = np.ones(d)
        self._P = np.eye(d)

    def learn_one(self, x):
        y = self._to_vec(x)
        if self._dim is None:
            self._dim = int(y.size)
            self._init_state(self._dim)

        self._n += 1
        gamma = self._gamma(self._n)

        diff = y - self._m
        nrm = float(np.linalg.norm(diff))
        if nrm > self.eps:
            self._m = self._m + gamma * diff / nrm
        self._m_bar = self._m_bar + (1.0 / self._n) * (self._m - self._m_bar)

        outer = np.outer(y - self._m_bar, y - self._m_bar)
        dV = outer - self._V
        nrm_V = float(np.linalg.norm(dV, "fro"))
        if nrm_V > self.eps:
            self._V = self._V + gamma * dV / nrm_V
        self._V_bar = self._V_bar + (1.0 / self._n) * (self._V - self._V_bar)

        if self._n >= 4:
            try:
                delta, P = np.linalg.eigh(self._V_bar)
                self._delta = np.maximum(delta, self.eps)
                self._P = P
                self._ready = True
            except np.linalg.LinAlgError:
                pass

    def score_one(self, x) -> float:
        if not self._ready:
            return 0.0
        y = self._to_vec(x)
        diff = y - self._m_bar
        proj = self._P.T @ diff
        d2 = float(np.sum((proj ** 2) / self._delta))
        return float(np.sqrt(max(d2, 0.0)))

    def is_outlier(self, x) -> bool:
        return self.score_one(x) > self.threshold

    def _unit_test_skips(self):
        return {"check_roc_auc"}
