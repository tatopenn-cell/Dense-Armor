# -*- coding: utf-8 -*-
"""
utility/streaming_filters.py
=============================
Causal (one sample at a time) versions of the four batch robust filters in
`robust_filters.py`: Hampel, Tukey fences, Chauvenet, iterative sigma
clipping. The batch filters look at a window *centred* on each point -- they
see the future -- which a robot's real-time control loop cannot do. Each
scorer here looks only at the `2 * radius` samples strictly *preceding* the
value being scored (the causal window) and scores the new value against the
robust centre/scale of that window, in the method's own scale.

The four scorers share the interface of
`river_anomaly.StreamingDeviationScorer` (`learn_one` / `score_one`, feature
selection, `is_outlier`), so they plug into river pipelines and
`anomaly.ThresholdFilter` the same way, but each uses its own method's scale
and threshold rather than the median/MAD robust deviation of
`StreamingDeviationDetector` alone.

`HampelFilter` is a transformer: `learn_one(x)` then `transform_one(x)`
returns the value, or the causal-window median when the value is an outlier
(the same replacement rule as the batch `hampel_filter`).

Conventions: see `SKILL.md`. Causal window = `2 * radius` samples strictly
before the value being scored; the value is never part of its own window.
See `ISTRUZIONI.md`, section "Important: centred vs causal windows".
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from dense_armor.base import AnomalyDetector, Transformer


def _scaled_mad(w: np.ndarray) -> float:
    """1.4826 * MAD of `w` (consistency constant for a Gaussian)."""
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    return 1.4826 * mad


def _clean_stats(w: np.ndarray, n_sigmas: float, max_iters: int = 5,
                 eps: float = 1e-12):
    """Iterative sigma-clipping on `w`: mean and std of the surviving points."""
    mask = np.ones(w.shape, dtype=bool)
    for _ in range(max_iters):
        subset = w[mask]
        if subset.size < 2:
            break
        mu = float(np.mean(subset))
        sigma = float(np.std(subset))
        if sigma < eps:
            break
        new_mask = np.abs(w - mu) <= n_sigmas * sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    subset = w[mask]
    if subset.size < 1:
        return float(np.mean(w)), float(np.std(w))
    return float(np.mean(subset)), float(np.std(subset))


class HampelScorer(AnomalyDetector):
    """Hampel outlier scorer on the causal window.

    The score is ``|x - med| / (1.4826 * MAD)``, where ``med`` and ``MAD``
    are the median and median absolute deviation of the ``2 * radius``
    samples strictly preceding the value being scored. This is the same
    formula as ``robust_filters.hampel_filter`` (Hampel 1974; Davies and
    Gather 1993), applied to the causal window instead of the centred one.
    ``is_outlier`` returns True when the score exceeds ``n_sigmas``.

    Parameters
    ----------
    radius
        Half-width unit; the causal window holds ``2 * radius`` samples.
    n_sigmas
        Threshold in scaled-MAD units (default 3.0, the robust "3 sigma" rule).
    feature
        Key of the feature to watch; ``None`` takes the feature with the
        smallest key, so the result does not depend on the order of the keys.
    eps
        Below this scale the window is flat, and the score is ``+inf`` when
        the value actually deviates from the window median (degenerate scale
        is the strongest possible evidence), ``0.0`` when it does not.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.filters import HampelScorer
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = HampelScorer(radius=5, n_sigmas=3.0)
    >>> for v in rng.normal(0, 1, 20):
    ...     _ = s.learn_one({"v": float(v)})
    >>> s.score_one({"v": 50.0}) > 3.0
    True
    >>> s.is_outlier({"v": 50.0})
    True

    References
    ----------
    Hampel, F. R. (1974). The influence curve and its role in robust
    estimation. JASA 69(346), 383-393.
    Davies, L., Gather, U. (1993). The identification of multiple outliers.
    JASA 88(423), 782-792.
    """

    def __init__(self, radius: int = 10, n_sigmas: float = 3.0,
                 feature: str | None = None, eps: float = 1e-9):
        self.radius = radius
        self.n_sigmas = n_sigmas
        self.feature = feature
        self.eps = eps
        self._buffer: deque = deque(maxlen=2 * radius)

    def _value(self, x):
        return float(x[self.feature if self.feature is not None else min(x)])

    def learn_one(self, x):
        self._buffer.append(self._value(x))

    def score_one(self, x):
        v = self._value(x)
        if len(self._buffer) < 4:
            return 0.0
        w = np.fromiter(self._buffer, dtype=float)
        med = float(np.median(w))
        scale = _scaled_mad(w)
        dev = abs(v - med)
        if scale < self.eps:
            return float("inf") if dev > self.eps else 0.0
        return dev / scale

    def is_outlier(self, x) -> bool:
        return self.score_one(x) > self.n_sigmas

    def _unit_test_skips(self):
        return {"check_roc_auc"}


class TukeyScorer(AnomalyDetector):
    """Tukey-fences outlier scorer on the causal window.

    The score is how far the value lies beyond the fences
    ``[Q1 - k * IQR, Q3 + k * IQR]`` of the ``2 * radius`` samples strictly
    preceding it, expressed in IQR units: 0 inside the fences, positive
    outside. This is the batch ``robust_filters.tukey_fences`` rule
    (Tukey 1977) applied to the causal window: ``is_outlier`` returns True
    exactly when the batch would flag the point, since "beyond the fences"
    is score > 0. On a flat window (IQR == 0) the score is ``+inf`` when the
    value differs from the degenerate fence and ``0.0`` when it does not,
    matching the batch rule for a flat window.

    Parameters
    ----------
    radius
        Half-width unit; the causal window holds ``2 * radius`` samples.
    k
        Fence multiplier (default 1.5, the Tukey standard; 3.0 for
        "extreme outliers").
    feature
        Key of the feature to watch; ``None`` takes the smallest key.
    eps
        Below this IQR the window is treated as flat.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.filters import TukeyScorer
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = TukeyScorer(radius=5)
    >>> for v in rng.normal(0, 1, 20):
    ...     _ = s.learn_one({"v": float(v)})
    >>> s.is_outlier({"v": 50.0})
    True

    References
    ----------
    Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    """

    def __init__(self, radius: int = 10, k: float = 1.5,
                 feature: str | None = None, eps: float = 1e-9):
        self.radius = radius
        self.k = k
        self.feature = feature
        self.eps = eps
        self._buffer: deque = deque(maxlen=2 * radius)

    def _value(self, x):
        return float(x[self.feature if self.feature is not None else min(x)])

    def learn_one(self, x):
        self._buffer.append(self._value(x))

    def score_one(self, x):
        v = self._value(x)
        if len(self._buffer) < 4:
            return 0.0
        w = np.fromiter(self._buffer, dtype=float)
        q1, q3 = np.percentile(w, [25, 75])
        iqr = float(q3 - q1)
        if iqr < self.eps:
            return float("inf") if (v < q1 or v > q3) else 0.0
        lo = float(q1) - self.k * iqr
        hi = float(q3) + self.k * iqr
        if v < lo:
            return (lo - v) / iqr
        if v > hi:
            return (v - hi) / iqr
        return 0.0

    def is_outlier(self, x) -> bool:
        return self.score_one(x) > 0.0

    def _unit_test_skips(self):
        return {"check_roc_auc"}


class ChauvenetScorer(AnomalyDetector):
    """Chauvenet outlier scorer on the causal window.

    The score is ``N * P(|Z| >= z)``, the expected number of observations
    that would lie at least ``z = |x - mu| / sigma`` away from the mean if
    the ``2 * radius`` samples strictly preceding it were a Gaussian sample
    of size ``N``. This is the criterion of Chauvenet (1863), the same as
    the batch ``robust_filters.chauvenet_criterion``, applied to the causal
    window.

    Direction of the test
    ---------------------
    Unlike the other three scorers, Chauvenet *rejects when the score is
    small*: ``N * P`` counts how many points of a Gaussian sample would be
    at least this far out, and the criterion says "reject when this count
    is below 0.5" -- i.e. ``is_outlier`` returns ``score < threshold`` with
    ``threshold = 0.5``. This is the natural direction of the formula and
    is documented here because it is the opposite of the "high score =
    anomalous" convention of the other scorers.

    ``mu`` and ``sigma`` are non-robust (mean and std), exactly as in the
    original criterion; on a window that already contains other outliers
    they are already distorted before the central point is judged. That is
    a known limitation of the original criterion and not specific to this
    implementation.

    Parameters
    ----------
    radius
        Half-width unit; the causal window holds ``2 * radius`` samples.
    threshold
        Rejection threshold on ``N * P`` (default 0.5, Chauvenet's value).
    feature
        Key of the feature to watch; ``None`` takes the smallest key.
    eps
        Below this scale the window is treated as flat.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.filters import ChauvenetScorer
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = ChauvenetScorer(radius=10)
    >>> for v in rng.normal(0, 1, 20):
    ...     _ = s.learn_one({"v": float(v)})
    >>> s.is_outlier({"v": 50.0})
    True

    References
    ----------
    Chauvenet, W. (1863). A Manual of Spherical and Practical Astronomy,
    Vol. II, Appendix on the Method of Least Squares.
    """

    def __init__(self, radius: int = 10, threshold: float = 0.5,
                 feature: str | None = None, eps: float = 1e-9):
        self.radius = radius
        self.threshold = threshold
        self.feature = feature
        self.eps = eps
        self._buffer: deque = deque(maxlen=2 * radius)

    def _value(self, x):
        return float(x[self.feature if self.feature is not None else min(x)])

    def learn_one(self, x):
        self._buffer.append(self._value(x))

    def score_one(self, x):
        v = self._value(x)
        if len(self._buffer) < 4:
            return float("inf")
        w = np.fromiter(self._buffer, dtype=float)
        n = w.size
        mu = float(np.mean(w))
        sigma = float(np.std(w))
        dev = abs(v - mu)
        if sigma < self.eps:
            return float(n) if dev <= self.eps else 0.0
        z = dev / sigma
        p = math.erfc(z / math.sqrt(2.0))
        return n * p

    def is_outlier(self, x) -> bool:
        return self.score_one(x) < self.threshold

    def _unit_test_skips(self):
        return {"check_roc_auc"}


class SigmaClipScorer(AnomalyDetector):
    """Iterative sigma-clipping outlier scorer on the causal window.

    On the ``2 * radius`` samples strictly preceding the value, the mean and
    standard deviation are recomputed after removing points beyond
    ``n_sigmas``, until stable or ``max_iters``; the value is then scored by
    ``|x - mu_clean| / sigma_clean``. This is the same rule as the batch
    ``robust_filters.sigma_clip`` (standard in astronomy), applied to the
    causal window. ``is_outlier`` returns True when the score exceeds
    ``n_sigmas``. On a window whose cleaned scale is degenerate, the score
    is ``+inf`` when the value deviates from the cleaned mean and ``0.0``
    when it does not.

    Parameters
    ----------
    radius
        Half-width unit; the causal window holds ``2 * radius`` samples.
    n_sigmas
        Clipping threshold and outlier threshold (default 3.0).
    max_iters
        Maximum number of clipping iterations (default 5).
    feature
        Key of the feature to watch; ``None`` takes the smallest key.
    eps
        Below this scale the cleaned window is treated as flat.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.filters import SigmaClipScorer
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = SigmaClipScorer(radius=10, n_sigmas=3.0)
    >>> for v in rng.normal(0, 1, 20):
    ...     _ = s.learn_one({"v": float(v)})
    >>> s.is_outlier({"v": 50.0})
    True

    References
    ----------
    Standard in astronomy; see e.g. Akritas, M. G., Bershady, M. A. (1996).
    Linear regression for astronomical data with measurement errors and
    intrinsic scatter. ApJ 470, 706.
    """

    def __init__(self, radius: int = 10, n_sigmas: float = 3.0,
                 max_iters: int = 5, feature: str | None = None,
                 eps: float = 1e-9):
        self.radius = radius
        self.n_sigmas = n_sigmas
        self.max_iters = max_iters
        self.feature = feature
        self.eps = eps
        self._buffer: deque = deque(maxlen=2 * radius)

    def _value(self, x):
        return float(x[self.feature if self.feature is not None else min(x)])

    def learn_one(self, x):
        self._buffer.append(self._value(x))

    def score_one(self, x):
        v = self._value(x)
        if len(self._buffer) < 4:
            return 0.0
        w = np.fromiter(self._buffer, dtype=float)
        mu, sigma = _clean_stats(w, self.n_sigmas, self.max_iters, self.eps)
        dev = abs(v - mu)
        if sigma < self.eps:
            return float("inf") if dev > self.eps else 0.0
        return dev / sigma

    def is_outlier(self, x) -> bool:
        return self.score_one(x) > self.n_sigmas

    def _unit_test_skips(self):
        return {"check_roc_auc"}


class HampelFilter(Transformer):
    """Streaming Hampel filter: replaces outliers with the causal-window median.

    ``transform_one`` returns the value itself when it is not an outlier
    (same rule as :class:`HampelScorer`), else the median of the
    ``2 * radius`` samples strictly preceding it. This is the causal
    equivalent of the batch ``robust_filters.hampel_filter`` replacement
    rule. Other keys in the input dict are passed through unchanged.

    Parameters
    ----------
    radius
        Half-width unit; the causal window holds ``2 * radius`` samples.
    n_sigmas
        Threshold in scaled-MAD units (default 3.0).
    feature
        Key of the feature to filter; ``None`` takes the smallest key.
    eps
        Below this scale the window is flat.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.filters import HampelFilter
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = HampelFilter(radius=5)
    >>> for v in rng.normal(0, 1, 20):
    ...     _ = f.learn_one({"v": float(v)})
    >>> out = f.transform_one({"v": 50.0})
    >>> abs(out["v"]) < 5.0
    True
    """

    def __init__(self, radius: int = 10, n_sigmas: float = 3.0,
                 feature: str | None = None, eps: float = 1e-9):
        self.radius = radius
        self.n_sigmas = n_sigmas
        self.feature = feature
        self.eps = eps
        self._scorer = HampelScorer(radius=radius, n_sigmas=n_sigmas,
                                    feature=feature, eps=eps)

    def _key(self, x):
        return self.feature if self.feature is not None else min(x)

    def learn_one(self, x):
        self._scorer.learn_one(x)

    def transform_one(self, x):
        key = self._key(x)
        v = float(x[key])
        if len(self._scorer._buffer) < 4:
            return {**x, key: v}
        if not self._scorer.is_outlier(x):
            return {**x, key: v}
        w = np.fromiter(self._scorer._buffer, dtype=float)
        return {**x, key: float(np.median(w))}

    def _unit_test_skips(self):
        return {"check_roc_auc"}
