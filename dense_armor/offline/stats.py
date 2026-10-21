"""Classic statistics: the exact quantity each online statistic or sketch computes.

Every class reads the whole dataset with ``fit``. The online class of
``dense_armor.utility.stats`` is the other side of the pair; the sketches
(quantiles, counts, membership) approximate the exact quantity that the
classic class returns.

References:
    Pearson correlation and the sample covariance with ``n - 1`` (textbook).
    West, D. H. D. (1979). Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only).
    Bodnar, T., Gupta, A. K., Parolya, N. (2014). Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq. (4.3) and (4.4),
        page 12 (not in the folder, bibliographic only).
    Masson, C., Rim, J. E., Lee, H. K. (2019). DDSketch; Dunning, T. (2019).
        Computing extremely accurate quantiles using t-digests; Cormode, G.,
        Muthukrishnan, S. (2005); Flajolet, P. et al. (2007); Bloom, B. H.
        (1970); Metwally, A. et al. (2005): the sketches, whose exact
        counterpart is the classic class (not in the folder, bibliographic
        only).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew

from dense_armor.offline.base import ClassicBase, finite, finite_pairs

__all__ = [
    "ClassicRunningCovariance",
    "ClassicRunningCorrelation",
    "ClassicRollingCovariance",
    "ClassicRollingCorrelation",
    "ClassicAutocorrelation",
    "ClassicRunningMoments",
    "ClassicRunningMomentsVector",
    "ClassicEWStats",
    "ClassicDDSketch",
    "ClassicTDigest",
    "ClassicRollingMedianVector",
    "ClassicOnlineShrinkageCovariance",
    "ClassicCountMinSketch",
    "ClassicHyperLogLog",
    "ClassicBloomFilter",
    "ClassicSpaceSaving",
]


def _cov(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.cov(a, b)[0, 1])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class ClassicRunningCovariance(ClassicBase):
    """Sample covariance (``n - 1``) and Pearson correlation of two channels.

    Args:
        x_feature: name of the first channel (kept for the online pair).
        y_feature: name of the second channel.

    Example:
        >>> m = ClassicRunningCovariance().fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        >>> round(m.cov_, 6), round(m.corr_, 6)
        (2.0, 1.0)
    """

    def __init__(self, x_feature: str = "x", y_feature: str = "y") -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature

    def fit(self, X, y=None):  # type: ignore[override]
        a, b, self.n_missing_ = finite_pairs(X, y)
        self.count_ = int(a.size)
        self.cov_ = _cov(a, b)
        self.corr_ = _corr(a, b)
        return self


class ClassicRunningCorrelation(ClassicRunningCovariance):
    """Pearson correlation of two channels over the whole dataset.

    Example:
        >>> ClassicRunningCorrelation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_
        -1.0
    """


class ClassicRollingCovariance(ClassicBase):
    """Sample covariance over the last ``window`` pairs and over every window.

    Args:
        x_feature: name of the first channel.
        y_feature: name of the second channel.
        window: number of pairs in a window.

    Example:
        >>> m = ClassicRollingCovariance(window=3).fit([1., 2., 3., 4.], [1., 2., 3., 5.])
        >>> m.cov_series_.shape
        (2,)
    """

    def __init__(
        self, x_feature: str = "x", y_feature: str = "y", window: int = 20
    ) -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.window = window

    def fit(self, X, y=None):  # type: ignore[override]
        a, b, self.n_missing_ = finite_pairs(X, y)
        w = self.window
        if a.size < w:
            self.cov_series_ = np.empty(0)
            self.corr_series_ = np.empty(0)
        else:
            wa = sliding_window_view(a, w)
            wb = sliding_window_view(b, w)
            ma = wa - wa.mean(axis=1, keepdims=True)
            mb = wb - wb.mean(axis=1, keepdims=True)
            cov = (ma * mb).sum(axis=1) / (w - 1)
            sa = np.sqrt((ma * ma).sum(axis=1) / (w - 1))
            sb = np.sqrt((mb * mb).sum(axis=1) / (w - 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where((sa > 0) & (sb > 0), cov / (sa * sb), np.nan)
            self.cov_series_ = cov
            self.corr_series_ = corr
        self.cov_ = (
            float(self.cov_series_[-1]) if self.cov_series_.size else float("nan")
        )
        self.corr_ = (
            float(self.corr_series_[-1]) if self.corr_series_.size else float("nan")
        )
        return self


class ClassicRollingCorrelation(ClassicRollingCovariance):
    """Pearson correlation over the last ``window`` pairs and over every window.

    Example:
        >>> m = ClassicRollingCorrelation(window=3).fit([1., 2., 3.], [1., 2., 3.])
        >>> round(m.corr_, 6)
        1.0
    """


class ClassicAutocorrelation(ClassicBase):
    """Sample autocorrelation at a fixed lag: the standard estimator.

    ``r_k = sum_t (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2`` with ``m`` the
    mean of the whole series (textbook).

    Args:
        feature: name of the channel (kept for the online pair).
        lag: the lag, at least 1.

    Example:
        >>> round(ClassicAutocorrelation(lag=1).fit([1., 2., 1., 2., 1.]).value_, 4)
        -0.8
    """

    def __init__(self, feature: str = "x", lag: int = 1) -> None:
        self.feature = feature
        self.lag = lag

    def fit(self, X, y=None):  # type: ignore[override]
        x, self.n_missing_ = finite(X)
        k = self.lag
        if x.size <= k:
            self.value_ = float("nan")
            return self
        d = x - x.mean()
        den = float((d * d).sum())
        self.value_ = float((d[:-k] * d[k:]).sum() / den) if den > 0 else float("nan")
        self.n_pairs_ = int(x.size - k)
        return self


class ClassicRunningMoments(ClassicBase):
    """Mean, variance (``n - 1``), skewness, excess kurtosis, min, max, range.

    Example:
        >>> m = ClassicRunningMoments().fit([1.0, 2.0, 3.0, 4.0])
        >>> m.mean_, round(m.var_, 4), m.ptp_
        (2.5, 1.6667, 3.0)
    """

    def __init__(self, feature: str | None = None) -> None:
        self.feature = feature

    def fit(self, X, y=None):  # type: ignore[override]
        x, self.n_missing_ = finite(X)
        self.count_ = int(x.size)
        if x.size == 0:
            self.mean_ = self.var_ = self.std_ = float("nan")
            self.skewness_ = self.kurtosis_ = float("nan")
            self.min_ = self.max_ = self.ptp_ = float("nan")
            return self
        self.mean_ = float(x.mean())
        self.var_ = float(x.var(ddof=1)) if x.size > 1 else float("nan")
        self.std_ = float(np.sqrt(self.var_)) if x.size > 1 else float("nan")
        self.skewness_ = float(_skew(x)) if x.size > 2 and x.std() > 0 else float("nan")
        self.kurtosis_ = float(_kurt(x)) if x.size > 3 and x.std() > 0 else float("nan")
        self.min_ = float(x.min())
        self.max_ = float(x.max())
        self.ptp_ = self.max_ - self.min_
        return self


class ClassicRunningMomentsVector(ClassicBase):
    """The moments of every column of a matrix.

    Example:
        >>> m = ClassicRunningMomentsVector().fit([[1., 10.], [3., 20.]])
        >>> m.mean_.tolist()
        [2.0, 15.0]
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        cols = [ClassicRunningMoments().fit(A[:, j]) for j in range(A.shape[1])]
        self.mean_ = np.array([c.mean_ for c in cols])
        self.var_ = np.array([c.var_ for c in cols])
        self.skewness_ = np.array([c.skewness_ for c in cols])
        self.kurtosis_ = np.array([c.kurtosis_ for c in cols])
        self.min_ = np.array([c.min_ for c in cols])
        self.max_ = np.array([c.max_ for c in cols])
        return self


class ClassicEWStats(ClassicBase):
    """Exponentially weighted mean and variance of the whole series.

    The recursion of West (1979), ``mu_t = mu_{t-1} + a d_t``,
    ``v_t = (1 - a)(v_{t-1} + a d_t^2)`` with ``d_t = x_t - mu_{t-1}``,
    started at ``mu_0 = x_0`` and ``v_0 = 0``, run as two linear filters on
    the whole array.

    Args:
        alpha: the weight of the new sample, in ``(0, 1]``.
        feature: name of the channel (kept for the online pair).

    Example:
        >>> round(ClassicEWStats(alpha=0.5).fit([1.0, 3.0]).mean_, 4)
        2.0
    """

    def __init__(self, alpha: float = 0.1, feature: str | None = None) -> None:
        self.alpha = alpha
        self.feature = feature

    def fit(self, X, y=None):  # type: ignore[override]
        x, self.n_missing_ = finite(X)
        self.count_ = int(x.size)
        if x.size == 0:
            self.mean_ = self.var_ = self.std_ = float("nan")
            return self
        a = self.alpha
        xs = x - x[0]
        mu = x[0] + lfilter([a], [1.0, -(1.0 - a)], xs, zi=[0.0])[0]
        mu_prev = np.concatenate([[x[0]], mu[:-1]])
        d = x - mu_prev
        d[0] = 0.0
        v = lfilter([(1.0 - a) * a], [1.0, -(1.0 - a)], d * d)
        self.mean_ = float(mu[-1])
        self.var_ = float(v[-1])
        self.std_ = float(np.sqrt(self.var_))
        return self


class _ExactQuantile(ClassicBase):
    """Exact quantiles of the whole dataset, the target of a quantile sketch."""

    def __init__(self, feature: str | None = None) -> None:
        self.feature = feature

    def fit(self, X, y=None):  # type: ignore[override]
        x, self.n_missing_ = finite(X)
        self.sorted_ = np.sort(x)
        self.count_ = int(x.size)
        self.min_ = float(x.min()) if x.size else float("nan")
        self.max_ = float(x.max()) if x.size else float("nan")
        return self

    def quantile(self, q: float) -> float:
        """Exact lower quantile ``q`` in ``[0, 1]``: the value of rank ``floor(q (n - 1))``."""
        if self.sorted_.size == 0:
            return float("nan")
        return float(np.quantile(self.sorted_, q, method="lower"))


class ClassicDDSketch(_ExactQuantile):
    """Exact quantiles: what DDSketch (Masson et al. 2019) approximates.

    Args:
        alpha: relative error of the online sketch (kept for the pair).
        max_buckets: bucket cap of the online sketch (kept for the pair).
        feature: name of the channel.

    Example:
        >>> ClassicDDSketch().fit([1.0, 2.0, 3.0]).quantile(0.5)
        2.0
    """

    def __init__(
        self, alpha: float = 0.01, max_buckets: int = 2048, feature: str | None = None
    ) -> None:
        super().__init__(feature)
        self.alpha = alpha
        self.max_buckets = max_buckets


class ClassicTDigest(_ExactQuantile):
    """Exact quantiles: what the t-digest (Dunning 2019) approximates.

    Args:
        delta: compression of the online digest (kept for the pair).
        buffer_size: buffer of the online digest (kept for the pair).
        feature: name of the channel.

    Example:
        >>> ClassicTDigest().fit([1.0, 2.0, 3.0, 4.0]).quantile(0.5)
        2.0
    """

    def __init__(
        self,
        delta: float = 100.0,
        buffer_size: int | None = None,
        feature: str | None = None,
    ) -> None:
        super().__init__(feature)
        self.delta = delta
        self.buffer_size = buffer_size


class ClassicRollingMedianVector(ClassicBase):
    """Median over every window and over the last window, per column.

    Args:
        window: number of rows in a window.
        features: names of the columns (kept for the online pair).

    Example:
        >>> m = ClassicRollingMedianVector(window=3).fit([1., 5., 2., 8.])
        >>> m.median_.tolist()
        [5.0]
    """

    def __init__(self, window: int = 20, features: list[str] | None = None) -> None:
        self.window = window
        self.features = features

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        w = self.window
        if A.shape[0] < w:
            self.series_ = np.empty((0, A.shape[1]))
        else:
            self.series_ = np.median(sliding_window_view(A, w, axis=0), axis=-1)
        self.median_ = (
            self.series_[-1] if self.series_.shape[0] else np.full(A.shape[1], np.nan)
        )
        return self


class ClassicOnlineShrinkageCovariance(ClassicBase):
    """Linear shrinkage toward ``(1/p) I`` on the sample covariance of all rows.

    Eq. (4.3) and (4.4), page 12 of Bodnar, Gupta and Parolya (2014):
    ``alpha = 1 - (tr(S)^2 / n) / (tr(S^2) - tr(S)^2 / p)`` and
    ``beta = (1 - alpha) tr(S)``; the shrunk matrix is
    ``alpha S + beta (1/p) I``.

    Args:
        n_features: number of columns ``p``.
        feature_prefix: prefix of the column names (kept for the pair).

    Example:
        >>> import numpy as np
        >>> X = np.random.default_rng(0).normal(size=(30, 3))
        >>> ClassicOnlineShrinkageCovariance(3).fit(X).shrunk_.shape
        (3, 3)
    """

    def __init__(self, n_features: int, feature_prefix: str = "v") -> None:
        self.n_features = n_features
        self.feature_prefix = feature_prefix

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        self.n_seen_ = int(A.shape[0])
        p = self.n_features
        s = np.cov(A.T) if A.shape[0] > 1 else np.zeros((p, p))
        n = max(self.n_seen_, 1)
        tr_s = float(np.trace(s))
        tr_s2 = float(np.sum(s * s))
        denom = tr_s2 - tr_s * tr_s / p
        if denom <= 0.0:
            self.alpha_, self.beta_ = 1.0, 0.0
        else:
            self.alpha_ = 1.0 - (tr_s * tr_s / n) / denom
            self.beta_ = (1.0 - self.alpha_) * tr_s
        self.sample_ = s
        self.shrunk_ = self.alpha_ * s + self.beta_ * np.eye(p) / p
        return self


class ClassicCountMinSketch(ClassicBase):
    """Exact item counts: what the Count-Min sketch approximates.

    Args:
        epsilon: error of the online sketch (kept for the pair).
        delta: failure probability of the online sketch (kept for the pair).
        feature: name of the channel.
        seed: seed of the online sketch (kept for the pair).

    Example:
        >>> ClassicCountMinSketch().fit(["a", "b", "a"]).estimate("a")
        2
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        delta: float = 0.01,
        feature: str | None = None,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.feature = feature
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        self.counts_ = Counter(X)
        self.count_ = int(sum(self.counts_.values()))
        return self

    def estimate(self, item: Any) -> int:
        """Exact number of occurrences of ``item``."""
        return int(self.counts_.get(item, 0))


class ClassicHyperLogLog(ClassicBase):
    """Exact number of distinct items: what HyperLogLog approximates.

    Args:
        precision: precision of the online sketch (kept for the pair).
        feature: name of the channel.
        seed: seed of the online sketch (kept for the pair).

    Example:
        >>> ClassicHyperLogLog().fit(["a", "b", "a"]).count_distinct_
        2
    """

    def __init__(
        self, precision: int = 14, feature: str | None = None, seed: int = 42
    ) -> None:
        self.precision = precision
        self.feature = feature
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        items = list(X)
        self.count_total_ = len(items)
        self.count_distinct_ = len(set(items))
        return self


class ClassicBloomFilter(ClassicBase):
    """Exact set membership: what the Bloom filter approximates.

    Args:
        n_expected: capacity of the online filter (kept for the pair).
        p_false: false-positive rate of the online filter (kept for the pair).
        feature: name of the channel.
        seed: seed of the online filter (kept for the pair).

    Example:
        >>> ClassicBloomFilter().fit(["a"]).check("b")
        False
    """

    def __init__(
        self,
        n_expected: int = 1000,
        p_false: float = 0.01,
        feature: str | None = None,
        seed: int = 42,
    ) -> None:
        self.n_expected = n_expected
        self.p_false = p_false
        self.feature = feature
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        self.set_ = set(X)
        self.count_ = len(self.set_)
        return self

    def check(self, item: Any) -> bool:
        """True if ``item`` was in the dataset."""
        return item in self.set_


class ClassicSpaceSaving(ClassicBase):
    """Exact most frequent items: what Space-Saving approximates.

    Args:
        k: number of items kept by the online algorithm.
        feature: name of the channel.

    Example:
        >>> ClassicSpaceSaving(k=1).fit(["a", "b", "a"]).top()
        [('a', 2)]
    """

    def __init__(self, k: int = 10, feature: str | None = None) -> None:
        self.k = k
        self.feature = feature

    def fit(self, X, y=None):  # type: ignore[override]
        self.counts_ = Counter(X)
        self.count_ = int(sum(self.counts_.values()))
        return self

    def estimate(self, item: Any) -> int:
        """Exact number of occurrences of ``item``."""
        return int(self.counts_.get(item, 0))

    def top(self, n: int | None = None) -> list[tuple[Any, int]]:
        """The ``n`` (default ``k``) most frequent items with exact counts."""
        return self.counts_.most_common(self.k if n is None else n)
