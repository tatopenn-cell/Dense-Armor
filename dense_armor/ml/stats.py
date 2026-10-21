"""Statistics computed on the whole table: covariance, moments, medians, shrinkage.

Every class reads the whole dataset with ``fit``.

References:
    Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae and a pairwise
        algorithm for computing sample variances, Compstat: the two-pass formulas for the
        mean, the variance and the covariance (not in the folder, bibliographic only).
    West, D. H. D. (1979). Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only).
    Bodnar, T., Gupta, A. K., Parolya, N. (2014). Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq. (4.3) and (4.4),
        page 12 (not in the folder, bibliographic only).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew

from dense_armor.ml.base import MLBase, finite, finite_pairs

STOPPED = {
    "DDSketch": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and the exact quantile is not the algorithm of Masson et al.",
    "TDigest": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and the exact quantile is not the algorithm of Dunning and Ertl",
    "CountMinSketch": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and the exact count is not the algorithm of Cormode and Muthukrishnan",
    "HyperLogLog": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and the exact distinct count is not the algorithm of Flajolet et al.",
    "BloomFilter": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and exact membership is not the filter of Bloom",
    "SpaceSaving": "a sketch does not depend on the order of the rows: its form on the whole table is the online class itself, and the exact counts are not the algorithm of Metwally et al.",
}

__all__ = [
    "Covariance",
    "Correlation",
    "RollingCovariance",
    "RollingCorrelation",
    "Autocorrelation",
    "Moments",
    "ColumnMoments",
    "EWStats",
    "RollingMedian",
    "ShrinkageCovariance",
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


class Covariance(MLBase):
    """Sample covariance (``n - 1``) and Pearson correlation of two channels.

    Args:
        x_feature: name of the first channel (kept for the online pair).
        y_feature: name of the second channel.

    Example:
        >>> m = Covariance().fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
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


class Correlation(Covariance):
    """Pearson correlation of two channels over the whole dataset.

    Example:
        >>> Correlation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_
        -1.0
    """


class RollingCovariance(MLBase):
    """Sample covariance over the last ``window`` pairs and over every window.

    Args:
        x_feature: name of the first channel.
        y_feature: name of the second channel.
        window: number of pairs in a window.

    Example:
        >>> m = RollingCovariance(window=3).fit([1., 2., 3., 4.], [1., 2., 3., 5.])
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


class RollingCorrelation(RollingCovariance):
    """Pearson correlation over the last ``window`` pairs and over every window.

    Example:
        >>> m = RollingCorrelation(window=3).fit([1., 2., 3.], [1., 2., 3.])
        >>> round(m.corr_, 6)
        1.0
    """


class Autocorrelation(MLBase):
    """Sample autocorrelation at a fixed lag: the standard estimator.

    ``r_k = sum_t (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2`` with ``m`` the
    mean of the whole series (textbook).

    Args:
        feature: name of the channel (kept for the online pair).
        lag: the lag, at least 1.

    Example:
        >>> round(Autocorrelation(lag=1).fit([1., 2., 1., 2., 1.]).value_, 4)
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


class Moments(MLBase):
    """Mean, variance (``n - 1``), skewness, excess kurtosis, min, max, range.

    Example:
        >>> m = Moments().fit([1.0, 2.0, 3.0, 4.0])
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


class ColumnMoments(MLBase):
    """The moments of every column of a matrix.

    Example:
        >>> m = ColumnMoments().fit([[1., 10.], [3., 20.]])
        >>> m.mean_.tolist()
        [2.0, 15.0]
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        cols = [Moments().fit(A[:, j]) for j in range(A.shape[1])]
        self.mean_ = np.array([c.mean_ for c in cols])
        self.var_ = np.array([c.var_ for c in cols])
        self.skewness_ = np.array([c.skewness_ for c in cols])
        self.kurtosis_ = np.array([c.kurtosis_ for c in cols])
        self.min_ = np.array([c.min_ for c in cols])
        self.max_ = np.array([c.max_ for c in cols])
        return self


class EWStats(MLBase):
    """Exponentially weighted mean and variance of the whole series.

    The recursion of West (1979), ``mu_t = mu_{t-1} + a d_t``,
    ``v_t = (1 - a)(v_{t-1} + a d_t^2)`` with ``d_t = x_t - mu_{t-1}``,
    started at ``mu_0 = x_0`` and ``v_0 = 0``, run as two linear filters on
    the whole array.

    Args:
        alpha: the weight of the new sample, in ``(0, 1]``.
        feature: name of the channel (kept for the online pair).

    Example:
        >>> round(EWStats(alpha=0.5).fit([1.0, 3.0]).mean_, 4)
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


class RollingMedian(MLBase):
    """Median over every window and over the last window, per column.

    Args:
        window: number of rows in a window.
        features: names of the columns (kept for the online pair).

    Example:
        >>> m = RollingMedian(window=3).fit([1., 5., 2., 8.])
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


class ShrinkageCovariance(MLBase):
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
        >>> ShrinkageCovariance(3).fit(X).shrunk_.shape
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
