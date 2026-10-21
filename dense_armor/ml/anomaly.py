"""Anomaly detection on a whole series or a whole table.

References:
    Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000). LOF:
        Identifying Density-Based Local Outliers. In Proceedings of the 2000
        ACM SIGMOD International Conference on Management of Data, pages
        93-104. Definitions 3 to 7 in Section 4 (not in the folder,
        bibliographic only).
    Chauvenet (1863), Tukey (1977), Hampel (1974), Davies and Gather (1993):
        the four rules on a centred local window, as the docstrings of the
        batch functions of the library write them (not in the folder,
        bibliographic only).
"""

from __future__ import annotations

import math

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = [
    "ChauvenetScorer",
    "HampelFilter",
    "LocalOutlierFactor",
    "SigmaClipScorer",
    "TukeyScorer",
]

_STREAM = "the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: it holds no algorithm for the whole dataset"
STOPPED = {
    "StreamingDeviationScorer": _STREAM,
    "StreamingDeviationDetector": _STREAM,
    "MultiChannelStreamingDeviationDetector": _STREAM,
    "HampelScorer": "its batch form is HampelFilter, which scores every value on the centred window",
    "HalfSpaceTrees": _STREAM,
    "LODA": _STREAM,
    "OneClassSGD": _STREAM,
    "OnlineIsolationForest": _STREAM,
    "OnlineRobustMahalanobis": "the whole-sample estimator is Section A.2 of Guillot et al. (2025), which is not in the folder; the library initialises the dispersion with the average of the centred outer products as a declared choice, and the rest is stochastic gradient on a stream",
}


def _window(x: np.ndarray, i: int, radius: int) -> np.ndarray:
    return x[max(0, i - radius) : min(len(x), i + radius + 1)]


class _Windowed(MLBase):
    """A rule applied to the centred window of radius ``radius`` of every value."""

    radius: int

    def _flag(self, x: np.ndarray, i: int) -> float | None:
        """The replacement of x[i] when it is an outlier, else ``None``."""
        raise NotImplementedError

    def fit(self, X, y=None):  # type: ignore[override]
        x = np.asarray(X, dtype=float).ravel()
        self.cleaned_ = x.copy()
        anomalies = []
        for i in range(len(x)):
            r = self._flag(x, i)
            if r is not None:
                anomalies.append(i)
                self.cleaned_[i] = r
        self.anomalies_ = anomalies
        return self


class HampelFilter(_Windowed):
    """Hampel filter on the centred window: median and scaled MAD.

    For every value the window holds the values at distance at most ``radius``;
    with ``med`` its median and ``scaled_MAD = 1.4826 * MAD``, the value is an
    outlier when ``|x - med| > n_sigmas * scaled_MAD``. A window with fewer than
    3 values is skipped; when the scaled MAD is below ``1e-12`` the value is an
    outlier when it differs from ``med`` by more than ``1e-9``. An outlier is
    replaced by ``med`` in ``cleaned_``.

    Args:
        radius: half-width of the window.
        n_sigmas: threshold in scaled-MAD units.

    Example:
        >>> m = HampelFilter(radius=2).fit([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        >>> m.anomalies_, m.cleaned_[4]
        ([4], np.float64(1.0))
    """

    def __init__(self, radius: int = 10, n_sigmas: float = 3.0) -> None:
        self.radius = radius
        self.n_sigmas = n_sigmas

    def _flag(self, x, i):
        w = _window(x, i, self.radius)
        if w.size < 3:
            return None
        med = float(np.median(w))
        s = 1.4826 * float(np.median(np.abs(w - med)))
        if s < 1e-12:
            return med if abs(x[i] - med) > 1e-9 else None
        return med if abs(x[i] - med) > self.n_sigmas * s else None


class TukeyScorer(_Windowed):
    """Tukey fences on the centred window: ``[Q1 - k IQR, Q3 + k IQR]``.

    A value outside the fences of its window is an outlier; a window with fewer
    than 4 values is skipped. An outlier is replaced by the median of its window.

    Args:
        radius: half-width of the window.
        k: fence multiplier (1.5 standard, 3.0 for extreme outliers).

    Example:
        >>> m = TukeyScorer(radius=2).fit([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        >>> m.anomalies_
        [4]
    """

    def __init__(self, radius: int = 10, k: float = 1.5) -> None:
        self.radius = radius
        self.k = k

    def _flag(self, x, i):
        w = _window(x, i, self.radius)
        if w.size < 4:
            return None
        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1
        if x[i] < q1 - self.k * iqr or x[i] > q3 + self.k * iqr:
            return float(np.median(w))
        return None


class ChauvenetScorer(_Windowed):
    """Chauvenet criterion on the centred window.

    With the window as a sample of size ``N``, mean ``mu`` and standard deviation
    ``sigma``, ``d = |x - mu| / sigma`` and ``P = erfc(d / sqrt(2))``; the value
    is rejected when ``N * P < 0.5``. A window with fewer than 3 values or with
    ``sigma < 1e-12`` is skipped. An outlier is replaced by the window median.

    Args:
        radius: half-width of the window.

    Example:
        >>> ChauvenetScorer(radius=4).fit([0.0, 0.0, 0.0, 0.0, 8.0]).anomalies_
        [4]
    """

    def __init__(self, radius: int = 10) -> None:
        self.radius = radius

    def _flag(self, x, i):
        w = _window(x, i, self.radius)
        if w.size < 3:
            return None
        mu, sigma = float(np.mean(w)), float(np.std(w))
        if sigma < 1e-12:
            return None
        p = math.erfc(abs(x[i] - mu) / sigma / math.sqrt(2))
        return float(np.median(w)) if w.size * p < 0.5 else None


class SigmaClipScorer(_Windowed):
    """Iterative sigma clipping on the centred window without the value itself.

    On the neighbours of the value (fewer than 3 are skipped) the mean and the
    standard deviation are recomputed after removing the points farther than
    ``n_sigmas`` standard deviations, until the kept points stop changing or
    ``max_iters`` is reached; the value is an outlier when it is farther than
    ``n_sigmas`` cleaned standard deviations from the cleaned mean, or when the
    cleaned deviation is below ``1e-12`` and the value differs from the cleaned
    mean by more than ``1e-9``. An outlier is replaced by the median of the
    kept points.

    Args:
        radius: half-width of the window.
        n_sigmas: clipping and outlier threshold.
        max_iters: maximum number of clipping iterations.

    Example:
        >>> m = SigmaClipScorer(radius=4).fit([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 50.0])
        >>> m.anomalies_
        [8]
    """

    def __init__(
        self, radius: int = 10, n_sigmas: float = 3.0, max_iters: int = 5
    ) -> None:
        self.radius = radius
        self.n_sigmas = n_sigmas
        self.max_iters = max_iters

    def _flag(self, x, i):
        lo, hi = max(0, i - self.radius), min(len(x), i + self.radius + 1)
        w = np.delete(x[lo:hi], i - lo)
        if w.size < 3:
            return None
        mask = np.ones(w.shape, dtype=bool)
        for _ in range(self.max_iters):
            sub = w[mask]
            if sub.size < 2:
                break
            mu, sigma = float(np.mean(sub)), float(np.std(sub))
            if sigma < 1e-12:
                break
            new = np.abs(w - mu) <= self.n_sigmas * sigma
            if np.array_equal(new, mask):
                break
            mask = new
        sub = w[mask]
        if sub.size < 1:
            return None
        mu, sigma = float(np.mean(sub)), float(np.std(sub))
        if sigma < 1e-12:
            return float(np.median(sub)) if abs(x[i] - mu) > 1e-9 else None
        return float(np.median(sub)) if abs(x[i] - mu) > self.n_sigmas * sigma else None


class LocalOutlierFactor(MLBase):
    """Local Outlier Factor of Breunig et al. (2000), Definitions 3 to 7, on the whole table.

    ``k_dist(p)`` is the distance to the k-th nearest other point; the
    neighbourhood of ``p`` holds every other point at distance at most
    ``k_dist(p)`` (ties included); ``reach_dist(p, o) = max(k_dist(o), d(p, o))``;
    ``lrd(p)`` is the size of the neighbourhood divided by the sum of the
    reachability distances; ``LOF(p)`` is the mean ``lrd`` of the neighbours
    divided by ``lrd(p)``. The sum of the reachability distances gets a term
    ``1e-10`` per neighbour, so that duplicates give a finite density: the one
    deviation from the paper, which assumes no duplicates.

    Args:
        n_neighbors: ``k``, also called MinPts.

    Example:
        >>> X = [[0.0], [1.0], [2.0], [3.0], [10.0]]
        >>> m = LocalOutlierFactor(n_neighbors=2).fit(X)
        >>> m.lof_.round(2).tolist()
        [1.0, 1.0, 1.0, 1.0, 5.0]
    """

    def __init__(self, n_neighbors: int = 20) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):  # type: ignore[override]
        W = np.asarray(X, dtype=float)
        k = self.n_neighbors
        if len(W) < k + 1:
            raise ValueError("the table needs at least n_neighbors + 1 rows")
        self.X_ = W
        D = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=-1))
        self.kdist_ = np.sort(D, axis=1)[:, k]
        n = len(W)
        nb = []
        lrd = np.empty(n)
        for i in range(n):
            m = D[i] <= self.kdist_[i]
            m[i] = False
            idx = np.flatnonzero(m)
            nb.append(idx)
            r = np.maximum(self.kdist_[idx], D[i, idx])
            lrd[i] = idx.size / (float(r.sum()) + idx.size * 1e-10)
        self.lrd_ = lrd
        self.lof_ = np.array([lrd[nb[i]].mean() / lrd[i] for i in range(n)])
        return self

    def score(self, X) -> np.ndarray:
        """LOF of new rows against the fitted table (the neighbours of a row are among the fitted rows)."""
        out = []
        k = self.n_neighbors
        for x in np.asarray(X, dtype=float):
            dx = np.sqrt(((self.X_ - x) ** 2).sum(axis=1))
            kd = float(np.sort(dx)[k - 1])
            idx = np.flatnonzero(dx <= kd)
            r = np.maximum(self.kdist_[idx], dx[idx])
            lrd_x = idx.size / (float(r.sum()) + idx.size * 1e-10)
            out.append(self.lrd_[idx].mean() / lrd_x)
        return np.array(out)
