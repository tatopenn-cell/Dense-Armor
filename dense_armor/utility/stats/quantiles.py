"""Streaming quantiles without a bounded window.

Two sketches, one sample at a time, constant memory:

- **DDSketch** (Masson et al. 2019) provides a formal *relative* error
  guarantee: for any ``q`` and any ``α`` in ``(0, 1)``,
  ``|q̃ − q| ≤ α · q`` where ``q̃`` is the estimate returned by the
  sketch. The buckets are geometric with ratio ``γ = (1 + α) / (1 − α)``
  and the value attached to a bucket is its geometric midpoint. Relative
  error is what matters for heavy-tailed data (latency, request
  response time): a rank-error guarantee alone can return wildly wrong
  values at the p99 of a heavy tail.
- **t-digest** (Dunning & Ertl 2019) clusters the samples and keeps
  *small* clusters near the tails, using a scale function
  ``k1(q) = δ / (2π) · arcsin(2q − 1)`` that maps the quantile ``q`` to a
  notional index ``k``. Accuracy is high near ``q = 0`` and ``q = 1``,
  modest near the median.

Both are ``Transformer``s: ``learn_one(x)`` feeds one sample,
``quantile(q)`` returns the estimate, ``merge(other)`` combines two
sketches into one.

References
----------
Masson, C., Rim, J. E., Lee, H. K. (2019). DDSketch: a fast and
    fully-mergeable quantile sketch with relative-error guarantees.
    PVLDB 12(12), 2195-2205.
Dunning, T., Ertl, O. (2019). Computing extremely accurate quantiles
    using t-digests. arXiv:1902.04023.
"""
from __future__ import annotations

import math
from typing import Any

from dense_armor.base import Transformer


class DDSketch(Transformer):
    """Relative-error quantile sketch (Masson, Rim, Lee 2019).

    Buckets are geometric: a value ``x > 0`` lands in bucket
    ``i = ceil(log_γ x)`` with ``γ = (1 + α) / (1 − α)``, and the bucket
    is represented by the geometric midpoint ``2 γ^i / (γ + 1)``. This
    gives a formal relative error guarantee
    ``|q̃ − q| ≤ α · q`` for every ``q`` and every ``α ∈ (0, 1)``
    (Lemma 2, Proposition 3 of the paper).

    Negative values are handled by a second sketch on ``|x|``; zero is
    tracked separately. When the number of non-empty buckets exceeds
    ``max_buckets``, the *smallest* buckets are collapsed.

    Args:
        alpha: relative error, in ``(0, 1)``. Common values: ``0.01``
            (1%), ``0.05`` (5%). Smaller ``alpha`` means a larger sketch.
        max_buckets: upper bound on the number of non-empty buckets per
            side (default 2048).
        feature: dict key to read. ``None`` reads the smallest key.

    Examples:
        >>> from dense_armor.utility.stats.quantiles import DDSketch
        >>> import numpy as np
        >>> s = DDSketch(alpha=0.01)
        >>> rng = np.random.default_rng(0)
        >>> for v in rng.normal(100.0, 1.0, 2000):
        ...     _ = s.learn_one({"x": float(v)})
        >>> 99.0 < s.quantile(0.5) < 101.0
        True

    References:
        Masson, C., Rim, J. E., Lee, H. K. (2019). PVLDB 12(12),
        2195-2205.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_buckets: int = 2048,
        feature: str | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_buckets < 2:
            raise ValueError(f"max_buckets must be >= 2, got {max_buckets}")
        self.alpha = alpha
        self.max_buckets = max_buckets
        self.feature = feature
        self.gamma_: float = (1.0 + alpha) / (1.0 - alpha)
        self.log_gamma_: float = math.log(self.gamma_)
        self.buckets_pos_: dict[int, int] = {}
        self.buckets_neg_: dict[int, int] = {}
        self.zero_: int = 0
        self.n_ = 0
        self.n_missing_ = 0
        self.min_: float | None = None
        self.max_: float | None = None

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        v = float(x[self._key(x)])
        return None if math.isnan(v) else v

    def _index(self, x: float) -> int:
        return math.ceil(math.log(x) / self.log_gamma_)

    def _collapse(self, buckets: dict[int, int]) -> None:
        while len(buckets) > self.max_buckets:
            i0 = min(buckets)
            i1 = min(j for j in buckets if j > i0)
            buckets[i1] += buckets[i0]
            del buckets[i0]

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "DDSketch":
        """Insert one sample."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if v == 0.0:
            self.zero_ += 1
        elif v > 0.0:
            i = self._index(v)
            self.buckets_pos_[i] = self.buckets_pos_.get(i, 0) + 1
            self._collapse(self.buckets_pos_)
        else:
            i = self._index(-v)
            self.buckets_neg_[i] = self.buckets_neg_.get(i, 0) + 1
            self._collapse(self.buckets_neg_)
        self.n_ += 1
        if self.min_ is None or v < self.min_:
            self.min_ = v
        if self.max_ is None or v > self.max_:
            self.max_ = v
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "DDSketch":
        return self.learn_one(x, y=y, t=t)

    def _bucket_value(self, i: int) -> float:
        return 2.0 * (self.gamma_**i) / (self.gamma_ + 1.0)

    def quantile(self, q: float) -> float:
        """Estimate the ``q``-quantile.

        Args:
            q: quantile in ``[0, 1]``.

        Returns:
            The estimate, within ``alpha`` relative error for every
            ``q`` that has not been collapsed (paper Proposition 4).
        """
        if self.n_ == 0:
            return 0.0
        if q <= 0.0:
            return self.min_ if self.min_ is not None else 0.0
        if q >= 1.0:
            return self.max_ if self.max_ is not None else 0.0
        target = q * (self.n_ - 1)
        cum = 0
        for i in sorted(self.buckets_neg_, reverse=True):
            c = self.buckets_neg_[i]
            if cum + c > target:
                return -self._bucket_value(i)
            cum += c
        if cum + self.zero_ > target:
            return 0.0
        cum += self.zero_
        for i in sorted(self.buckets_pos_):
            c = self.buckets_pos_[i]
            if cum + c > target:
                return self._bucket_value(i)
            cum += c
        return self.max_ if self.max_ is not None else 0.0

    def merge(self, other: "DDSketch") -> "DDSketch":
        """Combine two DDSketches built on disjoint samples.

        Args:
            other: another ``DDSketch`` with the same ``alpha`` and
                ``max_buckets``.

        Returns:
            A new ``DDSketch`` with the merged state.

        Raises:
            TypeError: if ``other`` is not a ``DDSketch``.
            ValueError: if the ``alpha`` or ``max_buckets`` differ.
        """
        if not isinstance(other, DDSketch):
            raise TypeError(
                f"merge expects DDSketch, got {type(other).__name__}"
            )
        if self.alpha != other.alpha or self.max_buckets != other.max_buckets:
            raise ValueError("merge requires the same alpha and max_buckets")
        out = DDSketch(
            alpha=self.alpha,
            max_buckets=self.max_buckets,
            feature=self.feature,
        )
        out.n_ = self.n_ + other.n_
        out.n_missing_ = self.n_missing_ + other.n_missing_
        out.zero_ = self.zero_ + other.zero_
        for i, c in self.buckets_pos_.items():
            out.buckets_pos_[i] = c + other.buckets_pos_.get(i, 0)
        for i, c in other.buckets_pos_.items():
            if i not in out.buckets_pos_:
                out.buckets_pos_[i] = c
        for i, c in self.buckets_neg_.items():
            out.buckets_neg_[i] = c + other.buckets_neg_.get(i, 0)
        for i, c in other.buckets_neg_.items():
            if i not in out.buckets_neg_:
                out.buckets_neg_[i] = c
        out._collapse(out.buckets_pos_)
        out._collapse(out.buckets_neg_)
        mins = [v for v in (self.min_, other.min_) if v is not None]
        maxs = [v for v in (self.max_, other.max_) if v is not None]
        out.min_ = min(mins) if mins else None
        out.max_ = max(maxs) if maxs else None
        return out

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def min(self) -> float | None:
        return self.min_

    @property
    def max(self) -> float | None:
        return self.max_

    @property
    def size(self) -> int:
        """Number of non-empty buckets in the sketch."""
        return len(self.buckets_pos_) + len(self.buckets_neg_)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "buckets": self.size,
            "min": self.min_,
            "max": self.max_,
            "n_missing": self.n_missing_,
        }


class TDigest(Transformer):
    """Clustered sketch with small clusters near the tails (Dunning 2019).

    Samples accumulate in a buffer. When the buffer is full (or
    ``quantile`` is called), the buffer plus the existing centroids are
    sorted and merged into a new set of centroids such that each cluster
    ``C`` satisfies ``|C|_k ≤ 1`` where
    ``|C|_k = k(q_right) − k(q_left)`` and the scale function is

    .. math::

        k_1(q) = \\frac{\\delta}{2\\pi} \\arcsin(2q - 1).

    The scale function's steepness at the tails forces clusters near
    ``q = 0`` and ``q = 1`` to be small, which is where high relative
    accuracy matters for heavy-tailed data.

    Args:
        delta: compression parameter (default 100). The number of
            centroids is bounded by ``ceil(delta)``; larger ``delta``
            gives smaller clusters and better accuracy.
        buffer_size: number of samples buffered before a merge. ``None``
            uses ``10 · delta``, the paper's recommended default.
        feature: dict key to read. ``None`` reads the smallest key.

    Examples:
        >>> from dense_armor.utility.stats.quantiles import TDigest
        >>> import numpy as np
        >>> t = TDigest(delta=100)
        >>> rng = np.random.default_rng(0)
        >>> for v in rng.normal(0.0, 1.0, 5000):
        ...     _ = t.learn_one({"x": float(v)})
        >>> abs(t.quantile(0.5)) < 0.1
        True

    References:
        Dunning, T., Ertl, O. (2019). arXiv:1902.04023.
    """

    def __init__(
        self,
        delta: float = 100.0,
        buffer_size: int | None = None,
        feature: str | None = None,
    ) -> None:
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.delta = delta
        self.buffer_size = buffer_size
        self.feature = feature
        self.centroids_: list[list[float]] = []
        self.buffer_: list[float] = []
        self.n_ = 0
        self.n_missing_ = 0
        self.min_: float | None = None
        self.max_: float | None = None

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        v = float(x[self._key(x)])
        return None if math.isnan(v) else v

    def _effective_buffer(self) -> int:
        if self.buffer_size is not None:
            return self.buffer_size
        return int(10 * self.delta)

    def _k(self, q: float) -> float:
        return self.delta / (2.0 * math.pi) * math.asin(2.0 * q - 1.0)

    def _k_inv(self, k: float) -> float:
        return (math.sin(2.0 * math.pi * k / self.delta) + 1.0) / 2.0

    def _consolidate(self, items: list[tuple[float, float]]) -> None:
        items.sort(key=lambda t: t[0])
        total = sum(c for _, c in items)
        q0 = 0.0
        q_limit = self._k_inv(self._k(q0) + 1.0)
        sigma_mean, sigma_count = items[0]
        new_centroids: list[list[float]] = []
        for m, c in items[1:]:
            q = q0 + (sigma_count + c) / total
            if q <= q_limit:
                sigma_count += c
                sigma_mean += (m - sigma_mean) * c / sigma_count
            else:
                new_centroids.append([sigma_mean, sigma_count])
                q0 += sigma_count / total
                q_limit = self._k_inv(self._k(q0) + 1.0)
                sigma_mean, sigma_count = m, c
        new_centroids.append([sigma_mean, sigma_count])
        self.centroids_ = new_centroids
        self.buffer_ = []

    def _merge(self) -> None:
        items: list[tuple[float, float]] = [
            (m, c) for m, c in self.centroids_
        ]
        items.extend((v, 1.0) for v in self.buffer_)
        if not items:
            return
        self._consolidate(items)

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "TDigest":
        """Insert one sample."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        self.buffer_.append(v)
        self.n_ += 1
        if self.min_ is None or v < self.min_:
            self.min_ = v
        if self.max_ is None or v > self.max_:
            self.max_ = v
        if len(self.buffer_) >= self._effective_buffer():
            self._merge()
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "TDigest":
        return self.learn_one(x, y=y, t=t)

    def quantile(self, q: float) -> float:
        """Estimate the ``q``-quantile.

        The cumulative distribution is approximated by linear
        interpolation between the midpoints of adjacent clusters, with
        half the weight of each cluster placed to the left of its
        centroid and half to the right (Dunning 2019, §2.9).

        Args:
            q: quantile in ``[0, 1]``.
        """
        if self.buffer_:
            self._merge()
        if not self.centroids_ or self.n_ == 0:
            return 0.0
        if q <= 0.0:
            return self.centroids_[0][0]
        if q >= 1.0:
            return self.centroids_[-1][0]
        n = sum(c for _, c in self.centroids_)
        target = q * (n - 1)
        cum = 0.0
        for i, (m, c) in enumerate(self.centroids_):
            center = cum + c / 2.0
            if center >= target:
                if i == 0:
                    return m
                prev_m, prev_c = self.centroids_[i - 1]
                prev_center = cum - prev_c / 2.0
                if center == prev_center:
                    return m
                t = (target - prev_center) / (center - prev_center)
                return prev_m + t * (m - prev_m)
            cum += c
        return self.centroids_[-1][0]

    def merge(self, other: "TDigest") -> "TDigest":
        """Combine two t-digests built on disjoint samples.

        Both digests are flushed, their weighted centroids are
        combined, and the same merging pass is applied once more.
        Merged digests are only weakly ordered (Dunning 2019 §2.5),
        so accuracy is close to but not identical to a single digest
        built from the full stream.

        Args:
            other: another ``TDigest``.

        Returns:
            A new ``TDigest`` with the merged state.

        Raises:
            TypeError: if ``other`` is not a ``TDigest``.
            ValueError: if the ``delta`` values differ.
        """
        if not isinstance(other, TDigest):
            raise TypeError(
                f"merge expects TDigest, got {type(other).__name__}"
            )
        if self.delta != other.delta:
            raise ValueError("merge requires the same delta")
        if self.buffer_:
            self._merge()
        if other.buffer_:
            other._merge()
        out = TDigest(
            delta=self.delta,
            buffer_size=self.buffer_size,
            feature=self.feature,
        )
        items: list[tuple[float, float]] = [
            (m, c) for m, c in self.centroids_
        ]
        items.extend((m, c) for m, c in other.centroids_)
        out.n_ = self.n_ + other.n_
        out.n_missing_ = self.n_missing_ + other.n_missing_
        mins = [v for v in (self.min_, other.min_) if v is not None]
        maxs = [v for v in (self.max_, other.max_) if v is not None]
        out.min_ = min(mins) if mins else None
        out.max_ = max(maxs) if maxs else None
        if items:
            out._consolidate(items)
        return out

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def min(self) -> float | None:
        return self.min_

    @property
    def max(self) -> float | None:
        return self.max_

    @property
    def n_centroids(self) -> int:
        """Number of centroids currently retained."""
        if self.buffer_:
            self._merge()
        return len(self.centroids_)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "centroids": self.n_centroids,
            "min": self.min_,
            "max": self.max_,
            "n_missing": self.n_missing_,
        }
