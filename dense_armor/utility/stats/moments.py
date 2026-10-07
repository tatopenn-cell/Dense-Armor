"""Incremental and exponentially weighted moments.

Two ways to summarize a stream of numbers without keeping the samples:

- **Welford / Chan / Pébay**: the incremental update maintains count,
  mean, variance, skewness and kurtosis with a fixed number of floats.
  Numerical stability is a theorem, not a hope: the second moment is
  accumulated as ``M2 = sum (x_i - mean)^2`` and updated incrementally,
  avoiding the cancellation that plagues ``sum(x^2) - n * mean^2``
  (Welford 1962).
- **West**: the exponentially weighted mean and variance give recent
  samples more weight, so the summary follows a slow drift of the
  signal instead of averaging over the whole past.

Two partial summaries with the same structure can be combined exactly
into the summary of the union (Chan et al. 1979; Pébay 2008), which is
how the same statistic is computed over many shards in a map-reduce
system.

References
----------
Welford, B. P. (1962). Note on a method for calculating corrected sums
    of squares and products. Technometrics 4(3), 419-420.
Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae and
    a pairwise algorithm for computing sample variances. In Compstat.
Pébay, P. (2008). Formulas for robust, one-pass parallel computation of
    covariances and arbitrary-order statistical moments. Sandia Report
    SAND2008-6212.
West, D. H. D. (1979). Updating mean and variance estimates: an improved
    method. Communications of the ACM 22(9), 532-535.
"""
from __future__ import annotations

import math
from typing import Any

from dense_armor.base import Transformer


class RunningMoments(Transformer):
    """Incremental mean, variance, skewness, kurtosis for one feature.

    The second, third and fourth central moments are kept in the
    "raw" form ``Mk = sum (x_i - mean)^k``. Each update adds one sample
    in constant time, with no accumulation of cancellation error.

    Args:
        feature: dict key to read. ``None`` (default) takes the smallest
            key, so the result does not depend on the order of the keys.

    Examples:
        >>> from dense_armor.utility.stats.moments import RunningMoments
        >>> m = RunningMoments()
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = m.learn_one({"x": v})
        >>> round(m.mean, 6), round(m.var, 6)
        (3.0, 2.5)
        >>> round(m.skewness, 6)
        0.0

    References:
        Welford, B. P. (1962). Technometrics 4(3), 419-420.
        Pébay, P. (2008). Sandia Report SAND2008-6212.
    """

    def __init__(self, feature: str | None = None) -> None:
        self.feature = feature
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0
        self.m3_ = 0.0
        self.m4_ = 0.0
        self.min_: float | None = None
        self.max_: float | None = None
        self.n_missing_ = 0

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        v = float(x[self._key(x)])
        return None if math.isnan(v) else v

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningMoments":
        """Update the summary with one sample from ``x``."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        n_old = self.n_
        n = n_old + 1
        d = v - self.mean_
        dn = d / n
        dn2 = dn * dn
        term = d * dn * n_old
        self.m4_ += (
            term * dn2 * (n * n - 3 * n + 3)
            + 6.0 * dn2 * self.m2_
            - 4.0 * dn * self.m3_
        )
        self.m3_ += term * dn * (n - 2) - 3.0 * dn * self.m2_
        self.m2_ += term
        self.mean_ += dn
        self.n_ = n
        if self.min_ is None or v < self.min_:
            self.min_ = v
        if self.max_ is None or v > self.max_:
            self.max_ = v
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningMoments":
        """Alias of :meth:`learn_one`, natural name for a statistic."""
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def mean(self) -> float:
        return self.mean_

    @property
    def var(self) -> float:
        return self.m2_ / (self.n_ - 1) if self.n_ >= 2 else 0.0

    @property
    def std(self) -> float:
        return self.var**0.5

    @property
    def skewness(self) -> float:
        if self.n_ < 3 or self.m2_ == 0:
            return 0.0
        m3 = self.m3_ / self.n_
        m2 = self.m2_ / self.n_
        return m3 / (m2**1.5)

    @property
    def kurtosis(self) -> float:
        if self.n_ < 4 or self.m2_ == 0:
            return 0.0
        return self.n_ * self.m4_ / (self.m2_**2) - 3.0

    @property
    def min(self) -> float | None:
        return self.min_

    @property
    def max(self) -> float | None:
        return self.max_

    @property
    def ptp(self) -> float:
        if self.min_ is None or self.max_ is None:
            return 0.0
        return self.max_ - self.min_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def merge(self, other: "RunningMoments") -> "RunningMoments":
        """Combine two partial summaries into the summary of the union.

        Uses the parallel formulae of Chan et al. (1979) for M2, extended
        to M3 and M4 by Pébay (2008). The result is exact: the merged
        summary is the same as if both halves had been fed into one
        instance.

        Args:
            other: a ``RunningMoments`` built on a disjoint set of samples.

        Returns:
            A new ``RunningMoments`` with the combined state.

        Raises:
            TypeError: if ``other`` is not a ``RunningMoments``.
        """
        if not isinstance(other, RunningMoments):
            raise TypeError(
                f"merge expects RunningMoments, got {type(other).__name__}"
            )
        m = RunningMoments(feature=self.feature)
        m.n_missing_ = self.n_missing_ + other.n_missing_
        na, nb = self.n_, other.n_
        if na == 0:
            m.n_, m.mean_, m.m2_, m.m3_, m.m4_ = (
                other.n_,
                other.mean_,
                other.m2_,
                other.m3_,
                other.m4_,
            )
            m.min_, m.max_ = other.min_, other.max_
            return m
        if nb == 0:
            m.n_, m.mean_, m.m2_, m.m3_, m.m4_ = (
                self.n_,
                self.mean_,
                self.m2_,
                self.m3_,
                self.m4_,
            )
            m.min_, m.max_ = self.min_, self.max_
            return m
        n = na + nb
        d = other.mean_ - self.mean_
        m.n_ = n
        m.mean_ = self.mean_ + d * nb / n
        m.m2_ = self.m2_ + other.m2_ + d * d * na * nb / n
        m.m3_ = (
            self.m3_
            + other.m3_
            + d**3 * na * nb * (na - nb) / (n * n)
            + 3.0 * d * (na * other.m2_ - nb * self.m2_) / n
        )
        m.m4_ = (
            self.m4_
            + other.m4_
            + d**4 * na * nb * (na * na - na * nb + nb * nb) / (n**3)
            + 6.0 * d * d * (na * na * other.m2_ + nb * nb * self.m2_) / (n * n)
            + 4.0 * d * (na * other.m3_ - nb * self.m3_) / n
        )
        mins = [v for v in (self.min_, other.min_) if v is not None]
        maxs = [v for v in (self.max_, other.max_) if v is not None]
        m.min_ = min(mins) if mins else None
        m.max_ = max(maxs) if maxs else None
        return m

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Return the current summary as a dict (``x`` is ignored)."""
        return {
            "count": self.n_,
            "mean": self.mean_,
            "var": self.var,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "min": self.min_,
            "max": self.max_,
            "ptp": self.ptp,
            "n_missing": self.n_missing_,
        }


class RunningMomentsVector(Transformer):
    """One ``RunningMoments`` per channel, updated in a single call.

    Channels are the sorted keys of the first dict seen, or the explicit
    ``features`` list. Each channel keeps its own ``RunningMoments``;
    calling ``learn_one(x)`` updates all channels from the same dict, so
    synchronised joints move through the tracker in lockstep.

    Args:
        features: explicit channel order. ``None`` (default) sorts the
            keys of the first dict seen.

    Examples:
        >>> from dense_armor.utility.stats.moments import RunningMomentsVector
        >>> v = RunningMomentsVector()
        >>> for q in ([0.1, 0.2], [0.11, 0.19], [0.09, 0.21]):
        ...     _ = v.learn_one({"q0": q[0], "q1": q[1]})
        >>> {k: round(s["mean"], 4) for k, s in v.transform_one({}).items()}
        {'q0': 0.1, 'q1': 0.2}
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features
        self.channels_: dict[str, RunningMoments] | None = None

    def _ensure(self, x: dict) -> dict[str, RunningMoments]:
        if self.channels_ is None:
            keys = list(self.features) if self.features is not None else sorted(x)
            self.channels_ = {k: RunningMoments(feature=k) for k in keys}
        return self.channels_

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningMomentsVector":
        """Update all channels from a single dict of synchronised values."""
        self._time_step(t)
        channels = self._ensure(x)
        for st in channels.values():
            st.learn_one(x, t=t)
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningMomentsVector":
        return self.learn_one(x, y=y, t=t)

    def merge(self, other: "RunningMomentsVector") -> "RunningMomentsVector":
        """Combine two vector summaries channel by channel.

        Args:
            other: another ``RunningMomentsVector`` on the same channels
                and a disjoint set of samples.

        Returns:
            A new ``RunningMomentsVector`` with the merged state per channel.

        Raises:
            TypeError: if ``other`` is not a ``RunningMomentsVector``.
            ValueError: if either has never seen data, or if the channel
                keys differ.
        """
        if not isinstance(other, RunningMomentsVector):
            raise TypeError(
                f"merge expects RunningMomentsVector, got {type(other).__name__}"
            )
        if self.channels_ is None or other.channels_ is None:
            raise ValueError("merge requires both summaries to have seen data")
        keys = list(self.channels_.keys())
        if set(keys) != set(other.channels_.keys()):
            raise ValueError("merge requires the same channel keys")
        out = RunningMomentsVector(features=keys)
        out.channels_ = {
            k: self.channels_[k].merge(other.channels_[k]) for k in keys
        }
        return out

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        if self.channels_ is None:
            return {}
        return {k: st.transform_one({}) for k, st in self.channels_.items()}


class EWStats(Transformer):
    """Exponentially weighted mean and variance, one sample at a time.

    Recent samples get more weight: the mean is
    ``mean_t = (1 - alpha) mean_{t-1} + alpha x_t`` and the variance
    follows West (1979) so that a slow drift of the signal is tracked
    instead of averaged over the whole past. ``alpha = 1`` copies the
    last value; small ``alpha`` (0.01 - 0.1) is the usual range.

    Args:
        alpha: weight of the new sample, in ``(0, 1]``.
        feature: dict key to read. ``None`` (default) takes the smallest
            key.

    Examples:
        >>> from dense_armor.utility.stats.moments import EWStats
        >>> e = EWStats(alpha=0.5)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = e.learn_one({"x": v})
        >>> round(e.mean, 6)
        4.0625

    References:
        West, D. H. D. (1979). Communications of the ACM 22(9), 532-535.
    """

    def __init__(self, alpha: float = 0.1, feature: str | None = None) -> None:
        self.alpha = alpha
        self.feature = feature
        self.mean_: float | None = None
        self.var_ = 0.0
        self.n_ = 0
        self.n_missing_ = 0

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        v = float(x[self._key(x)])
        return None if math.isnan(v) else v

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "EWStats":
        """Update the exponentially weighted mean and variance."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if self.mean_ is None:
            self.mean_ = v
            self.var_ = 0.0
        else:
            d = v - self.mean_
            self.mean_ = self.mean_ + self.alpha * d
            self.var_ = (1.0 - self.alpha) * (self.var_ + self.alpha * d * d)
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "EWStats":
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def mean(self) -> float:
        return 0.0 if self.mean_ is None else self.mean_

    @property
    def var(self) -> float:
        return self.var_

    @property
    def std(self) -> float:
        return self.var_**0.5

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "mean": self.mean,
            "var": self.var,
            "std": self.std,
            "n_missing": self.n_missing_,
        }
