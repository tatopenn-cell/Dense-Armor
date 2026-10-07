"""Rolling robust statistics over a bounded window.

Four statistics, same pattern: a deque keeps the last ``window`` samples
in arrival order, and a parallel sorted list keeps them in order. Each
update is ``O(window)``; memory is ``O(window)``. The window never grows.

The three scale constants are the standard robust-statistics constants:

- **1.4826** (Hampel 1974) makes the median absolute deviation (MAD) a
  consistent estimator of the standard deviation for Gaussian data:
  ``sigma = 1.4826 * MAD``.
- **IQR** (Tukey 1977) is the difference between the 75th and 25th
  percentiles; its Gaussian-consistent scale factor is
  ``1 / 1.349 = 0.7413``.
- **Quantile interpolation** uses the ``linear`` rule, matching
  ``numpy.percentile`` default (Hyndman & Fan 1996, type 7).

NaN is skipped and counted: a sensor glitch does not corrupt the
window, it bumps a counter.

References
----------
Hampel, F. R. (1974). The influence curve and its role in robust
    estimation. JASA 69(346), 383-393.
Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
Hyndman, R. J., Fan, Y. (1996). Sample quantiles in statistical
    packages. The American Statistician 50(4), 361-365.
"""
from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from typing import Any

from dense_armor.base import Transformer

MAD_SCALE = 1.4826
IQR_SCALE = 1.0 / 1.349


class _RollingWindow(Transformer):
    """Shared machinery for rolling robust statistics.

    Not part of the public API. Subclasses define the ``value`` property
    and their own constructor, and inherit the whole update machinery.

    Args:
        window: window size in samples, used if ``window_s`` is not given.
        window_s: window size in seconds, converted to samples using the
            base ``dt`` once two timestamps have been seen.
        feature: dict key to read. ``None`` reads the smallest key.
    """

    def __init__(
        self,
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
    ) -> None:
        self.window = window
        self.window_s = window_s
        self.feature = feature
        self.buf_: deque[float] = deque(maxlen=window)
        self.sorted_: list[float] = []
        self.n_ = 0
        self.n_missing_ = 0
        self._resolved_window: int | None = None

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        v = float(x[self._key(x)])
        return None if math.isnan(v) else v

    def _resolve_window(self) -> int:
        if self.window_s is not None and self.dt is not None:
            self._resolved_window = self.window_samples(self.window_s)
            return self._resolved_window
        if self._resolved_window is not None:
            return self._resolved_window
        return self.window

    def _push(self, v: float) -> None:
        n = self._resolve_window()
        if self.buf_.maxlen != n:
            self.buf_ = deque(self.buf_, maxlen=n)
            self.sorted_ = sorted(self.buf_)
        if len(self.buf_) == n and n > 0:
            old = self.buf_[0]
            i = bisect_left(self.sorted_, old)
            self.sorted_.pop(i)
        self.buf_.append(v)
        insort(self.sorted_, v)

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "_RollingWindow":
        """Push one sample into the window."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        self._push(v)
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "_RollingWindow":
        """Alias of :meth:`learn_one`, natural name for a statistic."""
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def window_size(self) -> int:
        """Effective window size in samples, after ``window_s`` resolution."""
        return self._resolve_window()

    def _quantile(self, q: float) -> float:
        if not self.sorted_:
            return 0.0
        n = len(self.sorted_)
        if n == 1:
            return self.sorted_[0]
        pos = q * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return self.sorted_[lo]
        return self.sorted_[lo] + (pos - lo) * (
            self.sorted_[hi] - self.sorted_[lo]
        )

    @property
    def value(self) -> float:
        """The statistic, computed on the current window."""
        raise NotImplementedError

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "value": self.value,
            "n": len(self.sorted_),
            "n_missing": self.n_missing_,
        }


class RollingMedian(_RollingWindow):
    """Median of the last ``window`` samples.

    The window is a bounded FIFO; each sample enters and the oldest
    leaves. The median is exact: no approximation, just a sorted list
    kept in step with the arrival-order deque. Cost is ``O(window)`` per
    update.

    Args:
        window: window size in samples (default 20).
        window_s: window size in seconds; if given, takes precedence over
            ``window`` once two timestamps have been seen.
        feature: dict key to read; ``None`` reads the smallest key.

    Examples:
        >>> from dense_armor.utility.stats.robust import RollingMedian
        >>> m = RollingMedian(window=5)
        >>> for v in [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]:
        ...     _ = m.learn_one({"x": v})
        >>> m.value
        4.0

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    """

    @property
    def value(self) -> float:
        return self._quantile(0.5)


class RollingMAD(_RollingWindow):
    """Median absolute deviation of the last ``window`` samples, scaled.

    The MAD is ``median(|x_i - median|)``; the 1.4826 factor makes it a
    consistent estimator of sigma for Gaussian data (Hampel 1974). The
    scale, not the raw MAD, is what the ``value`` property returns.

    Args:
        window: window size in samples (default 20).
        window_s: window size in seconds.
        feature: dict key to read; ``None`` reads the smallest key.
        scale: multiplicative factor applied to the raw MAD. Default
            ``1.4826`` (Hampel).

    Examples:
        >>> from dense_armor.utility.stats.robust import RollingMAD
        >>> m = RollingMAD(window=5)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = m.learn_one({"x": v})
        >>> round(m.value, 4)
        1.4826

    References:
        Hampel, F. R. (1974). JASA 69(346), 383-393.
    """

    def __init__(
        self,
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
        scale: float = MAD_SCALE,
    ) -> None:
        super().__init__(window=window, window_s=window_s, feature=feature)
        self.scale = scale

    @property
    def value(self) -> float:
        if not self.sorted_:
            return 0.0
        med = self._quantile(0.5)
        devs = sorted(abs(v - med) for v in self.sorted_)
        n = len(devs)
        if n == 1:
            return 0.0
        pos = 0.5 * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        raw = (
            devs[lo]
            if lo == hi
            else devs[lo] + (pos - lo) * (devs[hi] - devs[lo])
        )
        return self.scale * raw


class RollingIQR(_RollingWindow):
    """Interquartile range of the last ``window`` samples.

    The IQR is ``Q3 - Q1``, the width of the box in a box plot. To use
    it as a sigma-like scale, multiply by ``1 / 1.349 = 0.7413`` — that
    is what ``RollingIQR(..., scale=IQR_SCALE)`` does.

    Args:
        window: window size in samples (default 20).
        window_s: window size in seconds.
        feature: dict key to read; ``None`` reads the smallest key.
        scale: multiplicative factor applied to the raw IQR (default 1.0).

    Examples:
        >>> from dense_armor.utility.stats.robust import RollingIQR
        >>> m = RollingIQR(window=5)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = m.learn_one({"x": v})
        >>> m.value
        2.0

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    """

    def __init__(
        self,
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__(window=window, window_s=window_s, feature=feature)
        self.scale = scale

    @property
    def value(self) -> float:
        return self.scale * (self._quantile(0.75) - self._quantile(0.25))


class RollingQuantile(_RollingWindow):
    """Arbitrary quantile of the last ``window`` samples.

    Interpolation is the ``linear`` rule of ``numpy.percentile``
    (Hyndman & Fan type 7). For ``q = 0.5`` the result equals
    :class:`RollingMedian`.

    Args:
        q: quantile in ``[0, 1]``.
        window: window size in samples (default 20).
        window_s: window size in seconds.
        feature: dict key to read; ``None`` reads the smallest key.

    Examples:
        >>> from dense_armor.utility.stats.robust import RollingQuantile
        >>> m = RollingQuantile(q=0.9, window=5)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = m.learn_one({"x": v})
        >>> m.value
        4.6

    References:
        Hyndman, R. J., Fan, Y. (1996). The American Statistician 50(4),
        361-365.
    """

    def __init__(
        self,
        q: float = 0.5,
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
    ) -> None:
        super().__init__(window=window, window_s=window_s, feature=feature)
        self.q = q

    @property
    def value(self) -> float:
        return self._quantile(self.q)


class RollingMedianVector(Transformer):
    """One :class:`RollingMedian` per channel, one update call.

    Channels are the sorted keys of the first dict seen, or the explicit
    ``features`` list. Calling ``learn_one(x)`` feeds the same dict to
    all channel medians, so synchronised joints move in lockstep.

    Args:
        window: window size in samples (default 20).
        window_s: window size in seconds.
        features: explicit channel order. ``None`` sorts the keys of the
            first dict seen.

    Examples:
        >>> from dense_armor.utility.stats.robust import RollingMedianVector
        >>> v = RollingMedianVector(window=3)
        >>> for q in ([0.1, 0.2], [0.12, 0.18], [0.11, 0.21], [0.09, 0.19]):
        ...     _ = v.learn_one({"q0": q[0], "q1": q[1]})
        >>> {k: round(s["value"], 4) for k, s in v.transform_one({}).items()}
        {'q0': 0.11, 'q1': 0.19}
    """

    def __init__(
        self,
        window: int = 20,
        window_s: float | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.window = window
        self.window_s = window_s
        self.features = features
        self.channels_: dict[str, RollingMedian] | None = None

    def _ensure(self, x: dict) -> dict[str, RollingMedian]:
        if self.channels_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.channels_ = {
                k: RollingMedian(
                    window=self.window, window_s=self.window_s, feature=k
                )
                for k in keys
            }
        return self.channels_

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingMedianVector":
        """Update all channels from a single dict of synchronised values."""
        self._time_step(t)
        channels = self._ensure(x)
        for st in channels.values():
            st.learn_one(x, t=t)
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingMedianVector":
        return self.learn_one(x, y=y, t=t)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        if self.channels_ is None:
            return {}
        return {k: st.transform_one({}) for k, st in self.channels_.items()}
