"""Streaming measures of dependence between two channels.

Five estimators, all updated one sample at a time:

- **Running covariance** — Welford's incremental update generalized to
  two channels. Two partial states can be combined *exactly* into the
  state of the union (Chan et al. 1979, §1).
- **Running correlation** — the Pearson coefficient
  ``r = C / sqrt(M2x · M2y)``, read off the running covariance state.
- **Rolling covariance** / **Rolling correlation** — same quantities
  computed on the last ``window`` samples. The window is a bounded FIFO;
  covariance and correlation are recomputed from the window on demand.
- **Autocorrelation at lag k** — how much a signal at time ``t`` agrees
  with itself at time ``t − k``. Detects periodicity and gauges how much
  a new sample carries that the previous ``lag`` do not.

Convention: dict keys are two distinct strings (``x`` and ``y``), or a
single feature for the autocorrelation. ``NaN`` on either channel skips
the sample and bumps ``n_missing_``.

References
----------
Welford, B. P. (1962). Note on a method for calculating corrected sums
    of squares and products. Technometrics 4(3), 419-420.
Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae and
    a pairwise algorithm for computing sample variances. In Compstat.
Chatfield, C. (2003). The Analysis of Time Series: An Introduction,
    6th ed. Chapman & Hall/CRC.
"""
import math
from collections import deque
from typing import Any

from dense_armor.roles import Transformer


class RunningCovariance(Transformer):
    """Incremental covariance of two channels.

    Generalizes Welford's one-channel update: keep the running means of
    ``x`` and ``y`` plus the co-moment ``C = sum (x_i − x̄)(y_i − ȳ)``.
    Each sample updates all three in constant time. Two partial states
    can be merged exactly (Chan et al. 1979).

    Args:
        x_feature: dict key of the first channel.
        y_feature: dict key of the second channel.
        feature: unused placeholder kept for symmetry with the base
            API. Passing a value here has no effect; the two named
            channels are always ``x_feature`` and ``y_feature``.

    Examples:
        >>> from dense_armor.utility.stats.dependence import RunningCovariance
        >>> c = RunningCovariance("a", "b")
        >>> for a, b in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
        ...     _ = c.learn_one({"a": a, "b": b})
        >>> round(c.cov, 6), round(c.corr, 6)
        (2.0, 1.0)

    References:
        Welford, B. P. (1962). Technometrics 4(3), 419-420.
        Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Compstat.
    """

    def __init__(
        self,
        x_feature: str = "x",
        y_feature: str = "y",
        feature: str | None = None,
    ) -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.feature = feature
        self.n_ = 0
        self.n_missing_ = 0
        self.mean_x_ = 0.0
        self.mean_y_ = 0.0
        self.m2_x_ = 0.0
        self.m2_y_ = 0.0
        self.c_ = 0.0

    def _values(self, x: dict) -> tuple[float, float] | None:
        a = float(x[self.x_feature])
        b = float(x[self.y_feature])
        if math.isnan(a) or math.isnan(b):
            return None
        return a, b

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningCovariance":
        """Update the co-moment with one ``(x, y)`` pair."""
        self._time_step(t)
        pair = self._values(x)
        if pair is None:
            self.n_missing_ += 1
            return self
        a, b = pair
        n = self.n_ + 1
        dx = a - self.mean_x_
        dy = b - self.mean_y_
        self.mean_x_ += dx / n
        self.mean_y_ += dy / n
        self.m2_x_ += dx * (a - self.mean_x_)
        self.m2_y_ += dy * (b - self.mean_y_)
        self.c_ += dx * (b - self.mean_y_)
        self.n_ = n
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningCovariance":
        """Alias of :meth:`learn_one`."""
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def cov(self) -> float:
        """Sample covariance with ``n − 1`` in the denominator."""
        if self.n_ < 2:
            return 0.0
        return self.c_ / (self.n_ - 1)

    @property
    def corr(self) -> float:
        """Pearson correlation coefficient, in ``[-1, 1]``."""
        denom = math.sqrt(self.m2_x_ * self.m2_y_)
        if denom == 0.0:
            return 0.0
        return self.c_ / denom

    def merge(self, other: "RunningCovariance") -> "RunningCovariance":
        """Combine two partial states into the state of the union.

        Uses the pairwise formulae of Chan et al. (1979). The result is
        exact: the merged state is what a single accumulator would have
        produced from both halves.

        Args:
            other: another ``RunningCovariance`` on the same channels and
                a disjoint set of samples.

        Returns:
            A new ``RunningCovariance`` with the combined state.

        Raises:
            TypeError: if ``other`` is not a ``RunningCovariance``.
            ValueError: if the channel names differ.
        """
        if not isinstance(other, RunningCovariance):
            raise TypeError(
                f"merge expects RunningCovariance, got {type(other).__name__}"
            )
        if (self.x_feature, self.y_feature) != (
            other.x_feature,
            other.y_feature,
        ):
            raise ValueError("merge requires the same channel names")
        m = RunningCovariance(
            x_feature=self.x_feature,
            y_feature=self.y_feature,
            feature=self.feature,
        )
        m.n_missing_ = self.n_missing_ + other.n_missing_
        na, nb = self.n_, other.n_
        if na == 0:
            m.n_ = other.n_
            m.mean_x_ = other.mean_x_
            m.mean_y_ = other.mean_y_
            m.m2_x_ = other.m2_x_
            m.m2_y_ = other.m2_y_
            m.c_ = other.c_
            return m
        if nb == 0:
            m.n_ = self.n_
            m.mean_x_ = self.mean_x_
            m.mean_y_ = self.mean_y_
            m.m2_x_ = self.m2_x_
            m.m2_y_ = self.m2_y_
            m.c_ = self.c_
            return m
        n = na + nb
        dx = other.mean_x_ - self.mean_x_
        dy = other.mean_y_ - self.mean_y_
        m.n_ = n
        m.mean_x_ = self.mean_x_ + dx * nb / n
        m.mean_y_ = self.mean_y_ + dy * nb / n
        m.m2_x_ = self.m2_x_ + other.m2_x_ + dx * dx * na * nb / n
        m.m2_y_ = self.m2_y_ + other.m2_y_ + dy * dy * na * nb / n
        m.c_ = self.c_ + other.c_ + dx * dy * na * nb / n
        return m

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "cov": self.cov,
            "corr": self.corr,
            "n_missing": self.n_missing_,
        }


class RunningCorrelation(Transformer):
    """Incremental Pearson correlation between two channels.

    Same state as :class:`RunningCovariance`; exposes ``corr`` directly
    and hides ``cov``.

    Args:
        x_feature: dict key of the first channel.
        y_feature: dict key of the second channel.

    Examples:
        >>> from dense_armor.utility.stats.dependence import RunningCorrelation
        >>> r = RunningCorrelation("a", "b")
        >>> for a, b in [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]:
        ...     _ = r.learn_one({"a": a, "b": b})
        >>> round(r.corr, 6)
        -1.0
    """

    def __init__(
        self,
        x_feature: str = "x",
        y_feature: str = "y",
        feature: str | None = None,
    ) -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.feature = feature
        self._cov = RunningCovariance(x_feature=x_feature, y_feature=y_feature)

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningCorrelation":
        self._time_step(t)
        self._cov.learn_one(x, t=t)
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RunningCorrelation":
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self._cov.count

    @property
    def n_missing(self) -> int:
        return self._cov.n_missing

    @property
    def corr(self) -> float:
        return self._cov.corr

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.count,
            "corr": self.corr,
            "n_missing": self.n_missing,
        }


class RollingCovariance(Transformer):
    """Sample covariance of two channels over a bounded window.

    Two parallel deques hold the last ``window`` pairs; the covariance
    is recomputed from the window on demand. The window never grows.

    Args:
        x_feature: dict key of the first channel.
        y_feature: dict key of the second channel.
        window: window size in samples (default 20).
        window_s: window size in seconds; takes precedence over
            ``window`` once the base ``dt`` is known.
        feature: unused placeholder kept for symmetry with the base API.

    Examples:
        >>> from dense_armor.utility.stats.dependence import RollingCovariance
        >>> c = RollingCovariance("a", "b", window=3)
        >>> for a, b in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
        ...     _ = c.learn_one({"a": a, "b": b})
        >>> round(c.corr, 6)
        1.0
    """

    def __init__(
        self,
        x_feature: str = "x",
        y_feature: str = "y",
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
    ) -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.window = window
        self.window_s = window_s
        self.feature = feature
        self.buf_x_: deque[float] = deque(maxlen=window)
        self.buf_y_: deque[float] = deque(maxlen=window)
        self.n_ = 0
        self.n_missing_ = 0
        self._resolved_window: int | None = None

    def _values(self, x: dict) -> tuple[float, float] | None:
        a = float(x[self.x_feature])
        b = float(x[self.y_feature])
        if math.isnan(a) or math.isnan(b):
            return None
        return a, b

    def _resolve_window(self) -> int:
        if self.window_s is not None and self.dt is not None:
            self._resolved_window = self.window_samples(self.window_s)
            return self._resolved_window
        if self._resolved_window is not None:
            return self._resolved_window
        return self.window

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingCovariance":
        self._time_step(t)
        pair = self._values(x)
        if pair is None:
            self.n_missing_ += 1
            return self
        n = self._resolve_window()
        if self.buf_x_.maxlen != n:
            self.buf_x_ = deque(self.buf_x_, maxlen=n)
            self.buf_y_ = deque(self.buf_y_, maxlen=n)
        self.buf_x_.append(pair[0])
        self.buf_y_.append(pair[1])
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingCovariance":
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def window_size(self) -> int:
        return self._resolve_window()

    @property
    def cov(self) -> float:
        n = len(self.buf_x_)
        if n < 2:
            return 0.0
        mx = sum(self.buf_x_) / n
        my = sum(self.buf_y_) / n
        c = sum((a - mx) * (b - my) for a, b in zip(self.buf_x_, self.buf_y_))
        return c / (n - 1)

    @property
    def corr(self) -> float:
        n = len(self.buf_x_)
        if n < 2:
            return 0.0
        mx = sum(self.buf_x_) / n
        my = sum(self.buf_y_) / n
        dx = [a - mx for a in self.buf_x_]
        dy = [b - my for b in self.buf_y_]
        num = sum(a * b for a, b in zip(dx, dy))
        den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
        if den == 0.0:
            return 0.0
        return num / den

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "window_size": len(self.buf_x_),
            "cov": self.cov,
            "corr": self.corr,
            "n_missing": self.n_missing_,
        }


class RollingCorrelation(Transformer):
    """Pearson correlation of two channels over a bounded window.

    Same machinery as :class:`RollingCovariance`; exposes ``corr`` only.

    Args:
        x_feature: dict key of the first channel.
        y_feature: dict key of the second channel.
        window: window size in samples (default 20).
        window_s: window size in seconds.

    Examples:
        >>> from dense_armor.utility.stats.dependence import RollingCorrelation
        >>> r = RollingCorrelation("a", "b", window=3)
        >>> for a, b in [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]:
        ...     _ = r.learn_one({"a": a, "b": b})
        >>> round(r.corr, 6)
        -1.0
    """

    def __init__(
        self,
        x_feature: str = "x",
        y_feature: str = "y",
        window: int = 20,
        window_s: float | None = None,
        feature: str | None = None,
    ) -> None:
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.window = window
        self.window_s = window_s
        self.feature = feature
        self._cov = RollingCovariance(
            x_feature=x_feature,
            y_feature=y_feature,
            window=window,
            window_s=window_s,
        )

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingCorrelation":
        self._time_step(t)
        self._cov.learn_one(x, t=t)
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RollingCorrelation":
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self._cov.count

    @property
    def n_missing(self) -> int:
        return self._cov.n_missing

    @property
    def corr(self) -> float:
        return self._cov.corr

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.count,
            "corr": self.corr,
            "n_missing": self.n_missing,
        }


class Autocorrelation(Transformer):
    """Running autocorrelation of a single channel at a fixed lag.

    At time ``t`` compares the current sample with the sample ``lag``
    steps before, using the *running* mean and variance estimated over
    the whole history (Welford). The estimate is the biased one,
    ``rho_k = C_k / (n_k · var_n)``, where ``C_k`` accumulates the
    products of the two delayed copies over the samples that are at
    least ``lag`` apart and ``var_n`` is the biased running variance.

    Detects periodicity: a joint oscillating at a fixed frequency shows
    a peak in ``rho_k`` at the period in samples.

    Args:
        feature: dict key to read.
        lag: lag in samples, ``lag >= 1``.

    Examples:
        >>> from dense_armor.utility.stats.dependence import Autocorrelation
        >>> import numpy as np
        >>> ac = Autocorrelation("x", lag=1)
        >>> rng = np.random.default_rng(0)
        >>> for v in rng.normal(0.0, 1.0, 2000):
        ...     _ = ac.learn_one({"x": float(v)})
        >>> abs(ac.value) < 0.1
        True

    References:
        Chatfield, C. (2003). The Analysis of Time Series: An
        Introduction, 6th ed. Chapman & Hall/CRC.
    """

    def __init__(self, feature: str = "x", lag: int = 1) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.feature = feature
        self.lag = lag
        self.history_: deque[float] = deque(maxlen=lag)
        self.n_ = 0
        self.n_pairs_ = 0
        self.n_missing_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0
        self.c_ = 0.0

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "Autocorrelation":
        """Update the running mean, variance and lagged co-moment."""
        self._time_step(t)
        v = float(x[self.feature])
        if math.isnan(v):
            self.n_missing_ += 1
            return self
        n = self.n_ + 1
        d = v - self.mean_
        self.mean_ += d / n
        self.m2_ += d * (v - self.mean_)
        if len(self.history_) == self.lag:
            past = self.history_[0]
            self.c_ += (v - self.mean_) * (past - self.mean_)
            self.n_pairs_ += 1
        self.history_.append(v)
        self.n_ = n
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "Autocorrelation":
        return self.learn_one(x, y=y, t=t)

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_pairs(self) -> int:
        return self.n_pairs_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def value(self) -> float:
        """Biased running autocorrelation at the configured ``lag``."""
        if self.n_pairs_ == 0 or self.n_ == 0:
            return 0.0
        var_n = self.m2_ / self.n_
        if var_n == 0.0:
            return 0.0
        return self.c_ / (self.n_pairs_ * var_n)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "n_pairs": self.n_pairs_,
            "value": self.value,
            "n_missing": self.n_missing_,
        }
