"""Online per-feature scalers, one sample at a time.

Each scaler is a :class:`~dense_armor.roles.Transformer`: ``learn_one``
updates the running state, ``transform_one`` uses the current state.
Scalars and vectors are both supported: a scalar keeps one statistic,
a vector keeps one statistic per joint. A feature never seen, or with
zero spread, passes through unchanged.

The running mean and variance are updated incrementally with the
stable one-pass formulas of running moments. The exponentially
weighted mean and variance follow the recursive form of exponentially
weighted statistics, where recent samples carry more weight. The
rolling median and median absolute deviation come from a bounded FIFO
window. None of these carries a reference entry: the methods are
standard and the code documents them.
"""
from typing import Any

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.stats.moments import EWStats, RunningMoments
from dense_armor.utility.stats.robust import RollingMAD, RollingMedian


def _as_dict(x: Any) -> dict:
    """Return ``x`` as a plain dict; unwrap a Signal via ``to_dict``."""
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


def _to_vec(v: Any) -> tuple[np.ndarray | None, bool]:
    """Return ``(values, is_scalar)`` or ``(None, True)`` if unsupported."""
    if isinstance(v, bool):
        return None, True
    if isinstance(v, (int, float)):
        return np.asarray([float(v)], dtype=float), True
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return np.asarray([float(v)], dtype=float), True
        try:
            return v.astype(float).ravel(), False
        except (TypeError, ValueError):
            return None, True
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v, dtype=float).ravel(), False
        except (TypeError, ValueError):
            return None, True
    return None, True


def _write_out(values: np.ndarray, is_scalar: bool) -> Any:
    return float(values[0]) if is_scalar else values.tolist()


class StandardScaler(Transformer):
    """Per-feature standardization with running mean and std.

    ``z_i = (x_i - mean_i) / std_i`` for each feature and, when the
    value is a vector, for each joint. A feature (or a joint) seen
    fewer than twice, or whose std is below ``eps``, passes through
    unchanged.

    Memory grows with the number of features (and per-feature joints):
    each feature keeps two floats, so the total is ``O(d)`` for ``d``
    scalar features.

    Args:
        features: explicit feature order. ``None`` sorts the first
            dict's keys.
        eps: std floor below which the feature is left alone.

    Examples:
        >>> from dense_armor.utility.preprocessing.scale import StandardScaler
        >>> sc = StandardScaler()
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = sc.learn_one({"x": v})
        >>> round(sc.transform_one({"x": 5.0})["x"], 4)
        1.2649
        >>> sc2 = StandardScaler()
        >>> for q in ([0.0, 1.0], [1.0, 2.0], [2.0, 3.0]):
        ...     _ = sc2.learn_one({"q": q})
        >>> [round(v, 4) for v in sc2.transform_one({"q": [2.0, 3.0]})["q"]]
        [1.0, 1.0]
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, features: list[str] | None = None, eps: float = 1e-12
    ) -> None:
        self.features = features
        self.eps = eps
        self.stats_: dict[str, list[RunningMoments]] | None = None

    def _ensure(self, x: dict) -> dict[str, list[RunningMoments]]:
        if self.stats_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.stats_ = {}
            for k in keys:
                if k not in x:
                    continue
                vec, _ = _to_vec(x[k])
                if vec is None:
                    continue
                self.stats_[k] = [RunningMoments() for _ in range(vec.size)]
        return self.stats_

    def _update(
        self, d: dict, t: float | None
    ) -> None:
        self._time_step(t)
        stats = self._ensure(d)
        for k, s_list in stats.items():
            if k not in d:
                continue
            vec, _ = _to_vec(d[k])
            if vec is None or vec.size != len(s_list):
                continue
            for i, s in enumerate(s_list):
                s.learn_one({"_v": float(vec[i])})

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "StandardScaler":
        """Update the running mean and std for each feature in ``x``."""
        self._update(_as_dict(x), t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Standardize each feature using the current state."""
        d = _as_dict(x)
        if self.stats_ is None:
            return d
        out = dict(d)
        for k, s_list in self.stats_.items():
            if k not in out:
                continue
            vec, is_scalar = _to_vec(out[k])
            if vec is None or vec.size != len(s_list):
                continue
            res = np.empty(vec.size, dtype=float)
            for i, s in enumerate(s_list):
                if s.count >= 2 and s.std > self.eps:
                    res[i] = (vec[i] - s.mean) / s.std
                else:
                    res[i] = vec[i]
            out[k] = _write_out(res, is_scalar)
        return out


class EWScaler(Transformer):
    """Per-feature standardization with exponentially weighted stats.

    Same shape as :class:`StandardScaler` but the mean and variance
    follow the recursive exponential weighting, so recent samples
    weigh more and a slow drift is tracked instead of averaged out.

    Memory grows with the number of features (``O(d)`` for ``d``
    scalar features); each feature keeps one mean and one variance.

    Args:
        alpha: weight of the new sample in ``(0, 1]``.
        features: explicit feature order.
        eps: std floor.

    Examples:
        >>> from dense_armor.utility.preprocessing.scale import EWScaler
        >>> sc = EWScaler(alpha=0.5)
        >>> for v in [1.0, 2.0, 3.0]:
        ...     _ = sc.learn_one({"x": v})
        >>> got = sc.transform_one({"x": 3.0})["x"]
        >>> round(got, 6)
        0.904534
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self,
        alpha: float = 0.1,
        features: list[str] | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.alpha = alpha
        self.features = features
        self.eps = eps
        self.stats_: dict[str, list[EWStats]] | None = None

    def _ensure(self, x: dict) -> dict[str, list[EWStats]]:
        if self.stats_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.stats_ = {}
            for k in keys:
                if k not in x:
                    continue
                vec, _ = _to_vec(x[k])
                if vec is None:
                    continue
                self.stats_[k] = [
                    EWStats(alpha=self.alpha) for _ in range(vec.size)
                ]
        return self.stats_

    def _update(self, d: dict, t: float | None) -> None:
        self._time_step(t)
        stats = self._ensure(d)
        for k, s_list in stats.items():
            if k not in d:
                continue
            vec, _ = _to_vec(d[k])
            if vec is None or vec.size != len(s_list):
                continue
            for i, s in enumerate(s_list):
                s.learn_one({"_v": float(vec[i])})

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "EWScaler":
        """Update the exponentially weighted stats per feature."""
        self._update(_as_dict(x), t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Standardize with the current exponentially weighted state."""
        d = _as_dict(x)
        if self.stats_ is None:
            return d
        out = dict(d)
        for k, s_list in self.stats_.items():
            if k not in out:
                continue
            vec, is_scalar = _to_vec(out[k])
            if vec is None or vec.size != len(s_list):
                continue
            res = np.empty(vec.size, dtype=float)
            for i, s in enumerate(s_list):
                if s.count >= 1 and s.std > self.eps:
                    res[i] = (vec[i] - s.mean) / s.std
                else:
                    res[i] = vec[i]
            out[k] = _write_out(res, is_scalar)
        return out


class RobustScaler(Transformer):
    """Robust per-feature standardization with rolling median and MAD.

    ``z_i = (x_i - median_i) / MAD_i`` for each feature and, when the
    value is a vector, for each joint. A feature with MAD below
    ``eps`` passes through unchanged.

    Memory grows with the number of features and the window length:
    each feature keeps ``window`` samples, so the total is ``O(d*w)``.

    Args:
        window: rolling window length in samples.
        features: explicit feature order.
        eps: MAD floor.

    Examples:
        >>> from dense_armor.utility.preprocessing.scale import RobustScaler
        >>> sc = RobustScaler(window=5)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = sc.learn_one({"x": v})
        >>> round(sc.transform_one({"x": 5.0})["x"], 4)
        1.349
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        window: int = 20,
        features: list[str] | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.window = window
        self.features = features
        self.eps = eps
        self.medians_: dict[str, list[RollingMedian]] | None = None
        self.mads_: dict[str, list[RollingMAD]] | None = None

    def _ensure(
        self, x: dict
    ) -> tuple[
        dict[str, list[RollingMedian]], dict[str, list[RollingMAD]]
    ]:
        if self.medians_ is None or self.mads_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            meds: dict[str, list[RollingMedian]] = {}
            mads: dict[str, list[RollingMAD]] = {}
            for k in keys:
                if k not in x:
                    continue
                vec, _ = _to_vec(x[k])
                if vec is None:
                    continue
                meds[k] = [
                    RollingMedian(window=self.window) for _ in range(vec.size)
                ]
                mads[k] = [
                    RollingMAD(window=self.window) for _ in range(vec.size)
                ]
            self.medians_ = meds
            self.mads_ = mads
        return self.medians_, self.mads_

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "RobustScaler":
        """Push one sample into both rolling windows per feature."""
        self._time_step(t)
        d = _as_dict(x)
        med, mad = self._ensure(d)
        for k, med_list in med.items():
            if k not in d:
                continue
            vec, _ = _to_vec(d[k])
            if vec is None or vec.size != len(med_list):
                continue
            mad_list = mad[k]
            for i in range(vec.size):
                med_list[i].learn_one({"_v": float(vec[i])}, t=t)
                mad_list[i].learn_one({"_v": float(vec[i])}, t=t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Center by the median and scale by the MAD."""
        d = _as_dict(x)
        if self.medians_ is None or self.mads_ is None:
            return d
        out = dict(d)
        for k, med_list in self.medians_.items():
            if k not in out:
                continue
            vec, is_scalar = _to_vec(out[k])
            if vec is None or vec.size != len(med_list):
                continue
            mad_list = self.mads_[k]
            res = np.empty(vec.size, dtype=float)
            for i, med in enumerate(med_list):
                mad = mad_list[i].value
                if med.count >= 1 and mad > self.eps:
                    res[i] = (vec[i] - med.value) / mad
                else:
                    res[i] = vec[i]
            out[k] = _write_out(res, is_scalar)
        return out


class MinMaxScaler(Transformer):
    """Rescale each feature to ``[0, 1]`` with running min and max.

    ``z_i = (x_i - min_i) / (max_i - min_i)`` for each feature and,
    when the value is a vector, for each joint. A feature with
    ``max == min`` passes through unchanged.

    Memory grows with the number of features: each keeps two floats.

    Args:
        features: explicit feature order.
        eps: spread floor below which the feature is left alone.

    Examples:
        >>> from dense_armor.utility.preprocessing.scale import MinMaxScaler
        >>> sc = MinMaxScaler()
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = sc.learn_one({"x": v})
        >>> sc.transform_one({"x": 5.0})["x"]
        1.0
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, features: list[str] | None = None, eps: float = 1e-12
    ) -> None:
        self.features = features
        self.eps = eps
        self.stats_: dict[str, list[RunningMoments]] | None = None

    def _ensure(self, x: dict) -> dict[str, list[RunningMoments]]:
        if self.stats_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.stats_ = {}
            for k in keys:
                if k not in x:
                    continue
                vec, _ = _to_vec(x[k])
                if vec is None:
                    continue
                self.stats_[k] = [RunningMoments() for _ in range(vec.size)]
        return self.stats_

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "MinMaxScaler":
        """Update the running min and max for each feature."""
        self._time_step(t)
        d = _as_dict(x)
        stats = self._ensure(d)
        for k, s_list in stats.items():
            if k not in d:
                continue
            vec, _ = _to_vec(d[k])
            if vec is None or vec.size != len(s_list):
                continue
            for i, s in enumerate(s_list):
                s.learn_one({"_v": float(vec[i])})
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Shift and rescale each feature to ``[0, 1]``."""
        d = _as_dict(x)
        if self.stats_ is None:
            return d
        out = dict(d)
        for k, s_list in self.stats_.items():
            if k not in out:
                continue
            vec, is_scalar = _to_vec(out[k])
            if vec is None or vec.size != len(s_list):
                continue
            res = np.empty(vec.size, dtype=float)
            for i, s in enumerate(s_list):
                lo, hi = s.min, s.max
                if lo is None or hi is None:
                    res[i] = vec[i]
                    continue
                spread = hi - lo
                if spread > self.eps:
                    res[i] = (vec[i] - lo) / spread
                else:
                    res[i] = vec[i]
            out[k] = _write_out(res, is_scalar)
        return out
