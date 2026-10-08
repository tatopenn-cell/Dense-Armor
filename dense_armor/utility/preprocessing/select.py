"""Online feature selection.

``VarianceThreshold`` drops features whose running variance is below a
threshold. ``SelectKBest`` keeps the ``k`` features with the highest
absolute running Pearson correlation with the target; the ranking is
recomputed on every ``transform_one`` from the current statistics, so
the kept set changes as the stream evolves.
"""
import math
from typing import Any

from dense_armor.roles import Transformer
from dense_armor.utility.stats.moments import RunningMoments


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


class _CorrStats:
    """Running Pearson correlation between one feature and the target."""

    def __init__(self) -> None:
        self.n_ = 0
        self.mean_x_ = 0.0
        self.mean_y_ = 0.0
        self.m2_x_ = 0.0
        self.m2_y_ = 0.0
        self.c_xy_ = 0.0

    def update(self, x: float, y: float) -> None:
        """Add one ``(x, y)`` pair."""
        self.n_ += 1
        n = self.n_
        dx = x - self.mean_x_
        dy = y - self.mean_y_
        self.mean_x_ += dx / n
        self.mean_y_ += dy / n
        self.m2_x_ += dx * (x - self.mean_x_)
        self.m2_y_ += dy * (y - self.mean_y_)
        self.c_xy_ += dx * (y - self.mean_y_)

    @property
    def corr(self) -> float:
        """Pearson correlation, ``0.0`` while the statistics are too thin."""
        if self.n_ < 2 or self.m2_x_ <= 0.0 or self.m2_y_ <= 0.0:
            return 0.0
        return self.c_xy_ / math.sqrt(self.m2_x_ * self.m2_y_)


class VarianceThreshold(Transformer):
    """Drop features whose running variance is at or below ``threshold``.

    The variance is the unbiased running variance. A feature never
    seen passes through unchanged, so the first sample is never
    dropped by accident.

    Memory grows with the number of features (one accumulator per
    feature); the total is ``O(d)`` for ``d`` features.

    Args:
        threshold: variance floor; features with ``var <= threshold``
            are dropped from the output.
        features: explicit feature order.

    Examples:
        >>> from dense_armor.utility.preprocessing.select import VarianceThreshold
        >>> vt = VarianceThreshold(threshold=0.1)
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = vt.learn_one({"x": v, "c": 7.0})
        >>> sorted(vt.transform_one({"x": 3.0, "c": 7.0}).keys())
        ['x']
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, threshold: float = 0.0, features: list[str] | None = None
    ) -> None:
        self.threshold = threshold
        self.features = features
        self.stats_: dict[str, RunningMoments] | None = None

    def _ensure(self, x: dict) -> dict[str, RunningMoments]:
        if self.stats_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.stats_ = {k: RunningMoments(feature=k) for k in keys}
        return self.stats_

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "VarianceThreshold":
        """Update the running variance of each feature."""
        self._time_step(t)
        d = _as_dict(x)
        for k, s in self._ensure(d).items():
            if k in d:
                s.learn_one(d)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Keep only the features above the variance threshold."""
        d = _as_dict(x)
        if self.stats_ is None:
            return d
        out: dict = {}
        for k, v in d.items():
            if k in self.stats_:
                if self.stats_[k].var > self.threshold:
                    out[k] = v
            else:
                out[k] = v
        return out


class SelectKBest(Transformer):
    """Keep the ``k`` features with the highest absolute running correlation.

    ``learn_one`` takes the target ``y`` and updates, for each feature,
    a running Pearson correlation with ``y``. ``transform_one`` ranks
    the features by ``|corr|`` and keeps the top ``k``. Features never
    seen pass through unchanged.

    Memory grows with the number of features (one accumulator per
    feature); the total is ``O(d)`` for ``d`` features.

    Args:
        k: number of features to keep.
        features: explicit feature order.

    Examples:
        >>> from dense_armor.utility.preprocessing.select import SelectKBest
        >>> sk = SelectKBest(k=1)
        >>> for i in range(20):
        ...     _ = sk.learn_one({"a": float(i), "b": 1.0}, y=2.0 * i)
        >>> sorted(sk.transform_one({"a": 1.0, "b": 1.0}).keys())
        ['a']
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, k: int = 10, features: list[str] | None = None
    ) -> None:
        self.k = k
        self.features = features
        self.stats_: dict[str, _CorrStats] | None = None

    def _ensure(self, x: dict) -> dict[str, _CorrStats]:
        if self.stats_ is None:
            keys = (
                list(self.features)
                if self.features is not None
                else sorted(x)
            )
            self.stats_ = {k: _CorrStats() for k in keys}
        return self.stats_

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "SelectKBest":
        """Update the running correlation of each feature with ``y``."""
        self._time_step(t)
        if y is None:
            return self
        d = _as_dict(x)
        stats = self._ensure(d)
        for k, s in stats.items():
            v = d.get(k)
            if isinstance(v, (int, float)):
                s.update(float(v), float(y))
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Keep the ``k`` features with the highest absolute correlation."""
        d = _as_dict(x)
        if self.stats_ is None:
            return d
        ranked = sorted(
            self.stats_.items(), key=lambda kv: -abs(kv[1].corr)
        )
        keep = {k for k, _ in ranked[: self.k]}
        return {k: v for k, v in d.items() if k in keep or k not in self.stats_}
