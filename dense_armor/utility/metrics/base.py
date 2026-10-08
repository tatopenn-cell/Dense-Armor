"""The base class every Dense-Armor metric inherits from.

A metric is fed one ``(y_true, y_pred)`` pair at a time, keeps only a
few counters, and answers ``get()`` at any moment. If a pair is
``NaN`` or ``None`` on either side, the metric skips it and increments
``n_missing``; the state is never poisoned.

Every metric also declares the role it applies to through the
``role_`` class attribute. ``works_with(model)`` answers the question
for a concrete model.
"""

import math
from typing import Any

import numpy as np


def _is_missing(v: Any) -> bool:
    """True if ``v`` should be skipped: ``None``, NaN, or empty 0-d."""
    if v is None:
        return True
    if isinstance(v, np.floating):
        return bool(np.isnan(v))
    if isinstance(v, float):
        return math.isnan(v)
    try:
        arr = np.asarray(v)
    except (TypeError, ValueError):
        return False
    if arr.ndim == 0 and arr.dtype.kind in "fc":
        return bool(np.isnan(arr))
    return False


class Metric:
    """Base class for online metrics.

    Subclasses override ``_update``, ``_revert``, ``get`` and ``merge``;
    the base handles the validity check, the count and the timestamp.

    Attributes:
        role_: role class the metric applies to, or ``None`` for
            role-agnostic metrics.
        bigger_is_better: ``True`` for accuracy-like metrics, ``False``
            for loss-like metrics. Model selection uses it to rank.
    """

    role_: Any = None
    bigger_is_better: bool = True

    def __init__(self) -> None:
        self.n_ = 0
        self.n_missing_ = 0
        self.t_first_: float | None = None
        self.t_last_: float | None = None

    @staticmethod
    def _is_valid(v: Any) -> bool:
        return not _is_missing(v)

    def _time_step(self, t: float | None) -> None:
        if t is None:
            return
        t = float(t)
        if self.t_first_ is None:
            self.t_first_ = t
        self.t_last_ = t

    def update(
        self,
        y_true: Any,
        y_pred: Any,
        t: float | None = None,
        w: float = 1.0,
    ) -> "Metric":
        """Feed one pair into the metric.

        Args:
            y_true: the observed label.
            y_pred: the predicted label.
            t: timestamp in seconds; not required by every metric.
            w: sample weight, default 1.0.

        Returns:
            ``self``, for chaining.
        """
        self._time_step(t)
        if _is_missing(y_true) or _is_missing(y_pred):
            self.n_missing_ += 1
            return self
        self._update(y_true, y_pred, float(w))
        self.n_ += 1
        return self

    def revert(self, y_true: Any, y_pred: Any, w: float = 1.0) -> "Metric":
        """Undo one ``update``. Used by the rolling wrappers.

        Args:
            y_true: the label that was passed to ``update``.
            y_pred: the prediction that was passed to ``update``.
            w: the same weight used in ``update``.

        Returns:
            ``self``, for chaining.
        """
        if _is_missing(y_true) or _is_missing(y_pred):
            return self
        self._revert(y_true, y_pred, float(w))
        self.n_ -= 1
        return self

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        raise NotImplementedError

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support revert"
        )

    def get(self) -> Any:
        """Return the current value of the metric.

        Most metrics return a ``float``. :class:`ConfusionMatrix` is
        the one exception: it returns a nested dict
        ``{y_true: {y_pred: count}}``, which is why the return type
        here is ``Any``. Concrete subclasses refine it.
        """
        raise NotImplementedError

    def merge(self, other: "Metric") -> "Metric":
        """Combine two partial states into the summary of the union."""
        raise NotImplementedError

    def works_with(self, model: Any) -> bool:
        """Return ``True`` if the metric applies to ``model``'s role."""
        if self.role_ is None:
            return True
        return isinstance(model, self.role_)

    @property
    def n(self) -> int:
        return self.n_

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n_}, value={self.get()!r})"
