"""Rolling wrappers for any metric that supports ``revert``.

A rolling metric keeps only the last ``window`` samples in its state.
It does that by using the underlying metric's ``revert``: every time a
new sample comes in and the window is full, the oldest sample is
removed from the metric before the new one is added.

Two forms:

- ``Rolling(metric, window=n)`` --- window of ``n`` samples.
- ``Rolling(metric, window_s=w)`` --- window of ``w`` seconds. Samples
  must carry a timestamp ``t``; the window holds the entries with
  ``t_now - t_entry < w``, so ``w = 0.05`` at 100 Hz holds 5 samples.
"""

from collections import deque
from typing import Any

from dense_armor.utility.metrics.base import Metric


class Rolling(Metric):
    """Rolling window over any metric with ``revert``.

    Args:
        metric: the underlying metric. It must implement ``revert``.
        window: window length in samples. Exactly one of ``window`` and
            ``window_s`` must be given.
        window_s: window length in seconds. The window slides on the
            timestamps as they come in.

    Raises:
        ValueError: if neither or both of ``window`` and ``window_s``
            are given, or if ``window < 1``, or if ``window_s <= 0``.

    Examples:
        >>> from dense_armor.utility.metrics import Accuracy
        >>> from dense_armor.utility.metrics.rolling import Rolling
        >>> m = Rolling(Accuracy(), window=5)
        >>> for a, b in [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 1)]:
        ...     _ = m.update(a, b)
        >>> m.get()
        0.8
    """

    def __init__(
        self,
        metric: Metric,
        window: int | None = None,
        window_s: float | None = None,
    ) -> None:
        super().__init__()
        if (window is None) == (window_s is None):
            raise ValueError(
                "exactly one of 'window' and 'window_s' must be given"
            )
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if window_s is not None and window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.metric = metric
        self.window = window
        self.window_s = window_s
        self.role_ = metric.role_
        self.bigger_is_better = metric.bigger_is_better
        self.entries_: deque[tuple[float | None, Any, Any, float]] = deque()
        self._last_t_: float | None = None

    def _evict_samples(self) -> None:
        if self.window is None:
            return
        while len(self.entries_) > self.window:
            _t, yt, yp, ww = self.entries_.popleft()
            self.metric.revert(yt, yp, w=ww)

    def _evict_seconds(self, t_now: float) -> None:
        if self.window_s is None:
            return
        while self.entries_:
            t_entry = self.entries_[0][0]
            if t_entry is None or t_now - t_entry < self.window_s - 1e-9:
                break
            _t, yt, yp, ww = self.entries_.popleft()
            self.metric.revert(yt, yp, w=ww)

    def update(
        self,
        y_true: Any,
        y_pred: Any,
        t: float | None = None,
        w: float = 1.0,
    ) -> "Rolling":
        """Add a sample and evict the entries that left the window.

        Args:
            y_true: the observed label.
            y_pred: the predicted label.
            t: timestamp in seconds; required for the ``window_s`` mode.
            w: sample weight.

        Returns:
            ``self``.

        Raises:
            ValueError: if ``t`` is not increasing.
        """
        if t is not None:
            t = float(t)
            if self._last_t_ is not None and t <= self._last_t_:
                raise ValueError(
                    f"non-increasing t: previous {self._last_t_}, got {t}"
                )
            self._last_t_ = t
        self.entries_.append((t, y_true, y_pred, w))
        self.metric.update(y_true, y_pred, t=t, w=w)
        self.n_ += 1
        if self.window is not None:
            self._evict_samples()
        elif self.window_s is not None and t is not None:
            self._evict_seconds(t)
        return self

    def revert(self, y_true: Any, y_pred: Any, w: float = 1.0) -> "Rolling":
        """Undo one update, dropping the newest entry.

        Args:
            y_true: the label that was passed to ``update``.
            y_pred: the prediction that was passed to ``update``.
            w: the same weight used in ``update``.

        Returns:
            ``self``.
        """
        if self.entries_:
            self.entries_.pop()
        self.metric.revert(y_true, y_pred, w=w)
        self.n_ -= 1
        return self

    def get(self) -> Any:
        return self.metric.get()

    def merge(self, other: Metric) -> "Rolling":
        """Combine two rolling windows.

        Only the last ``window`` entries of the union are kept; the
        intermediate contributions to the underlying metric are not
        recovered, so this is a pragmatic merge for reporting, not the
        exact state of the union.

        Args:
            other: another :class:`Rolling` with the same configuration.

        Returns:
            A new :class:`Rolling` with the combined window.

        Raises:
            TypeError: if ``other`` is not a :class:`Rolling`.
            ValueError: if the configuration does not match.
        """
        if not isinstance(other, Rolling):
            raise TypeError(
                f"merge expects Rolling, got {type(other).__name__}"
            )
        if (self.window, self.window_s) != (other.window, other.window_s):
            raise ValueError("merge requires the same window configuration")
        m = Rolling(
            self.metric.merge(other.metric),
            window=self.window,
            window_s=self.window_s,
        )
        union = list(self.entries_) + list(other.entries_)
        if self.window is not None:
            union = union[-self.window:]
        m.entries_ = deque(union)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m

    @property
    def count(self) -> int:
        return len(self.entries_)
