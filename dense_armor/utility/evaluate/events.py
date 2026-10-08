"""Run a drift or anomaly detector on a stream with labelled events.

The detector reads the stream one sample at a time. Every sample on
which it raises an alarm is registered with its timestamp in an
:class:`~dense_armor.utility.metrics.events.EventMetrics`, which then
reports detection delay, missed events, false alarms per hour,
range-based precision / recall / F1 (Tatbul et al. 2018) and the NAB
score (Lavin and Ahmad 2015).

References
----------
Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018).
    Precision and recall for time series. In NeurIPS. arXiv:1803.03639.
Lavin, A., Ahmad, S. (2015). Evaluating real-time anomaly detection
    algorithms --- the Numenta Anomaly Benchmark. In IEEE ICMLA.
    arXiv:1510.03336.
"""

from collections.abc import Iterable
from typing import Any

from dense_armor.utility.metrics.events import EventMetrics, EventWindow


def _value_of(x: Any) -> Any:
    if isinstance(x, dict) and len(x) == 1:
        return next(iter(x.values()))
    return x


def _alarm(detector: Any, x: Any, t: float) -> bool:
    if hasattr(detector, "drift_detected"):
        try:
            detector.update(_value_of(x), t=t)
        except TypeError:
            detector.update(_value_of(x))
        return bool(detector.drift_detected)
    if hasattr(detector, "score_one"):
        threshold = getattr(detector, "score_threshold", None)
        if threshold is None:
            raise ValueError(
                "an anomaly detector needs a 'score_threshold' attribute"
            )
        score = detector.score_one(x, t=t)
        detector.learn_one(x, t=t)
        return bool(score > threshold)
    raise TypeError(
        "detector must expose 'update' + 'drift_detected' or "
        "'score_one' + 'score_threshold'"
    )


def evaluate_events(
    stream: Iterable[tuple[float, Any]],
    detector: Any,
    events: Iterable[EventWindow | tuple[float, float]],
    **kwargs: Any,
) -> dict:
    """Run ``detector`` on ``stream`` and report the event metrics.

    Args:
        stream: ``(t, x)`` pairs, ``t`` in seconds, in time order. ``x``
            is a scalar, or a dict passed to ``score_one``; a dict with
            one key is unwrapped for a drift detector's ``update``.
        detector: a drift detector (``update`` + ``drift_detected``) or
            an anomaly detector (``score_one`` + ``learn_one`` +
            ``score_threshold``; the score is taken before learning).
        events: the labelled event windows.
        **kwargs: forwarded to :class:`EventMetrics` (``dt``, ``alpha``,
            ``positional_bias``, ``cardinality``, ``normal_time_s``,
            ``threshold_window``, ``a_tp``, ``a_fp``, ``a_fn``).

    Returns:
        The dict of :meth:`EventMetrics.report`.

    Raises:
        TypeError: if the detector exposes neither interface.
        ValueError: if an anomaly detector has no ``score_threshold``.

    Example:
        >>> from dense_armor.utility.evaluate import evaluate_events
        >>> class TriggerAt:
        ...     def __init__(self, at):
        ...         self.at, self.drift_detected = at, False
        ...     def update(self, v, t=None):
        ...         self.drift_detected = t == self.at
        ...         return self
        >>> stream = [(float(i), float(i)) for i in range(100)]
        >>> r = evaluate_events(stream, TriggerAt(15.0), [(10.0, 20.0)])
        >>> r["mean_delay_s"], r["missed"], r["false_alarms"]
        (5.0, 0, 0)
    """
    em = EventMetrics(events, **kwargs)
    for t, x in stream:
        if _alarm(detector, x, float(t)):
            em.add_alarm(float(t))
    return em.report()
