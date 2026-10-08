"""Prequential (test-then-train) evaluation, with optional label delay.

In the prequential protocol the model sees each sample in order:
predict first, score against the available label, then learn from it.

When labels are delayed, the prediction is made at time ``t`` but the
label arrives at ``t + dt``. The protocol buffers the sample until the
label arrives, then scores it and learns from it. The model never sees
the label at prediction time. This is the delayed-label setting of
Amekoe et al. (2024).

References
----------
Ksieniewicz, P., Zyblewski, P. (2020). Stream-learn. arXiv:2001.11077.
Amekoe, K. M., Lebbah, M., Jaffre, G., Azzag, H., Chelly Dagdia, Z.
    (2024). arXiv:2409.10111.
"""

from collections.abc import Iterable
from typing import Any

from dense_armor.roles import Classifier, Signal
from dense_armor.utility.metrics.base import Metric

_PENDING = tuple[Any, Any, Any, float | None]


def _timestamp_of(x: Any) -> float | None:
    if isinstance(x, Signal):
        return x.t
    return None


def _split_ready(
    buffer: list[_PENDING],
    delay: float | None,
    now: float | None,
) -> tuple[list[_PENDING], list[_PENDING]]:
    if delay is None:
        return list(buffer), []
    if isinstance(delay, int):
        n_ready = max(0, len(buffer) - delay)
        return buffer[:n_ready], buffer[n_ready:]
    if now is None:
        return [], list(buffer)
    ready: list[_PENDING] = []
    rest: list[_PENDING] = []
    for item in buffer:
        t = item[3]
        if t is not None and now - t >= delay:
            ready.append(item)
        else:
            rest.append(item)
    return ready, rest


def _learn_or_score(
    model: Any,
    metric: Metric | None,
    x: Any,
    y: Any,
    pred: Any,
    t: float | None,
) -> None:
    if metric is not None:
        metric.update(y, pred, t=t)
    model.learn_one(x, y, t=t)


def progressive_val_score(
    stream: Iterable[Any],
    model: Any,
    metric: Metric,
    delay: float | None = None,
    every: int | None = None,
) -> Metric | tuple[Metric, list[float]]:
    """Score a model on a stream, test first, then train.

    Each element of ``stream`` is a pair ``(x, y)``. For every sample,
    in order:

    1. the model predicts (if it exposes ``predict_one``);
    2. the sample is buffered with its prediction;
    3. the samples whose delay has expired are scored against their
       (now available) label and learned from.

    With ``delay=None`` every sample is scored and learned immediately.
    With an integer ``delay`` the last ``delay`` samples wait in the
    buffer; with a float the delay is measured in seconds on the
    timestamps.

    Args:
        stream: an iterable of ``(x, y)`` pairs.
        model: the estimator.
        metric: the online metric to update.
        delay: label delay in samples (int) or seconds (float).
        every: if given, records the metric every ``every`` samples
            and returns the trace.

    Returns:
        The metric, or ``(metric, trace)`` when ``every`` is given.

    Raises:
        ValueError: if ``every`` is given and is not positive.

    Example:
        >>> from dense_armor.utility.evaluate import progressive_val_score
        >>> from dense_armor.utility.metrics import Accuracy
        >>> class LastValue:
        ...     def __init__(self): self._last = 0
        ...     def learn_one(self, x, y, t=None):
        ...         self._last = y; return self
        ...     def predict_one(self, x, t=None): return self._last
        >>> stream = [({}, 1)] * 20
        >>> m = progressive_val_score(stream, LastValue(), Accuracy())
        >>> round(m.get(), 3)
        0.95
    """
    if every is not None and every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    buffer: list[_PENDING] = []
    trace: list[float] = []
    for i, (x, y) in enumerate(stream, 1):
        t = _timestamp_of(x)
        pred = (
            model.predict_one(x, t=t)
            if hasattr(model, "predict_one")
            else None
        )
        buffer.append((x, y, pred, t))
        ready, buffer = _split_ready(buffer, delay, t)
        for rx, ry, rpred, rt in ready:
            _learn_or_score(model, metric, rx, ry, rpred, rt)
        if every is not None and i % every == 0:
            trace.append(metric.get())
    for rx, ry, rpred, rt in buffer:
        _learn_or_score(model, metric, rx, ry, rpred, rt)
    if every is None:
        return metric
    return metric, trace


def progressive_val_proba_score(
    stream: Iterable[Any],
    classifier: Classifier,
    metric: Metric,
    delay: float | None = None,
    every: int | None = None,
) -> Metric | tuple[Metric, list[float]]:
    """Same as :func:`progressive_val_score`, but scores a probability.

    Use when the metric needs a probability (``LogLoss``,
    ``BrierScore``) and the classifier exposes ``predict_proba_one``.
    The class ``positive=1`` is used as the probability of the positive
    class; if the classifier is not binary, the metric must select the
    class itself.

    Args:
        stream: an iterable of ``(x, y)`` pairs.
        classifier: a classifier exposing ``predict_proba_one``.
        metric: an online metric.
        delay: label delay as in :func:`progressive_val_score`.
        every: if given, records the metric every ``every`` samples.

    Returns:
        The metric, or ``(metric, trace)``.
    """
    if every is not None and every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    buffer: list[_PENDING] = []
    trace: list[float] = []
    for i, (x, y) in enumerate(stream, 1):
        t = _timestamp_of(x)
        proba = classifier.predict_proba_one(x, t=t)
        p = proba.get(1, proba.get(True, 0.5)) if proba else 0.5
        buffer.append((x, y, p, t))
        ready, buffer = _split_ready(buffer, delay, t)
        for rx, ry, rp, rt in ready:
            metric.update(ry, rp, t=rt)
            classifier.learn_one(rx, ry, t=rt)
        if every is not None and i % every == 0:
            trace.append(metric.get())
    for rx, ry, rp, rt in buffer:
        metric.update(ry, rp, t=rt)
        classifier.learn_one(rx, ry, t=rt)
    if every is None:
        return metric
    return metric, trace
