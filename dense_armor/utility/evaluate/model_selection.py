"""Online model selection.

Run several models on the same stream with the same prequential
protocol, keep a per-model metric, and answer which one is currently
best. ``best_of`` runs them all; ``current_best`` returns the index of
the currently best model.

References
----------
Ksieniewicz, P., Zyblewski, P. (2020). Stream-learn: open-source Python
    library for difficult data stream batch analysis. arXiv:2001.11077.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from dense_armor.utility.metrics.base import Metric


class OnlineModelSelection:
    """Track several models on a stream and expose the current best.

    Args:
        models: the estimators.
        metric_factory: a callable returning a **fresh** metric. The
            selection uses one metric per model.
        delay: label delay forwarded to :func:`progressive_val_score`.
        every: if given, records the trace every ``every`` samples.

    Raises:
        ValueError: if ``models`` is empty.
    """

    def __init__(
        self,
        models: Sequence[Any],
        metric_factory: Callable[[], Metric],
        delay: float | None = None,
        every: int | None = None,
    ) -> None:
        if not models:
            raise ValueError("models must be non-empty")
        self.models = list(models)
        self.metric_factory = metric_factory
        self.metrics: list[Metric] = [metric_factory() for _ in models]
        self.delay = delay
        self.every = every

    def run(
        self, stream: Iterable[Any]
    ) -> list[Metric]:
        """Run every model on the same stream.

        Args:
            stream: the iterable of ``(x, y)`` pairs. It is materialised
                once so every model sees the same samples in the same
                order.

        Returns:
            The list of metrics, one per model.
        """
        from dense_armor.utility.evaluate.prequential import (
            progressive_val_score,
        )

        samples = list(stream)
        for model, metric in zip(self.models, self.metrics):
            progressive_val_score(
                samples, model, metric, delay=self.delay
            )
        return self.metrics

    def scores(self) -> list[float]:
        """Current metric value per model."""
        return [m.get() for m in self.metrics]

    def current_best(self) -> int:
        """Index of the model with the best metric value.

        ``bigger_is_better`` of the metric is respected.

        Raises:
            ValueError: if the models have not been run yet.
        """
        if not self.metrics:
            raise ValueError("no metrics")
        sign = 1.0 if self.metrics[0].bigger_is_better else -1.0
        scores = [sign * m.get() for m in self.metrics]
        return max(range(len(scores)), key=scores.__getitem__)

    def best_model(self) -> Any:
        """The model with the best metric value."""
        return self.models[self.current_best()]


def best_of(
    models: Sequence[Any],
    metric_factory: Callable[[], Metric],
    stream: Iterable[Any] | None = None,
    delay: float | None = None,
) -> OnlineModelSelection:
    """Return an :class:`OnlineModelSelection` for ``models``.

    If ``stream`` is given, the selection is run immediately on it.

    Args:
        models: the estimators.
        metric_factory: a callable returning a fresh metric.
        stream: optional stream to run the selection on.
        delay: label delay for the stream.

    Returns:
        The :class:`OnlineModelSelection`.
    """
    sel = OnlineModelSelection(models, metric_factory, delay=delay)
    if stream is not None:
        sel.run(stream)
    return sel
