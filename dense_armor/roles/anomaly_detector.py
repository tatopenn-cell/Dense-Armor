"""Anomaly detector role (high score = anomalous)."""

from dense_armor.roles._util import call_with_t, sum_budgets, worst_memory_class
from dense_armor.roles.estimator import Estimator


class AnomalyDetector(Estimator):
    """Unsupervised anomaly detector; ``score_one`` high = anomalous."""

    _supervised = False

    def learn_one(self, x: dict, t: float | None = None) -> "AnomalyDetector":
        raise NotImplementedError

    def score_one(self, x: dict, t: float | None = None) -> float:
        raise NotImplementedError

    def learn_many(self, X, t=None) -> "AnomalyDetector":
        for x in X:
            call_with_t(self.learn_one, x, t=t)
        return self

    def score_many(self, X, t=None):
        return [call_with_t(self.score_one, x, t=t) for x in X]


class AnomalyGate(Estimator):
    """Detector + classification, with optional protection of the detector.

    A score above the threshold is anomalous. The threshold is the
    ``threshold`` argument if it is given, otherwise the ``threshold``
    attribute of the detector, read at every call so that a detector that
    adapts its threshold is followed. A detector with no ``threshold`` and
    no argument is refused: a default of ``0.0`` would classify as anomalous
    every positive score, and with ``protect=True`` the detector would stop
    learning after the first sample.

    Args:
        detector: an ``AnomalyDetector``.
        protect: when True, the detector does not learn from samples it
            classifies as anomalous.
        threshold: the score above which a sample is anomalous. ``None``
            reads the ``threshold`` attribute of the detector.

    A subclass that defines its own ``classify`` decides by itself and needs
    neither the argument nor the attribute.

    Raises:
        ValueError: if ``threshold`` is ``None``, the detector has no
            ``threshold`` attribute and ``classify`` is not overridden.

    Examples:
        >>> from dense_armor.roles.anomaly_detector import AnomalyDetector, AnomalyGate
        >>> class Always(AnomalyDetector):
        ...     def score_one(self, x, t=None):
        ...         return 2.0
        >>> AnomalyGate(Always(), threshold=3.0).classify(2.0)
        False
        >>> AnomalyGate(Always())
        Traceback (most recent call last):
            ...
        ValueError: Always has no 'threshold' attribute: pass threshold= to AnomalyGate
    """

    _supervised = False

    def __init__(
        self,
        detector: "AnomalyDetector",
        protect: bool = True,
        threshold: float | None = None,
    ):
        decides_itself = type(self).classify is not AnomalyGate.classify
        if (
            threshold is None
            and not decides_itself
            and not hasattr(detector, "threshold")
        ):
            raise ValueError(
                f"{type(detector).__name__} has no 'threshold' attribute: "
                "pass threshold= to AnomalyGate"
            )
        self.detector = detector
        self.protect = protect
        self.threshold = threshold

    def classify(self, score: float) -> bool:
        """``True`` if ``score`` is above the threshold."""
        if self.threshold is not None:
            return score > self.threshold
        return score > float(getattr(self.detector, "threshold"))

    @property
    def budget_s(self) -> float | None:
        """Latency of ``learn_one``: a ``score_one`` and a ``learn_one`` of the detector.

        ``None`` if the detector declares no ``budget_s``.
        """
        return sum_budgets([(2.0, self.detector)])

    @property
    def memory_class(self) -> str:
        """The memory class of the detector."""
        return worst_memory_class([self.detector])

    def learn_one(self, x: dict, t: float | None = None) -> "AnomalyGate":
        s = call_with_t(self.detector.score_one, x, t=t)
        if not (self.protect and self.classify(s)):
            call_with_t(self.detector.learn_one, x, t=t)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        return call_with_t(self.detector.score_one, x, t=t)
