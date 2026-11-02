"""Anomaly detector role (high score = anomalous)."""

from dense_armor.roles._util import call_with_t
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

    Raises:
        ValueError: if ``threshold`` is ``None`` and the detector has no
            ``threshold`` attribute.

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
        if threshold is None and not hasattr(detector, "threshold"):
            raise ValueError(
                f"{type(detector).__name__} has no 'threshold' attribute: "
                "pass threshold= to AnomalyGate"
            )
        self.detector = detector
        self.protect = protect
        self.threshold = threshold

    def classify(self, score: float) -> bool:
        """``True`` if ``score`` is above the threshold."""
        limit = (
            self.threshold if self.threshold is not None else self.detector.threshold
        )
        return score > limit

    def learn_one(self, x: dict, t: float | None = None) -> "AnomalyGate":
        s = call_with_t(self.detector.score_one, x, t=t)
        if not (self.protect and self.classify(s)):
            call_with_t(self.detector.learn_one, x, t=t)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        return call_with_t(self.detector.score_one, x, t=t)
