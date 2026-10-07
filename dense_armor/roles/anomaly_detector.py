"""Anomaly detector role (high score = anomalous)."""

from dense_armor.roles._util import call_with_t
from dense_armor.roles.estimator import Estimator


class AnomalyDetector(Estimator):
    """Unsupervised anomaly detector; ``score_one`` high = anomalous."""

    _supervised = False

    def learn_one(self, x: dict, t: float | None = None) -> 'AnomalyDetector':
        raise NotImplementedError

    def score_one(self, x: dict, t: float | None = None) -> float:
        raise NotImplementedError

    def learn_many(self, X, t=None) -> 'AnomalyDetector':
        for x in X:
            call_with_t(self.learn_one, x, t=t)
        return self

    def score_many(self, X, t=None):
        return [call_with_t(self.score_one, x, t=t) for x in X]


class AnomalyGate(Estimator):
    """Detector + classification, with optional protection of the detector.

    Args:
        detector: an ``AnomalyDetector``.
        protect: when True, the detector does not learn from samples it
            classifies as anomalous.
    """

    _supervised = False

    def __init__(self, detector: 'AnomalyDetector', protect: bool = True):
        self.detector = detector
        self.protect = protect

    def classify(self, score: float) -> bool:
        return score > getattr(self.detector, "threshold", 0.0)

    def learn_one(self, x: dict, t: float | None = None) -> 'AnomalyGate':
        s = call_with_t(self.detector.score_one, x, t=t)
        if not (self.protect and self.classify(s)):
            call_with_t(self.detector.learn_one, x, t=t)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        return call_with_t(self.detector.score_one, x, t=t)
