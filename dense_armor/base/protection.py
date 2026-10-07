"""Protected learning: skip flagged samples, fall back to last good prediction."""

from __future__ import annotations

from typing import Any

from dense_armor.base.estimator import Estimator


class Protected(Estimator):
    """Score first; on flag, skip learning and return the fallback.

    Args:
        model: the wrapped estimator.
        detector: an anomaly or drift detector with ``score_one`` or
            ``update``.
        fallback: value returned for flagged samples before any good
            prediction; after the first unflagged sample the last good
            prediction is used.
    """

    _supervised = False

    def __init__(self, model: Estimator, detector: Any, fallback: Any = None):
        self.model = model
        self.detector = detector
        self.fallback = fallback
        self._last_good = fallback

    def _score(self, x: dict) -> float:
        if hasattr(self.detector, "score_one"):
            return float(self.detector.score_one(x))
        return 0.0

    def _flagged(self, x: dict) -> bool:
        if hasattr(self.detector, "classify"):
            return bool(self.detector.classify(self._score(x)))
        if hasattr(self.detector, "drift_detected"):
            self.detector.update(float(next(iter(x.values()), 0.0)))
            return bool(self.detector.drift_detected)
        return False

    def learn_one(self, x: dict, y: Any = None, t: float | None = None):
        if self._flagged(x):
            return self
        m: Any = self.model
        if y is None:
            m.learn_one(x, t=t)
        else:
            m.learn_one(x, y, t=t)
        return self

    def predict_one(self, x: dict, t: float | None = None):
        if self._flagged(x):
            return self._last_good
        m: Any = self.model
        out = m.predict_one(x, t=t)
        self._last_good = out
        return out

    def score_one(self, x: dict, t: float | None = None) -> float:
        if self._flagged(x):
            return float("inf")
        m: Any = self.model
        return m.score_one(x, t=t)
