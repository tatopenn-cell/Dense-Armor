"""Protected learning: skip flagged samples, fall back to last good prediction."""

from typing import Any

from dense_armor.roles._util import call_with_t, sum_budgets, worst_memory_class
from dense_armor.roles.estimator import Estimator


class Protected(Estimator):
    """Score first; on flag, skip learning and return the fallback.

    Args:
        model: the wrapped estimator.
        detector: what decides that a sample is flagged, tried in this
            order. An object with ``classify`` (an ``AnomalyGate``) flags
            when ``classify(score_one(x))`` is true. A drift detector
            (``update`` and ``drift_detected``) flags when it reports a drift
            after it has seen the value of ``feature``; only ``learn_one``
            feeds it, once per sample, and ``predict_one`` and ``score_one``
            read its current state without an ``update``. A detector with
            ``score_one`` and a ``threshold`` attribute flags when the score
            is above the threshold. A detector with only ``score_one`` has no
            way to decide and never flags.
        fallback: value returned for flagged samples before any good
            prediction; after the first unflagged sample the last good
            prediction is used.
        feature: for a drift detector, the key of ``x`` whose value is fed
            to ``update``. ``None`` takes the feature with the smallest key,
            so the result does not depend on the order of the keys, as in
            the other detectors of the library. An anomaly detector reads
            the whole of ``x`` and ignores it.

        train_detector: when True, ``learn_one`` also calls
            ``detector.learn_one(x)`` on the samples that are not flagged, so
            the detector learns what is normal from the same stream and
            nothing else has to feed it. When False (the default) the
            detector is only read: whoever builds it trains it, and a
            detector that is never trained flags nothing. A drift detector
            has no ``learn_one`` and is fed by ``update`` in both cases.

    Raises:
        KeyError: on a sample without the ``feature`` key, when a drift
            detector is used.
    """

    _supervised = False

    def __init__(
        self,
        model: Estimator,
        detector: Any,
        fallback: Any = None,
        feature: str | None = None,
        train_detector: bool = False,
    ):
        self.model = model
        self.detector = detector
        self.fallback = fallback
        self.feature = feature
        self.train_detector = train_detector
        self._last_good = fallback

    @property
    def budget_s(self) -> float | None:
        """Latency of ``learn_one``: the model, plus the detector once to decide.

        With ``train_detector`` the detector costs twice (it also learns).
        ``None`` if the model or the detector declares no ``budget_s``.
        """
        n = 2.0 if self.train_detector else 1.0
        return sum_budgets([(1.0, self.model), (n, self.detector)])

    @property
    def memory_class(self) -> str:
        """The larger memory class of the model and the detector."""
        return worst_memory_class([self.model, self.detector])

    def _score(self, x: dict) -> float:
        if hasattr(self.detector, "score_one"):
            return float(self.detector.score_one(x))
        return 0.0

    def _flagged(self, x: dict, update: bool = False) -> bool:
        if hasattr(self.detector, "classify"):
            return bool(self.detector.classify(self._score(x)))
        if hasattr(self.detector, "drift_detected"):
            if update:
                self.detector.update(self._watched(x))
            return bool(self.detector.drift_detected)
        if hasattr(self.detector, "score_one") and hasattr(self.detector, "threshold"):
            return bool(self._score(x) > self.detector.threshold)
        return False

    def _watched(self, x: dict) -> float:
        if not x:
            return 0.0
        key = self.feature if self.feature is not None else min(x)
        return float(x[key])

    def learn_one(self, x: dict, y: Any = None, t: float | None = None):
        if self._flagged(x, update=True):
            return self
        m: Any = self.model
        if y is None:
            call_with_t(m.learn_one, x, t=t)
        else:
            call_with_t(m.learn_one, x, y, t=t)
        if self.train_detector and hasattr(self.detector, "learn_one"):
            call_with_t(self.detector.learn_one, x, t=t)
        return self

    def predict_one(self, x: dict, t: float | None = None):
        if self._flagged(x):
            return self._last_good
        m: Any = self.model
        out = call_with_t(m.predict_one, x, t=t)
        self._last_good = out
        return out

    def score_one(self, x: dict, t: float | None = None) -> float:
        if self._flagged(x):
            return float("inf")
        m: Any = self.model
        return call_with_t(m.score_one, x, t=t)
