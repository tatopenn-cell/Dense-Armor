"""Classifier role."""

from __future__ import annotations

from typing import Any

from dense_armor.base._util import call_with_t
from dense_armor.base.estimator import Estimator


class Classifier(Estimator):
    """Supervised classifier.

    ``predict_one`` returns ``None`` and ``predict_proba_one`` returns
    ``{}`` until the first ``learn_one``, for every classifier.
    """

    _supervised = True

    @property
    def _multiclass(self) -> bool:
        return False

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> Classifier:
        raise NotImplementedError

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict[Any, float]:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide probabilities"
        )

    def predict_one(self, x: dict, t: float | None = None):
        proba = call_with_t(self.predict_proba_one, x, t=t)
        if not proba:
            return None
        return max(proba, key=proba.get)

    def learn_many(self, X, y, t=None) -> Classifier:
        for i in range(len(X)):
            call_with_t(self.learn_one, X[i], y[i], t=t)
        return self

    def predict_many(self, X, t=None):
        return [call_with_t(self.predict_one, x, t=t) for x in X]

    def predict_proba_many(self, X, t=None):
        return [call_with_t(self.predict_proba_one, x, t=t) for x in X]
