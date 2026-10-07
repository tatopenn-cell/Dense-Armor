"""Regressor role (supports vector targets for joint torques)."""

from __future__ import annotations

from typing import Any

from dense_armor.base._util import accepts_kwarg, call_with_t
from dense_armor.base.estimator import Estimator


class Regressor(Estimator):
    """Supervised regressor; scalar or vector target."""

    _supervised = True

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> Regressor:
        raise NotImplementedError

    def predict_one(self, x: dict, t: float | None = None, return_std: bool = False):
        raise NotImplementedError

    def learn_many(self, X, y, t=None) -> Regressor:
        for i in range(len(X)):
            call_with_t(self.learn_one, X[i], y[i], t=t)
        return self

    def predict_many(self, X, t=None, return_std=False):
        out = []
        for x in X:
            if return_std and accepts_kwarg(self.predict_one, "return_std"):
                out.append(call_with_t(self.predict_one, x, t=t, return_std=True))
            else:
                out.append(call_with_t(self.predict_one, x, t=t))
        return out
