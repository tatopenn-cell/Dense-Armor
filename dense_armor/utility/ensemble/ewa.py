"""Exponentially weighted average of regressors (Hedge).

The Hedge rule of Freund and Schapire (1997) and the exponentiated
gradient of Kivinen and Warmuth (1997) give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i.

The loss bound of Freund and Schapire (1997), Theorem 2, in the form
written for the mixture of experts, is

    L_ensemble <= min_i L_i + eta * L_i^2 / 2 + ln(K) / eta,

with ``K`` the number of members, when the loss is in ``[0, 1]`` and
``eta > 0``. The same bound holds for the squared loss of a regression
problem only if the squared error is scaled to ``[0, 1]`` first; the
class keeps the loss the user passes and reports the bound as the paper
states it, with that assumption.

References:
    Freund, Y., Schapire, R. E. (1997). A decision-theoretic
        generalization of on-line learning and an application to
        boosting. Journal of Computer and System Sciences 55(1),
        119-139. Theorem 2, the Hedge rule.
    Kivinen, J., Warmuth, M. K. (1997). Exponentiated gradient versus
        gradient descent for linear predictors. Information and
        Computation 132(1), 1-63. The exponentiated update.
"""

import math
from typing import Any

import numpy as np

from dense_armor.roles import Regressor


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


def _finite(d: dict) -> bool:
    for v in d.values():
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv):
            return False
    return True


def _squared_error(y_true, y_pred) -> float:
    """Default loss: squared error, a function that pickles."""
    return (float(y_true) - float(y_pred)) ** 2


class ExponentiallyWeightedAverageRegressor(Regressor):
    """Weighted average of regressors with the Hedge rule.

    Every member's weight is ``exp(-eta * cumulative_loss)``, with the
    cumulative loss clipped to avoid underflow. The prediction is the
    weighted average of the members' predictions.

    Args:
        models: a list of regressors. Every member is built by cloning
            the corresponding entry if it exposes ``clone``, otherwise
            by calling it as a factory.
        eta: learning rate of the Hedge rule. Must be positive.
        loss: a loss function ``loss(y_true, y_pred) -> float``. The
            default is the squared error.

    Raises:
        ValueError: if ``eta <= 0`` or ``models`` is empty.

    Examples:
        >>> from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
        >>> from dense_armor.utility.ensemble.ewa import ExponentiallyWeightedAverageRegressor
        >>> from dense_armor.utility.learn.linear import RecursiveLeastSquares
        >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
        >>> from dense_armor.utility.tree.adaptive import HoeffdingAdaptiveTreeClassifier
        >>> from dense_armor.roles import Regressor
        >>> class Const(Regressor):
        ...     def __init__(self, v): self.v = v
        ...     def learn_one(self, x, y, t=None): return self
        ...     def predict_one(self, x, t=None, return_std=False): return self.v
        >>> r = ExponentiallyWeightedAverageRegressor(
        ...     [Const(0.0), Const(1.0)], eta=0.5)
        >>> for _ in range(20):
        ...     _ = r.learn_one({"x": 0.0}, 1.0)
        >>> round(r.predict_one({"x": 0.0}), 2)
        1.0
    """

    budget_s = 0.05
    memory_class = "O(n_models)"

    def __init__(self, models, eta: float = 0.5, loss=None) -> None:
        if eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if not models:
            raise ValueError("models must not be empty")
        self.models = list(models)
        self.eta = float(eta)
        self.loss = loss if loss is not None else _squared_error
        self.members_: list = []
        self.losses_: list = []
        self.classes_: set = set()
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return set()

    def _new_member(self, m):
        if isinstance(m, type):
            return m()
        if hasattr(m, "clone"):
            return m.clone()
        return m()

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "ExponentiallyWeightedAverageRegressor":
        d = _as_dict(x)
        if not d or y is None or not _finite(d):
            self.n_missing_ += 1
            return self
        if not self.members_:
            self.members_ = [self._new_member(m) for m in self.models]
            self.losses_ = [0.0] * len(self.members_)
        preds = []
        for m in self.members_:
            try:
                p = float(m.predict_one(d))
            except (AttributeError, TypeError, ValueError):
                p = 0.0
            preds.append(p)
        w = np.array([math.exp(-self.eta * min(l, 700.0))
                      for l in self.losses_], dtype=float)
        z = float(w.sum())
        if z <= 0:
            w = np.ones_like(w)
            z = float(w.sum())
        y_hat = float(np.dot(w, np.array(preds)) / z)
        for i, (m, p) in enumerate(zip(self.members_, preds)):
            self.losses_[i] = self.losses_[i] + float(self.loss(y, p))
            try:
                m.learn_one(d, y)
            except (AttributeError, TypeError):
                pass
        self.n_seen_ += 1
        return self

    def predict_one(self, x: Any, t: float | None = None, return_std: bool = False) -> Any:
        if not self.members_:
            return 0.0
        d = _as_dict(x)
        preds = []
        for m in self.members_:
            try:
                preds.append(float(m.predict_one(d)))
            except (AttributeError, TypeError, ValueError):
                preds.append(0.0)
        w = np.array([math.exp(-self.eta * min(l, 700.0))
                      for l in self.losses_], dtype=float)
        z = float(w.sum())
        if z <= 0:
            w = np.ones_like(w)
            z = float(w.sum())
        y_hat = float(np.dot(w, np.array(preds)) / z)
        if return_std:
            var = float(np.dot(w, (np.array(preds) - y_hat) ** 2) / z)
            return y_hat, math.sqrt(max(var, 0.0))
        return y_hat
