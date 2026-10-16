"""Exponentially weighted average of regressors (Hedge).

The Hedge rule of Freund and Schapire (1997) and the exponentiated
gradient of Kivinen and Warmuth (1997) give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i.

The weight update ``w_i = exp(-eta * L_i)`` is the multiplicative
update of Hedge with ``beta = exp(-eta)`` (Freund and Schapire 1997,
Section 2.1). The bound of Theorem 2, specialised to uniform initial
weights (equation 9 of the paper), is

    L_ensemble <= (eta / (1 - exp(-eta))) * min_i L_i
                  + ln(K) / (1 - exp(-eta)),

with ``K`` the number of members. The factor ``eta / (1 - exp(-eta))``
tends to one as ``eta -> 0``, so a small learning rate loses almost
nothing over the best member, and the price of a larger learning rate
is paid by the factor ``ln(K) / (1 - exp(-eta))``. The bound assumes
the loss is in ``[0, 1]``; the class keeps the loss the user passes and
reports the bound as the paper states it, with that assumption.

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
from dense_armor.utility.ensemble._common import as_dict, finite, new_member


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
        >>> from dense_armor.utility.ensemble.ewa import ExponentiallyWeightedAverageRegressor
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

    References:
        Freund, Y., Schapire, R. E. (1997). A decision-theoretic
            generalization of on-line learning and an application to
            boosting. Journal of Computer and System Sciences 55(1),
            119-139. Section 2.1 (Hedge) and Theorem 2, equation 9.
        Kivinen, J., Warmuth, M. K. (1997). Exponentiated gradient versus
            gradient descent for linear predictors. Information and
            Computation 132(1), 1-63. The exponentiated update.
    """

    budget_s = 1e-3
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
        return new_member(m)

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "ExponentiallyWeightedAverageRegressor":
        d = as_dict(x)
        if not d or y is None or not finite(d):
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
        for i, (m, p) in enumerate(zip(self.members_, preds)):
            self.losses_[i] = self.losses_[i] + float(self.loss(y, p))
            try:
                m.learn_one(d, y)
            except (AttributeError, TypeError):
                pass
        self.n_seen_ += 1
        return self

    def bound(self) -> float:
        """Upper bound on the cumulative loss of the ensemble.

        The value is ``(eta / (1 - exp(-eta))) * min_i L_i
        + ln(K) / (1 - exp(-eta))``, the equation 9 of Freund and
        Schapire (1997) with ``beta = exp(-eta)``. It assumes the loss
        is in ``[0, 1]``.
        """
        if not self.losses_:
            return 0.0
        k = len(self.losses_)
        denom = -math.expm1(-self.eta)
        return (self.eta / denom) * min(self.losses_) + math.log(k) / denom

    def predict_one(
        self, x: Any, t: float | None = None, return_std: bool = False
    ) -> Any:
        if not self.members_:
            return 0.0
        d = as_dict(x)
        preds = []
        for m in self.members_:
            try:
                preds.append(float(m.predict_one(d)))
            except (AttributeError, TypeError, ValueError):
                preds.append(0.0)
        l_min = min(self.losses_)
        w = np.array(
            [math.exp(-self.eta * min(l - l_min, 700.0)) for l in self.losses_],
            dtype=float,
        )
        z = float(w.sum())
        y_hat = float(np.dot(w, np.array(preds)) / z)
        if return_std:
            var = float(np.dot(w, (np.array(preds) - y_hat) ** 2) / z)
            return y_hat, math.sqrt(max(var, 0.0))
        return y_hat
