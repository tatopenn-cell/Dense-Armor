"""Every output with its variance.

A robot cannot act on a number alone: it needs to know how much that
number can be trusted. This module turns point predictions into
:class:`Estimate` objects carrying a mean and a variance, and provides
two adapters that give calibrated uncertainty around any base model:

- :class:`AdaptiveConformalRegressor` — an online wrapper that forms a
  prediction *interval* with adaptive coverage, following the adaptive
  conformal inference of Gibbs & Candes (2021). The coverage level is
  re-estimated at every sample from the recent miscoverage frequency,
  so the interval tracks a drifting stream.
- :class:`OnlinePlattScaling` — the online Platt scaling of Gupta &
  Ramdas (2023), re-exported here for convenience. It calibrates the
  probabilities of a classifier online.

Adaptive conformal inference
----------------------------
Given a score function ``S(x, y)`` (the residual ``|y - f(x)|`` for a
regressor), the base method keeps a running quantile ``Q_t`` of past
scores and predicts the interval ``f(x) +/- Q_t``. The adaptive version
maintains a *target miscoverage* level ``alpha_t`` and updates it every
sample:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where ``alpha`` is the user target, ``gamma > 0`` is a step size, and
``err_t`` is 1 when the previous interval missed ``y_t``. Gibbs &
Candes (2021, eq. 2) show that the long-run empirical miscoverage
converges to ``alpha`` without any assumption on the data-generating
process, and the interval width adapts automatically to a drift.

References
----------
Gibbs, I., Candes, E. J. (2021). Adaptive conformal inference under
    distribution shift. NeurIPS.
Gupta, C., Ramdas, A. (2023). Online Platt scaling with calibeating.
    In ICML. arXiv:2305.00070.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from dense_armor.roles import Regressor
from dense_armor.roles._util import call_with_t
from dense_armor.roles.signal import Signal
from dense_armor.learn.calibration import OnlinePlattScaling

__all__ = [
    "Estimate",
    "AdaptiveConformalRegressor",
    "OnlinePlattScaling",
]


@dataclass
class Estimate:
    """A prediction with its variance.

    Attributes:
        mean: the point prediction.
        var: the variance of the prediction. Use ``std`` for the
            standard deviation.
    """

    mean: float
    var: float

    @property
    def std(self) -> float:
        """Square root of the variance, ``sqrt(max(var, 0))``."""
        return max(self.var, 0.0) ** 0.5

    def __repr__(self) -> str:
        return f"Estimate(mean={self.mean}, std={self.std})"


class AdaptiveConformalRegressor(Regressor):
    """Conformal interval with adaptive coverage, wrapping a regressor.

    At each sample, the base model ``f`` predicts ``yhat``. A sliding
    window of the last ``window`` residuals ``|y - yhat|`` provides the
    running quantile ``Q_t`` at level ``1 - alpha_t``. The interval is
    ``[yhat - Q_t, yhat + Q_t]``. The target miscoverage ``alpha_t`` is
    updated after seeing the outcome:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where ``err_t`` is 1 when the interval missed. Gibbs & Candes (2021,
    eq. 2) prove that the long-run miscoverage converges to the target
    ``alpha`` under any distribution.

    Args:
        model: any regressor exposing ``learn_one`` / ``predict_one``.
        alpha: target miscoverage, e.g. ``0.1`` for a 90% interval.
        gamma: step size of the ``alpha_t`` update. The paper uses small
            values (0.005 in their experiments). Larger values adapt
            faster but produce noisier intervals.
        window: number of past residuals used to compute ``Q_t``.
    """

    budget_s: Optional[float] = 5e-4
    memory_class: str = "O(window)"

    def __init__(
        self,
        model: Regressor,
        alpha: float = 0.1,
        gamma: float = 0.005,
        window: int = 200,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.alpha_t_ = alpha
        self.residuals_: deque[float] = deque(maxlen=window)
        self.n_ = 0
        self.n_miss_ = 0
        self._last_lo_: Optional[float] = None
        self._last_hi_: Optional[float] = None

    def _quantile(self, p: float) -> float:
        if not self.residuals_:
            return 0.0
        s = sorted(self.residuals_)
        if len(s) == 1:
            return s[0]
        pos = p * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (pos - lo) * (s[hi] - s[lo])

    def learn_one(
        self, x: Signal | dict, y: Any, t: Optional[float] = None
    ) -> "AdaptiveConformalRegressor":
        """Update the base model and the miscoverage level.

        The interval for this sample is built from the base model's
        prediction and the running residuals quantile at the current
        ``alpha_t``. Whether ``y`` falls inside determines ``err_t``,
        which drives the ACI update of the paper:

            alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        """
        self._time_step(t)
        yhat = float(call_with_t(self.model.predict_one, x, t=t))
        q = self._quantile(1.0 - self.alpha_t_)
        lo, hi = yhat - q, yhat + q
        err = 0.0 if lo <= float(y) <= hi else 1.0
        self.n_miss_ += int(err)
        self.alpha_t_ = self.alpha_t_ + self.gamma * (self.alpha - err)
        self.alpha_t_ = min(1.0, max(0.0, self.alpha_t_))
        self.residuals_.append(abs(float(y) - yhat))
        try:
            call_with_t(self.model.learn_one, x, y, t=t)
        except TypeError:
            self.model.learn_one(x, y)
        self.n_ += 1
        return self

    def predict_interval(
        self, x: Signal | dict, t: Optional[float] = None
    ) -> tuple[float, float]:
        """Return the current conformal interval ``(lo, hi)``."""
        self._time_step(t)
        proba = float(call_with_t(self.model.predict_one, x, t=t))
        q = self._quantile(1.0 - self.alpha_t_)
        lo, hi = proba - q, proba + q
        self._last_lo_, self._last_hi_ = lo, hi
        return lo, hi

    def predict_one(
        self,
        x: Signal | dict,
        t: Optional[float] = None,
        return_std: bool = False,
        return_estimate: bool = False,
    ) -> Any:
        """Predict one value, optionally with an uncertainty."""
        self._time_step(t)
        mean = float(call_with_t(self.model.predict_one, x, t=t))
        if return_estimate or return_std:
            std = self._quantile(1.0 - self.alpha_t_)
            if return_estimate:
                return Estimate(mean=mean, var=std * std)
            return mean, std
        return mean

    @property
    def coverage(self) -> float:
        """Empirical coverage over the samples seen so far."""
        if self.n_ == 0:
            return 0.0
        return 1.0 - self.n_miss_ / self.n_

    def transform_one(
        self, x: Signal | dict, t: Optional[float] = None
    ) -> dict:
        lo, hi = self.predict_interval(x, t=t)
        return {
            "lo": lo,
            "hi": hi,
            "alpha_t": self.alpha_t_,
            "coverage": self.coverage,
        }
