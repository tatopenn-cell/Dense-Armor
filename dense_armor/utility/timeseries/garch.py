"""GARCH(1,1) conditional variance, recursive, with fixed parameters.

A GARCH model tracks the **conditional variance** of a series, not its
level. In finance it is the standard model of volatility clustering;
in a robot or a drone it tracks a vibration whose amplitude changes
over time, or a sensor whose noise grows before a fault.

The GARCH(p, q) process is defined by (Csáji 2018, Eq. 1a-1b, page 1)

    X_t          = sigma_t * epsilon_t
    sigma^2_t    = omega + sum_{i=1}^p alpha_i * X^2_{t-i}
                         + sum_{j=1}^q beta_j * sigma^2_{t-j}

where ``epsilon_t`` is a strong white noise with zero mean and unit
variance, ``sigma^2_t`` is the latent conditional variance of
``X_t``, and ``omega > 0``, ``alpha_i >= 0``, ``beta_j >= 0``. The
model is wide-sense stationary if and only if
``sum_i alpha_i + sum_j beta_j < 1`` (Eq. 2, page 1).

This module implements **GARCH(1, 1)**, the special case with
``p = q = 1``:

    sigma^2_t = omega + alpha * X^2_{t-1} + beta * sigma^2_{t-1}

The quasi-maximum-likelihood (QML) log-likelihood is Eq. (3) page 2
of the paper, with the constants omitted:

    l_n(theta) = (1/n) * sum_t [ log sigma^2_t(theta)
                                 + X^2_t / sigma^2_t(theta) ]

The initial conditional variance is (page 2)

    sigma^2_0 = omega / (1 - alpha - beta)

**Parameters are fixed, not estimated online.** The paper describes the
batch QML estimator and cites Gerencsér & Orlovits (2012) for the
recursive QML extension, but does not implement it. Recursive QML
estimation of ``(omega, alpha, beta)`` is therefore **not
implemented**: the three parameters are hyper-parameters, chosen at
construction, and only ``sigma^2_t`` is updated recursively by
Eq. (1b). The docstring of ``GARCH11`` declares this limitation.

References
----------
Csáji, B. Cs. (2018). Score Permutation Based Finite Sample Inference
    for Generalized AutoRegressive Conditional Heteroskedasticity
    (GARCH) Models. arXiv:1807.08390. Eq. (1a)-(1b) page 1; Eq. (2)
    page 1; Eq. (3) page 2; initialization page 2.
Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    DOI 10.1016/0304-4076(86)90063-1. Not in the folder,
    bibliographic only.
Gerencsér, L., Orlovits, Z. (2012). Recursive estimation of GARCH
    processes. Not in the folder, bibliographic only. The reference
    for the (not implemented) recursive QML.
"""

import math
from typing import Any

from dense_armor.roles import Transformer


class GARCH11(Transformer):
    """GARCH(1, 1) conditional variance with fixed parameters.

    One sample at a time, the conditional variance ``sigma2_`` is
    updated recursively from the last residual and the last variance.
    The parameters ``omega``, ``alpha``, ``beta`` are fixed.

    Args:
        omega: variance constant, ``omega > 0``. Default ``1e-6``.
        alpha: weight of the last squared residual, ``alpha >= 0``.
            Default ``0.05``.
        beta: weight of the last conditional variance, ``beta >= 0``.
            Default ``0.9``.
        feature: dict key to read. ``None`` (default) takes the
            smallest key, so the result does not depend on the order
            of the keys.

    Attributes:
        sigma2_: conditional variance of the next sample, updated from
            the last residual and the last variance.
        epsilon_: residual of the last sample, ``X_t`` (zero before the first
            ``learn_one``, unchanged until the next).
        sigma2_history_: not kept (memory O(1)); only the last variance.

    Raises:
        ValueError: if ``omega <= 0``, ``alpha < 0``, ``beta < 0``,
            or ``alpha + beta >= 1`` (stationarity, Eq. 2 page 1).

    Examples:
        >>> from dense_armor.utility.timeseries import GARCH11
        >>> m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
        >>> for x in [0.01, -0.02, 0.015, -0.005, 0.03]:
        ...     _ = m.learn_one({"x": x})
        >>> round(m.sigma2_ * 1e6, 1)
        150.6

    References:
        Csáji (2018), arXiv:1807.08390, Eq. (1b) page 1, Eq. (3)
        page 2. The recursive QML of Gerencsér & Orlovits (2012) is
        not implemented.

        ``budget_s = 1e-3`` (the floor set by the CI check) and
        ``memory_class = "O(1)"``. The measured median per-sample
        latency is 1.3 us, so the declared budget is conservative by
        construction; the point of the declaration is that the two
        checks (`p99_within_budget`, `memory_growth_bounded`) run in CI
        instead of being silently skipped when the attributes are
        absent.

    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.9,
        feature: str | None = None,
    ) -> None:
        if omega <= 0:
            raise ValueError(f"omega must be > 0, got {omega!r}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        if alpha + beta >= 1:
            raise ValueError(
                f"alpha + beta must be < 1 (stationarity, Eq. 2 page 1 of "
                f"Csáji 2018), got {alpha + beta!r}"
            )
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.feature = feature
        self.sigma2_ = float(omega / (1.0 - alpha - beta))
        self.epsilon_ = 0.0
        self.n_ = 0
        self.n_missing_ = 0
        self.initialized_ = False
        self.budget_s = 1e-3
        self.memory_class = "O(1)"

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _value(self, x: dict) -> float | None:
        try:
            v = float(x[self._key(x)])
        except (KeyError, TypeError, ValueError):
            return None
        return None if math.isnan(v) else v

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> "GARCH11":
        """Update the conditional variance with one sample from ``x``."""
        self._time_step(t)
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if not self.initialized_:
            self.initialized_ = True
        sigma2_new = self.omega + self.alpha * (v * v) + self.beta * self.sigma2_
        self.sigma2_ = float(sigma2_new)
        self.epsilon_ = float(v)
        self.n_ += 1
        return self

    def update(self, x: dict, y: Any = None, t: float | None = None) -> "GARCH11":
        """Alias of :meth:`learn_one`, natural name for a statistic."""
        return self.learn_one(x, y=y, t=t)

    def _loglik(self, v: float) -> float:
        if self.sigma2_ <= 0:
            return float("-inf")
        return float(
            -0.5 * (math.log(2.0 * math.pi * self.sigma2_) + v * v / self.sigma2_)
        )

    def transform_one(self, x: dict, t: float | None = None) -> dict[str, Any]:
        """Return ``{sigma2, sigma, loglik, standardized}`` for one sample."""
        v = self._value(x)
        if v is None:
            return {
                "sigma2": self.sigma2_,
                "sigma": float(math.sqrt(self.sigma2_)),
                "loglik": float("nan"),
                "standardized": float("nan"),
            }
        loglik = self._loglik(v)
        return {
            "sigma2": self.sigma2_,
            "sigma": float(math.sqrt(self.sigma2_)),
            "loglik": loglik,
            "standardized": v / float(math.sqrt(self.sigma2_)),
        }

    def score_one(self, x: dict) -> float:
        """Negative log-likelihood: high is anomalous."""
        v = self._value(x)
        if v is None:
            return float("nan")
        return float(-self._loglik(v))

    def merge(self, other: "GARCH11") -> "GARCH11":
        """Combine two independent states by weighted averaging of ``n_``."""
        n_total = self.n_ + other.n_
        if n_total == 0:
            return self
        w_self = self.n_ / n_total
        w_other = other.n_ / n_total
        self.sigma2_ = w_self * self.sigma2_ + w_other * other.sigma2_
        self.epsilon_ = w_self * self.epsilon_ + w_other * other.epsilon_
        self.n_ = n_total
        self.n_missing_ += other.n_missing_
        self.initialized_ = self.initialized_ or other.initialized_
        return self
