"""Page-Hinkley test for a change in the mean of a scalar stream.

Page, E. S. (1954). Continuous inspection schemes. Biometrika
41(1-2), 100-115. The formulation used here is the log-likelihood-ratio
CUSUM written down in Xie, Zou, Xie, Veeravalli (2021), "Sequential
(Quickest) Change Detection: Classical Results and New Directions",
Section II-C2: the log-likelihood ratio l(X) is accumulated and
compared with its running minimum (eq. 2),

    W_n = S_n - min_{0 <= k <= n} S_k,   S_n = sum_{k=1}^{n} l(X_k),

and the detector alarms when W_n exceeds a threshold b (eq. 3, the
maximum-likelihood form). With a Gaussian reference of unit variance
the log-likelihood ratio reduces to (x - mu_ref) / sigma_ref - delta,
where mu_ref is the running mean, sigma_ref the running standard
deviation of the stream, and delta is the slack in scale units.
Working in scale units makes delta and threshold independent of the
stream units, so the same parameters detect the same change on x and
on 1e-4 * x.
"""

import math

import numpy as np

from dense_armor.roles import DriftDetector


class PageHinkley(DriftDetector):
    """Page-Hinkley test for a change in the mean of a scalar stream.

    Args:
        delta: reference slack, in scale units (multiples of the
            running standard deviation). Classical choice: half the
            shift size worth detecting. Default ``0.005``.
        threshold: alarm threshold on the accumulated deviation from
            its running minimum, in scale units. Default ``50.0``.
        two_sided: if ``True`` both upward and downward shifts are
            monitored; if ``False`` only upward. Default ``True``.
        warmup: number of samples skipped before the first check.
            Default ``30``.

    Examples:
        >>> from dense_armor.utility.drift.page_hinkley import PageHinkley
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> det = PageHinkley(delta=0.05, threshold=20.0)
        >>> stream = list(rng.normal(0, 1, 500)) + list(rng.normal(2, 1, 500))
        >>> for i, x in enumerate(stream):
        ...     _ = det.update(x)
        ...     if det.drift_detected:
        ...         print(i)
        ...         break
        508
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0,
                 two_sided: bool = True, warmup: int = 30) -> None:
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.two_sided = two_sided
        self.warmup = warmup
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._mean = 0.0
        self._n = 0
        self._s_pos = 0.0
        self._min_pos = 0.0
        self._s_neg = 0.0
        self._min_neg = 0.0
        self.n_missing_ = 0

    def _reset(self) -> None:
        super()._reset()
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._mean = 0.0
        self._n = 0
        self._s_pos = 0.0
        self._min_pos = 0.0
        self._s_neg = 0.0
        self._min_neg = 0.0
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def update(self, x: float, t: float | None = None) -> "PageHinkley":
        self._drift_detected = False
        if not np.isfinite(x):
            self.n_missing_ += 1
            return self
        x = float(x)
        self._n += 1
        self._sum_x += x
        self._sum_x2 += x * x
        self._mean = self._sum_x / self._n
        var = max(0.0, self._sum_x2 / self._n - self._mean * self._mean)
        scale = math.sqrt(var) if var > 1e-12 else 1.0
        if self._n < self.warmup:
            return self
        z = (x - self._mean) / scale
        self._s_pos += z - self.delta
        if self.two_sided:
            self._s_neg += -z - self.delta
        self._min_pos = min(self._min_pos, self._s_pos)
        if self.two_sided and self._s_neg < self._min_neg:
            self._min_neg = self._s_neg
        if self._s_pos - self._min_pos > self.threshold:
            self._drift_detected = True
            self._s_pos = 0.0
            self._min_pos = 0.0
        elif self.two_sided and self._s_neg - self._min_neg > self.threshold:
            self._drift_detected = True
            self._s_neg = 0.0
            self._min_neg = 0.0
        return self
