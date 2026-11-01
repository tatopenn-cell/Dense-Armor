"""KSWIN drift detector (Raab, Heusinger, Schleif 2020).

KSWIN keeps a sliding window of size ``window_size``. The most recent
``stat_size`` samples are the new sample; a second sample of the same
size is drawn at random from the older part of the window. The two are
compared with a two-sample Kolmogorov-Smirnov test; a drift is flagged
when the p-value falls below ``alpha``. The random draw is seeded so a
run is reproducible.

The paper does not specify what to do with the window after an alarm.
This implementation keeps only the last ``stat_size`` samples, so a
single change produces a single alarm: the window has to refill to
``window_size`` before the KS test runs again, and the refill uses
post-change samples only.
"""

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from dense_armor.roles import DriftDetector


class KSWIN(DriftDetector):
    """Kolmogorov-Smirnov Windowing drift detector.

    Args:
        alpha: significance level of the two-sample KS test. Default
            ``0.0001`` (the paper's value).
        window_size: total sliding window length. Default ``300``
            (the paper's value).
        stat_size: size of each of the two samples compared by KS.
            Default ``30`` (the paper's value).
        seed: seed of the random generator used to draw the reference
            sample. Default ``None``.

    Raises:
        ValueError: if ``stat_size >= window_size``.
    """

    def __init__(self, alpha: float = 0.0001, window_size: int = 300,
                 stat_size: int = 30, seed: int | None = None) -> None:
        super().__init__()
        if stat_size >= window_size:
            raise ValueError(
                f"stat_size ({stat_size}) must be < window_size "
                f"({window_size})"
            )
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._window: deque[float] = deque(maxlen=window_size)
        self.n_missing_ = 0

    def _reset(self) -> None:
        super()._reset()
        self._window = deque(maxlen=self.window_size)
        self._rng = np.random.default_rng(self.seed)
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def update(self, x: float, t: float | None = None) -> "KSWIN":
        self._drift_detected = False
        if not np.isfinite(x):
            self.n_missing_ += 1
            return self
        self._window.append(float(x))
        if len(self._window) < self.window_size:
            return self
        arr = np.asarray(self._window, dtype=float)
        new = arr[-self.stat_size:]
        pool = arr[:-self.stat_size]
        if pool.size < self.stat_size:
            return self
        idx = self._rng.choice(pool.size, size=self.stat_size,
                               replace=False)
        ref = pool[idx]
        _, p = ks_2samp(new, ref)
        if p < self.alpha:
            self._drift_detected = True
            keep = list(self._window)[-self.stat_size:]
            self._window = deque(keep, maxlen=self.window_size)
        return self
