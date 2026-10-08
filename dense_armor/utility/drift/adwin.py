"""ADWIN drift detector (Bifet & Gavaldà 2007).

The original reference is not on arXiv; the description used here is
the one in Lu, Liu, Dong, Gu, Gama, Zhang (2018), "Learning under
Concept Drift: A Review", Section 3.2.1. ADWIN keeps a window W,
considers every cut of W into a historical part W_hist and a new part
W_new, and drops W_hist as soon as the two sub-window means differ by
more than a Hoeffding-style bound.

Derivation of the bound used here. For values in a range R, the
two-sample Hoeffding bound for one cut gives

    P(|mean0 - mean1| >= eps) <= 2 exp(-2 m eps^2 / R^2),

with m half the harmonic mean of the two sub-window sizes,

    m = 1 / (1 / n0 + 1 / n1) = n0 n1 / (n0 + n1).

A union bound over the n cuts scanned at one step, with total level
delta, gives 2 n exp(-2 m eps^2 / R^2) <= delta, hence

    eps_cut = R * sqrt(ln(2 n / delta) / (2 m)).

R is estimated online from the observed range of the window
(max - min), so the detector does not depend on the units of the
stream.

Memory: this implementation stores the whole window as a Python list,
so memory is O(W) with W the maximum window size. Each `update` costs
O(W) for the cut scan, with prefix sums so every cut is O(1).
"""

import math

import numpy as np

from dense_armor.roles import DriftDetector


class ADWIN(DriftDetector):
    """Adaptive Windowing drift detector.

    Args:
        delta: confidence parameter, in ``(0, 1)``. Default ``0.002``.
        max_window: maximum window size before the oldest samples start
            being dropped. Default ``200``.
        min_sub: minimum sub-window size considered during the cut
            scan. Default ``2``.

    Raises:
        ValueError: if ``delta`` is not in ``(0, 1)`` or ``max_window``
            is too small.

    Examples:
        >>> from dense_armor.utility.drift.adwin import ADWIN
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> det = ADWIN(delta=0.002, max_window=100)
        >>> stream = list(rng.normal(0, 1, 300)) + list(rng.normal(4, 1, 300))
        >>> for i, x in enumerate(stream):
        ...     _ = det.update(x)
        ...     if det.drift_detected:
        ...         print(i)
        ...         break
        339
    """

    def __init__(self, delta: float = 0.002, max_window: int = 200,
                 min_sub: int = 2) -> None:
        super().__init__()
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if max_window < 4 * min_sub:
            raise ValueError(
                f"max_window must be at least {4 * min_sub}, "
                f"got {max_window}"
            )
        self.delta = delta
        self.max_window = max_window
        self.min_sub = min_sub
        self._window: list[float] = []
        self.n_missing_ = 0

    def _reset(self) -> None:
        super()._reset()
        self._window = []
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @staticmethod
    def _range(arr: np.ndarray) -> float:
        r = float(np.max(arr) - np.min(arr))
        if r <= 1e-12:
            r = 1.0
        return r

    def _best_cut(self) -> int | None:
        n = len(self._window)
        if n < 2 * self.min_sub:
            return None
        arr = np.asarray(self._window, dtype=float)
        s1 = np.concatenate(([0.0], np.cumsum(arr)))
        total = float(s1[-1])
        r = self._range(arr)
        n_cuts = n - 2 * self.min_sub + 1
        log_term = math.log(2.0 * n_cuts / self.delta)
        best_cut = None
        best_excess = 0.0
        for cut in range(self.min_sub, n - self.min_sub + 1):
            n0 = cut
            n1 = n - cut
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            mean0 = float(s1[cut]) / n0
            mean1 = (total - float(s1[cut])) / n1
            eps = r * math.sqrt(log_term / (2.0 * m))
            excess = abs(mean0 - mean1) - eps
            if excess > best_excess:
                best_excess = excess
                best_cut = cut
        return best_cut

    def update(self, x: float, t: float | None = None) -> "ADWIN":
        self._drift_detected = False
        if not np.isfinite(x):
            self.n_missing_ += 1
            return self
        self._window.append(float(x))
        cut = self._best_cut()
        if cut is not None:
            self._window = self._window[cut:]
            self._drift_detected = True
            return self
        if len(self._window) > self.max_window:
            self._window.pop(0)
        return self
