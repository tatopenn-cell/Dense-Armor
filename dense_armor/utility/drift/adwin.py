"""ADWIN drift detector (Bifet & Gavalda 2006).

Two algorithms share this name. The first one (Section 3.2 of the
paper) stores the window as an explicit list of samples and tests
every cut; the cost of ``update`` is O(W) with W the window size.
The second one (Section 3.3 of the paper, ADWIN2) stores the window
as an exponential histogram of buckets: level ``i`` holds buckets
that pack ``2**i`` samples, each level keeps at most ``max_buckets``
buckets, and when a level overflows its two oldest buckets are
merged into one at the level above. The number of buckets is
O(log W), so ``update`` costs O(log W) worst case and O(1)
amortised.

The cut threshold. For a partition of the window into a historical
part ``W0`` (with ``n0`` samples, mean ``mu0``) and a recent part
``W1`` (with ``n1`` samples, mean ``mu1``), with ``m = n0 n1 /
(n0 + n1)`` the harmonic-mean-like factor, ``R`` the observed range
of the window, and ``delta' = delta / ln(n)`` the multiple-
hypothesis correction of Section 3.2 ("only O(log n) subwindows are
checked, which justifies using delta' = delta / ln(n)"), the window
is dropped when

    |mu0 - mu1| > R sqrt( ln(4/delta') / (2 m) ).

Both sides carry the units of the data, so the detector is scale-
invariant.

One change can be flagged on several consecutive updates. At each update
the oldest buckets are dropped while some cut passes the test (Algorithm 1
of the paper); a change that has only just started leaves part of the old
regime in the window, and as more samples of the new regime arrive the
test passes again a few updates later. Act on the first flag of a burst,
or start a new detector after it, if one event per change is wanted. Before
this version the detector flagged a change once, later.

Memory: O(max_buckets * log W).
"""

import math

from dense_armor.roles import DriftDetector


class _Bucket:
    """A run of consecutive samples stored as one unit.

    Args:
        total: sum of the samples in the bucket.
        total_sq: sum of the squares of the samples in the bucket.
        n: number of samples in the bucket.
        size: bucket level; ``2**size`` samples fit in the bucket.
    """

    __slots__ = ("n", "size", "total", "total_sq")

    def __init__(self, total: float, total_sq: float, n: int, size: int) -> None:
        self.total = total
        self.total_sq = total_sq
        self.n = n
        self.size = size


class ADWIN(DriftDetector):
    """Adaptive Windowing drift detector.

    Args:
        delta: confidence parameter, in ``(0, 1)``. Default ``0.002``.
        max_window: maximum window size before the oldest samples start
            being dropped. Default ``200``.
        min_sub: minimum sub-window size considered during the cut
            scan. Default ``2``.
        max_buckets: maximum number of buckets kept per level before the
            two oldest are merged into one at the level above.
            Default ``5``.

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
        329
    """

    def __init__(self, delta: float = 0.002, max_window: int = 200,
                 min_sub: int = 2, max_buckets: int = 5) -> None:
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
        self.max_buckets = max_buckets
        self._buckets: list[list[_Bucket]] = [[]]
        self._total = 0.0
        self._total_sq = 0.0
        self._n = 0
        self._min = math.inf
        self._max = -math.inf
        self.n_missing_ = 0

    def _reset(self) -> None:
        super()._reset()
        self._buckets = [[]]
        self._total = 0.0
        self._total_sq = 0.0
        self._n = 0
        self._min = math.inf
        self._max = -math.inf
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _add(self, x: float) -> None:
        self._buckets[0].append(_Bucket(x, x * x, 1, 1))
        self._total += x
        self._total_sq += x * x
        self._n += 1
        self._min = min(self._min, x)
        self._max = max(self._max, x)
        i = 0
        while i < len(self._buckets) and len(self._buckets[i]) > self.max_buckets:
            b1 = self._buckets[i].pop(0)
            b2 = self._buckets[i].pop(0)
            if i + 1 >= len(self._buckets):
                self._buckets.append([])
            self._buckets[i + 1].append(
                _Bucket(b1.total + b2.total, b1.total_sq + b2.total_sq,
                        b1.n + b2.n, b1.size + b2.size)
            )
            i += 1

    def _remove_oldest(self) -> None:
        for level in range(len(self._buckets) - 1, -1, -1):
            if self._buckets[level]:
                b = self._buckets[level].pop(0)
                self._total -= b.total
                self._total_sq -= b.total_sq
                self._n -= b.n
                return

    def _drift(self) -> bool:
        n = self._n
        if n < 2 * self.min_sub:
            return False
        r = self._max - self._min
        if r <= 1e-12:
            r = 1.0
        ln_n = math.log(n) if n > 1 else 1.0
        delta_prime = self.delta / ln_n if ln_n > 0.0 else self.delta
        coef = 0.5 * r * r * math.log(4.0 / delta_prime)
        total = self._total
        min_sub = self.min_sub
        buckets = self._buckets
        last_level = len(buckets) - 1
        left_sum = 0.0
        left_n = 0
        last_bucket = None
        for level in range(last_level, -1, -1):
            level_buckets = buckets[level]
            for b in level_buckets:
                if last_bucket is not None:
                    left_sum += last_bucket.total
                    left_n += last_bucket.n
                    n0 = left_n
                    n1 = n - left_n
                    if n0 >= min_sub and n1 >= min_sub:
                        diff = left_sum / n0 - (total - left_sum) / n1
                        m = n0 * n1 / (n0 + n1)
                        if diff * diff * m > coef:
                            return True
                last_bucket = b
        return False

    def update(self, x: float, t: float | None = None) -> "ADWIN":
        self._drift_detected = False
        if not math.isfinite(x):
            self.n_missing_ += 1
            return self
        self._add(float(x))
        if self._n > self.max_window:
            self._remove_oldest()
        while self._drift():
            self._drift_detected = True
            self._remove_oldest()
        return self
