"""The cut threshold of ADWIN, against the practical threshold of the paper.

``ADWIN`` cuts its window when the two parts differ by more than
``R * sqrt(ln(4 / delta') / (2 m))`` with ``delta' = delta / ln n`` (the form of Hoeffding). The paper
(Bifet and Gavalda, equation 3.1) uses in practice a threshold with the observed variance of the window,
``sqrt(2 / m * var * ln(2 / delta')) + 2 / (3 m) * ln(2 / delta')``, with the same ``delta'``. The paper
has the values in [0, 1]; here the second term is multiplied by the range ``R`` of the window because the
streams are not in [0, 1].

``PaperADWIN`` below replaces only the threshold: same buckets, same cuts. Both detectors go through the
same seeded streams: 300 samples of N(0, 1) and then 300 samples of N(shift, 1). Each cell is the number of
streams, out of 20, where the shift was found, and the median delay in samples. The last lines count the false
alarms on 50 stationary streams of 2000 samples.

Run from the repository root:

    python benchmarks/adwin_threshold.py
"""

import math

import numpy as np

from dense_armor.core.online.drift.adwin import ADWIN


class PaperADWIN(ADWIN):
    """ADWIN with the practical threshold of the paper (equation 3.1), scaled by the range of the window."""

    def _drift(self):
        n = self._n
        if n < 2 * self.min_sub:
            return False
        r = self._max - self._min
        if r <= 1e-12:
            r = 1.0
        mean = self._total / n
        var = max(self._total_sq / n - mean * mean, 0.0)
        dp = self.delta / (math.log(n) if n > 1 else 1.0)
        lg = math.log(2.0 / dp)
        left_sum = 0.0
        left_n = 0
        prev_total = 0.0
        prev_n = 0
        first = True
        for level in range(len(self._buckets) - 1, -1, -1):
            for b in self._buckets[level]:
                if not first:
                    left_sum += prev_total
                    left_n += prev_n
                    n1 = n - left_n
                    if left_n >= self.min_sub and n1 >= self.min_sub:
                        m = left_n * n1 / (left_n + n1)
                        eps = math.sqrt(2.0 / m * var * lg) + 2.0 * r / (3.0 * m) * lg
                        if abs(left_sum / left_n - (self._total - left_sum) / n1) > eps:
                            return True
                first = False
                prev_total = b.total
                prev_n = b.n
        return False


def delay(cls, s, shift, sig2=1.0, spike=None, mw=None):
    r = np.random.default_rng(s)
    a = r.normal(0, 1, 300)
    if spike:
        a[290] = spike
    y = np.concatenate([a, r.normal(shift, sig2, 300)])
    det = cls(max_window=mw)
    return next((i - 300 for i, v in enumerate(y) if det.update(v).drift_detected and i >= 300), None)


def row(cls, **kw):
    d = [v for v in (delay(cls, s, **kw) for s in range(20)) if v is not None]
    return len(d), (float(np.median(d)) if d else None)


def false_alarms(cls, draw):
    n = 0
    for s in range(50):
        det = cls(max_window=None)
        n += sum(det.update(v).drift_detected for v in draw(np.random.default_rng(s)))
    return n


if __name__ == "__main__":
    for name, cls in (("library", ADWIN), ("paper, equation 3.1", PaperADWIN)):
        print(name)
        for mw in (100, 200, None):
            print("  max_window", mw, {sh: row(cls, shift=sh, mw=mw) for sh in (2.0, 3.0, 4.0)})
        print("  3 sigma, one value of 20 before the step:", row(cls, shift=3.0, spike=20.0))
        print("  variance from 1 to 3, same mean:", row(cls, shift=0.0, sig2=3.0))
        print("  false alarms, normal:", false_alarms(cls, lambda r: r.normal(0, 1, 2000)))
        print("  false alarms, heavy tails (t, 3 degrees):", false_alarms(cls, lambda r: r.standard_t(3, 2000)))
