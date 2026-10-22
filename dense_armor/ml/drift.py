"""Change detection on a whole series.

References:
    Page, E. S. (1954). Continuous inspection schemes, Biometrika 41(1-2):
        100-115: the CUSUM with a fixed reference decided once from an
        in-control baseline (not in the folder, bibliographic only).
    Hawkins, D. M., Olwell, D. H. (1998). Cumulative Sum Charts and Charting
        for Quality Improvement, Springer: the slack ``k`` as half the shift to
        detect (not in the folder, bibliographic only).
    The recursions below are the ones written in the docstring of
    ``cusum_detector`` of the library.
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["CUSUMDriftDetector"]

_STREAM = "the reference defines the test on a stream, with a window or a running statistic that moves at each new value: the algorithm of the reference is written for one value at a time"
STOPPED = {
    "ADWIN": _STREAM,
    "KSWIN": _STREAM,
    "PageHinkley": _STREAM,
    "BayesianOnlineChangepoint": "Algorithm 1 of Adams and MacKay (2007) is the recursion on the run length that receives one value at a time; the cited pages give the recursion and the run-length posterior of the stream",
}


class CUSUMDriftDetector(MLBase):
    """Two-sided CUSUM of Page with a fixed reference, run on the whole series.

    The median and the scale ``1.4826 * MAD`` are computed once from the first
    ``radius * ref_mult`` values and kept fixed. Then, for every later value,
    ``z = (x - median) / scale``, ``S+ = max(0, S+ + z - k)``,
    ``S- = min(0, S- + z + k)``; the value is flagged when ``S+ > h`` or
    ``S- < -h``, and the sum that crossed restarts from 0. Values before the
    reference span, non-finite values, and a series whose reference scale is
    below ``eps`` are skipped.

    Args:
        radius: with ``ref_mult``, the reference span is ``radius * ref_mult``.
        ref_mult: see ``radius``.
        k: slack, in scale units (half the shift to detect).
        h: decision threshold, in accumulated scale units.
        eps: smallest scale accepted.

    Example:
        >>> x = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0] * 5 + [6.0] * 5
        >>> m = CUSUMDriftDetector(radius=3, ref_mult=2, k=0.5, h=5.0).fit(x)
        >>> np.flatnonzero(m.flagged_).tolist()
        [31, 33]
    """

    def __init__(
        self,
        radius: int = 10,
        ref_mult: int = 3,
        k: float = 0.5,
        h: float = 20.0,
        eps: float = 1e-9,
    ) -> None:
        self.radius = radius
        self.ref_mult = ref_mult
        self.k = k
        self.h = h
        self.eps = eps

    def fit(self, X, y=None):  # type: ignore[override]
        x = np.asarray(X, dtype=float).ravel()
        n = x.size
        span = self.radius * self.ref_mult
        self.flagged_ = np.zeros(n, dtype=bool)
        self.cusum_ = np.zeros(n)
        self.median_ = self.scale_ = float("nan")
        if n <= span or not np.all(np.isfinite(x[:span])):
            return self
        w = x[:span]
        med = float(np.median(w))
        scale = 1.4826 * float(np.median(np.abs(w - med)))
        self.median_, self.scale_ = med, scale
        if not np.isfinite(scale) or scale < self.eps:
            return self
        sp = sn = 0.0
        for i in range(span, n):
            if not np.isfinite(x[i]):
                continue
            z = (x[i] - med) / scale
            sp = max(0.0, sp + z - self.k)
            sn = min(0.0, sn + z + self.k)
            self.cusum_[i] = sp if sp >= abs(sn) else sn
            if sp > self.h:
                self.flagged_[i] = True
                sp = 0.0
            elif sn < -self.h:
                self.flagged_[i] = True
                sn = 0.0
        return self
