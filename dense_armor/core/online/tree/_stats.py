"""Running statistics shared by the tree classes: a Gaussian per (leaf, class, feature)."""

import math
from typing import Any

import numpy as np


def _two_sum(a, b):
    """Error-free transformation ``a + b = s + e`` in the dtype of ``a``."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _phi(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class _Gauss:
    """Running Gaussian for one (leaf, class, feature) triple.

    With ``dtype=None`` (default) the update is Welford's and the
    result is bit for bit the one of the previous version. With
    ``dtype=numpy.float32`` the values are shifted by ``K = x[0]``, the
    shift of the shifted-data algorithm of Healy (2022), Introduction
    and Algorithm 2, PDF pages 2 and 4. Healy accumulates the sums
    ``S = sum(y)`` and ``T = sum(y**2)`` with Kahan's compensation and
    returns ``V = T/n - (M - K)**2``. This class is a variant: it keeps
    Welford's recurrence on ``y = x - K`` (the mean of ``y`` and ``M2``),
    and ``TwoSum`` carries the rounding error of the update of the mean
    and of the update of ``M2``. ``var`` is ``M2 / n`` as in the
    default path. ``m2_`` and ``mean_`` are kept as float64 so the tree
    code that reads them does not change.
    """

    __slots__ = (
        "m2_",
        "mean_",
        "n_",
        "_f32",
        "_K",
        "_mean_y",
        "_c_mean",
        "_m2_y",
        "_c_m2",
        "dtype",
    )

    def __init__(self, dtype=None) -> None:
        self.dtype = dtype
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0

        self._f32 = dtype is not None and np.dtype(dtype) == np.float32
        self._K: Any = None
        self._mean_y: Any = np.float32(0.0)
        self._c_mean = np.float32(0.0)
        self._m2_y: Any = np.float32(0.0)
        self._c_m2: Any = np.float32(0.0)

    def update(self, v: float) -> None:
        if not self._f32:
            self.n_ += 1
            d = v - self.mean_
            self.mean_ += d / self.n_
            self.m2_ += d * (v - self.mean_)
            return

        vf = np.float32(v)
        if self._K is None:
            self._K = vf
        y = np.float32(vf - self._K)
        self.n_ += 1
        n = np.float32(self.n_)
        delta = np.float32(y - self._mean_y)
        d_over_n = np.float32(delta / n)
        s, e = _two_sum(self._mean_y, d_over_n)
        self._c_mean = np.float32(self._c_mean + e)
        self._mean_y = s
        term = np.float32(delta * np.float32(y - self._mean_y))
        s2, e2 = _two_sum(self._m2_y, term)
        self._c_m2 = np.float32(self._c_m2 + e2)
        self._m2_y = s2
        my = np.float32(self._mean_y + self._c_mean)
        m2 = np.float32(self._m2_y + self._c_m2)
        self.mean_ = float(self._K + my)
        self.m2_ = float(m2)

    @property
    def var(self) -> float:
        if self._f32:
            if self.n_ <= 1:
                return 0.0

            n = np.float32(self.n_)
            m2 = np.float32(self._m2_y + self._c_m2)
            return float(np.float32(m2 / n))
        return self.m2_ / self.n_ if self.n_ > 1 else 0.0
