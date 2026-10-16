"""Numba-accelerated inner loop for ADWIN (optional).

This module is imported only if ``numba`` is available. The public
``ADWIN`` class falls back to the pure-Python scan when numba is
absent or when the user passes ``use_jit=False``.
"""

import math

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _drift_scan(flat_n, flat_total, total, n, min_val, max_val,
                    delta, min_sub):
        """Return True if any cut of the bucket sequence exceeds the bound.

        ``flat_n[i]`` and ``flat_total[i]`` are the number of samples and
        the sum of samples of the i-th bucket, ordered from the oldest to
        the most recent. The scan is exactly the pure-Python loop of
        ``ADWIN._drift``, compiled.
        """
        if n < 2 * min_sub:
            return False
        r = max_val - min_val
        if r <= 1e-12:
            r = 1.0
        ln_n = math.log(n) if n > 1 else 1.0
        delta_prime = delta / ln_n if ln_n > 0.0 else delta
        coef = 0.5 * r * r * math.log(4.0 / delta_prime)
        k_total = len(flat_n)
        left_sum = 0.0
        left_n = 0
        for k in range(k_total - 1):
            left_sum += flat_total[k]
            left_n += flat_n[k]
            n0 = left_n
            n1 = n - left_n
            if n0 < min_sub or n1 < min_sub:
                continue
            diff = left_sum / n0 - (total - left_sum) / n1
            m = n0 * n1 / (n0 + n1)
            if diff * diff * m > coef:
                return True
        return False

else:

    def _drift_scan(*args, **kwargs):
        raise RuntimeError("numba is not installed")
