"""The all-pole filter and the reflection coefficients that make it stable.

Sources:

* Yu, Mitcheltree, Carson, Bilbao, Reiss and Fazekas (2024), "Differentiable all-pole filters
  for time-varying audio systems", DAFx-24, Eq. (1), PDF page 2: the ``M``-th order
  time-varying all-pole filter ``y(n) = x(n) - sum_{i=1}^{M} a_i(n) y(n - i)``;
* Hayes, Shier, Fazekas, McPherson and Saitis (2023), Eq. (5), PDF page 20: the recursion that
  turns reflection coefficients ``k_i`` into the coefficients ``a_m`` of an all-pole filter;
  with ``|k_i| < 1`` the filter is stable.

The two papers do not write the filter with the same sign. Hayes Eq. (4) is
``y[n] = e[n] + sum a_m y[n - m]``, and with that sign the coefficients of Eq. (5) are
**not** stable (1570 of 2000 random cases were not, in a check made for this module); with the
sign of Yu Eq. (1), ``y = x - sum a y``, all 2000 were. This module uses the sign of Yu Eq. (1)
for both, so the output of :func:`reflection_to_lpc` goes straight into :class:`AllPole`.
"""

from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot


def reflection_to_lpc(k) -> np.ndarray:
    """The coefficients ``a_1 .. a_M`` from the reflection coefficients ``k_1 .. k_M`` (Eq. 5).

    ``a_m^(i) = k_i`` if ``m = i`` and ``a_m^(i-1) + k_i a_(i-m)^(i-1)`` otherwise, for
    ``i = 1 .. M``. The result is for the filter ``y = x - sum a_m y[n - m]`` and is stable
    when every ``|k_i| < 1``.

    Args:
        k: reflection coefficients, shape ``(M,)``, at least one.

    Raises:
        ValueError: if ``k`` is not a non-empty 1D array.

    Example:
        >>> reflection_to_lpc([0.5, 0.25]).tolist()
        [0.625, 0.25]
    """
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("k must be a non-empty 1D array")
    a = np.zeros(0)
    for i, ki in enumerate(k, start=1):
        new = np.zeros(i)
        for m in range(1, i):
            new[m - 1] = a[m - 1] + ki * a[i - m - 1]
        new[i - 1] = ki
        a = new
    return a


class AllPole(StatefulRoot):
    """Time-varying all-pole filter, ``y(n) = x(n) - sum_{i=1}^{M} a_i(n) y(n - i)`` (Eq. 1).

    The coefficients ``a`` of the constructor are used when none are given at a call, so
    ``update_many(values)`` is the time-invariant filter. A call can bring its own
    coefficients, which is the time-varying case of the paper. The history holds the last
    ``M`` outputs. Stability is not enforced: use :meth:`from_reflection`, or check ``stable``.

    Args:
        a: the coefficients ``a_1 .. a_M``, at least one.

    Raises:
        ValueError: if ``a`` is not a non-empty 1D array.

    Example:
        >>> f = AllPole([-0.5])
        >>> f.update_many([1.0, 0.0, 0.0, 0.0]).tolist()
        [1.0, 0.5, 0.25, 0.125]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_y", "out")

    def __init__(self, a: Any) -> None:
        a = np.asarray(a, dtype=float)
        if a.ndim != 1 or a.size == 0:
            raise ValueError("a must be a non-empty 1D array")
        self.a = a
        self._y = np.zeros(a.size)
        self.out = 0.0

    @classmethod
    def from_reflection(cls, k: Any) -> "AllPole":
        """The filter whose coefficients come from the reflection coefficients ``k``."""
        return cls(reflection_to_lpc(k))

    @property
    def stable(self) -> bool:
        """True if every pole of the filter with the constructor coefficients is inside the
        unit circle."""
        return bool(np.all(np.abs(np.roots(np.r_[1.0, self.a])) < 1.0))

    def update(self, x: float, a: Any = None) -> "AllPole":
        """Filter one sample; the result is ``out``. ``a`` replaces the coefficients for it."""
        c = self.a if a is None else np.asarray(a, dtype=float)
        if c.shape != self.a.shape:
            raise ValueError(f"a must have shape {self.a.shape}, got {c.shape}")
        y = float(x) - float(np.dot(c, self._y))
        self._y = np.concatenate([[y], self._y[:-1]])
        self.out = y
        return self

    def update_many(self, values: Any, coeffs: Any = None) -> np.ndarray:
        """Filter a block; ``coeffs`` of shape ``(n, M)`` gives the coefficients of each sample."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x, None if coeffs is None else coeffs[i]).out
        return out
