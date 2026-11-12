"""Convolution, the FIR filter and the delay line.

Sources: the FIR filter is the convolution of the input with its impulse response,
``y[n] = (W * h)[n]``, Eq. (6) of Hayes, Shier, Fazekas, McPherson and Saitis (2023), PDF
page 21, where it is the same operation as a convolutional layer; the delay line is the
wavetable "viewed as a delay line of length p" of Karplus and Strong (1983), page 3.
"""

from collections import deque
from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot


def convolve(x: Any, h: Any) -> np.ndarray:
    """Direct convolution ``y[n] = sum_m h[m] x[n - m]``, of length ``len(x) + len(h) - 1``.

    Args:
        x: input samples.
        h: impulse response, not empty.

    Raises:
        ValueError: if ``x`` or ``h`` is empty.

    Example:
        >>> convolve([1.0, 2.0, 3.0], [1.0, 1.0]).tolist()
        [1.0, 3.0, 5.0, 3.0]
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.size == 0 or h.size == 0:
        raise ValueError("x and h must not be empty")
    y = np.zeros(x.size + h.size - 1)
    for m, hm in enumerate(h):
        y[m : m + x.size] += hm * x
    return y


class FIR(StatefulRoot):
    """Finite impulse response filter, ``y[n] = sum_{m=0}^{M-1} h[m] x[n - m]`` (Eq. 6).

    Before ``M`` samples have arrived the missing past inputs are 0. The result of the
    last sample is in ``out``.

    Args:
        h: the ``M`` taps (the impulse response), not empty.

    Raises:
        ValueError: if ``h`` is empty.

    Example:
        >>> f = FIR([0.5, 0.5])
        >>> f.update_many([1.0, 1.0, 3.0]).tolist()
        [0.5, 1.0, 2.0]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_x", "out")

    def __init__(self, h: Any) -> None:
        h = np.asarray(h, dtype=float)
        if h.size == 0:
            raise ValueError("h must not be empty")
        self.h = h
        self._x: deque = deque([0.0] * h.size, maxlen=h.size)
        self.out = 0.0

    def update(self, x: float) -> "FIR":
        """Filter one sample; the result is ``out``."""
        self._x.appendleft(float(x))
        self.out = float(
            np.dot(self.h, np.fromiter(self._x, dtype=float, count=self.h.size))
        )
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Filter a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out


class DelayLine(StatefulRoot):
    """A delay of ``d`` samples: ``y[n] = x[n - d]``, 0 before the first ``d`` samples arrive.

    Args:
        d: delay in samples, at least 1.

    Raises:
        ValueError: if ``d < 1``.

    Example:
        >>> DelayLine(2).update_many([1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 0.0, 1.0, 2.0]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_b", "out")

    def __init__(self, d: int = 1) -> None:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self._b: deque = deque([0.0] * self.d, maxlen=self.d)
        self.out = 0.0

    def update(self, x: float) -> "DelayLine":
        """Push one sample; the delayed sample is ``out``."""
        self.out = self._b[0]
        self._b.append(float(x))
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Push a block; returns the delayed samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out
