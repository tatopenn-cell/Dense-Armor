"""Biquad filters: the coefficient formulas of the Audio EQ Cookbook and a Direct Form 1 filter.

Source: R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients"
(technical note, not a peer-reviewed paper). Every function below is the set of ``b0, b1, b2,
a0, a1, a2`` of the note, normalised by ``a0``; the filter is its Eq. (4), Direct Form 1.
"""

from typing import Any

import numpy as np

from dense_armor.utility.signals.audio._state import StatefulRoot


def _w0_alpha(f0: float, sr: float, q: float) -> tuple[float, float]:
    if not 0.0 < f0 < sr / 2.0:
        raise ValueError(f"f0 must be in (0, sr/2), got {f0} with sr {sr}")
    if q <= 0.0:
        raise ValueError(f"q must be positive, got {q}")
    w0 = 2.0 * np.pi * f0 / sr
    return w0, np.sin(w0) / (2.0 * q)


def _norm(b: list[float], a: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(b) / a[0], np.array(a) / a[0]


def lowpass(f0: float, sr: float, q: float = 0.7071067811865476):
    """LPF of the cookbook; returns ``(b, a)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    return _norm([(1 - c) / 2, 1 - c, (1 - c) / 2], [1 + al, -2 * c, 1 - al])


def highpass(f0: float, sr: float, q: float = 0.7071067811865476):
    """HPF of the cookbook; returns ``(b, a)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    return _norm([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al])


def bandpass(f0: float, sr: float, q: float = 1.0, constant_peak: bool = True):
    """BPF of the cookbook; returns ``(b, a)``.

    Args:
        f0: centre frequency.
        sr: sample rate.
        q: quality factor.
        constant_peak: if True the peak gain is 0 dB (``b0 = alpha``); if False the
            skirt gain is ``q`` (``b0 = sin(w0) / 2``).
    """
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    b0 = al if constant_peak else np.sin(w0) / 2.0
    return _norm([b0, 0.0, -b0], [1 + al, -2 * c, 1 - al])


def notch(f0: float, sr: float, q: float = 1.0):
    """Notch filter of the cookbook; returns ``(b, a)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    return _norm([1.0, -2 * c, 1.0], [1 + al, -2 * c, 1 - al])


def peaking(f0: float, sr: float, gain_db: float, q: float = 1.0):
    """peakingEQ of the cookbook; returns ``(b, a)``. ``A = 10 ** (gain_db / 40)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    A = 10.0 ** (gain_db / 40.0)
    return _norm([1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A])


def lowshelf(f0: float, sr: float, gain_db: float, q: float = 0.7071067811865476):
    """lowShelf of the cookbook with the ``Q`` form of ``alpha``; returns ``(b, a)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    A = 10.0 ** (gain_db / 40.0)
    s = 2.0 * np.sqrt(A) * al
    return _norm(
        [
            A * ((A + 1) - (A - 1) * c + s),
            2 * A * ((A - 1) - (A + 1) * c),
            A * ((A + 1) - (A - 1) * c - s),
        ],
        [
            (A + 1) + (A - 1) * c + s,
            -2 * ((A - 1) + (A + 1) * c),
            (A + 1) + (A - 1) * c - s,
        ],
    )


def highshelf(f0: float, sr: float, gain_db: float, q: float = 0.7071067811865476):
    """highShelf of the cookbook with the ``Q`` form of ``alpha``; returns ``(b, a)``."""
    w0, al = _w0_alpha(f0, sr, q)
    c = np.cos(w0)
    A = 10.0 ** (gain_db / 40.0)
    s = 2.0 * np.sqrt(A) * al
    return _norm(
        [
            A * ((A + 1) + (A - 1) * c + s),
            -2 * A * ((A - 1) + (A + 1) * c),
            A * ((A + 1) + (A - 1) * c - s),
        ],
        [
            (A + 1) - (A - 1) * c + s,
            2 * ((A - 1) - (A + 1) * c),
            (A + 1) - (A - 1) * c - s,
        ],
    )


class Biquad(StatefulRoot):
    """Direct Form 1 biquad, ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]``.

    Eq. (4) of the cookbook, with the coefficients divided by ``a0``.

    Args:
        b: three numerator coefficients.
        a: three denominator coefficients, ``a[0]`` not zero.

    Raises:
        ValueError: if a coefficient list has not three entries, ``a[0]`` is zero, or a
            pole lies on or outside the unit circle.

    Example:
        >>> b, a = lowpass(1000.0, 8000.0)
        >>> f = Biquad(b, a)
        >>> bool(abs(f.update(1.0).out - b[0]) < 1e-12)
        True
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"
    _state_attrs = ("_x1", "_x2", "_y1", "_y2", "out")

    def __init__(self, b: Any, a: Any) -> None:
        b = np.asarray(b, dtype=float)
        a = np.asarray(a, dtype=float)
        if b.shape != (3,) or a.shape != (3,):
            raise ValueError("b and a must have three coefficients each")
        if a[0] == 0.0:
            raise ValueError("a[0] must not be zero")
        self.b = b / a[0]
        self.a = a / a[0]
        if np.any(np.abs(np.roots(self.a)) >= 1.0):
            raise ValueError("the filter has a pole on or outside the unit circle")
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0
        self.out = 0.0

    def update(self, x: float) -> "Biquad":
        """Filter one sample; the result is ``out``."""
        x = float(x)
        y = (
            self.b[0] * x
            + self.b[1] * self._x1
            + self.b[2] * self._x2
            - self.a[1] * self._y1
            - self.a[2] * self._y2
        )
        self._x2, self._x1 = self._x1, x
        self._y2, self._y1 = self._y1, y
        self.out = float(y)
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Filter a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out
