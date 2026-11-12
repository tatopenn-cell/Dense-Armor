"""Resampling by an arbitrary, possibly changing, ratio with linear interpolation.

Source: A. Falkenberg (2026), "Real-Time Arbitrary Resampling for Phase-Continuous
Microtuning in Audio Streams" (preprint by one author), Algorithm 3, PDF page 4:
``k = floor(u)``, ``alpha = u - k``, ``Y[n] = (1 - alpha) s0 + alpha s1`` with
``s0 = H[k]``, ``s1 = H[k + 1]``, then ``u <- u + q``. The interpolation of the source is
linear (Sec. 10.2, page 6), so the stopband is not that of a windowed FIR filter.
"""

from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot
from dense_armor.utility.audio.signal import Audio


class Resampler(StatefulRoot):
    """Demand-driven resampler; ``q`` is the input step per output sample.

    ``q = 2`` reads the input at twice the speed (half as many output samples, an octave up);
    ``q = Fin / Fout`` converts a sample rate. ``q`` can be changed at any time with
    :meth:`set_ratio`; the read position ``u`` stays continuous.

    Args:
        q: positive input step per output sample.

    Example:
        >>> r = Resampler(2.0)
        >>> r.update_many([0.0, 1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 2.0]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_buf", "_u")

    def __init__(self, q: float = 1.0) -> None:
        if q <= 0.0:
            raise ValueError(f"q must be positive, got {q}")
        self.q = float(q)
        self._buf = np.empty(0)
        self._u = 0.0

    def set_ratio(self, q: float) -> "Resampler":
        """Change the step ``q``; the position is kept."""
        if q <= 0.0:
            raise ValueError(f"q must be positive, got {q}")
        self.q = float(q)
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Push input samples; return the output samples that can be computed."""
        self._buf = np.concatenate([self._buf, np.asarray(values, dtype=float)])
        out = []
        while True:
            k = int(np.floor(self._u))
            if k + 1 >= len(self._buf):
                break
            a = self._u - k
            out.append((1.0 - a) * self._buf[k] + a * self._buf[k + 1])
            self._u += self.q
        k = int(np.floor(self._u))
        if k > 0:
            self._buf = self._buf[k:]
            self._u -= k
        return np.array(out)


def resample(audio: Audio, ratio: float) -> Audio:
    """Resample a mono ``Audio`` with a constant step ``ratio = Fin / Fout``.

    Args:
        audio: the input sound, one channel.
        ratio: the step ``q`` of the source.

    Returns:
        The output sound with sample rate ``audio.sr / ratio`` rounded.
    """
    if audio.channels != 1:
        raise ValueError("resample expects one channel")
    y = Resampler(ratio).update_many(audio.x)
    return Audio(y, int(round(audio.sr / ratio)), audio.t0)
