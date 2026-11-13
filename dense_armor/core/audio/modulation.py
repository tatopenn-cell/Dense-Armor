"""Modulation effects: a chorus (a delay modulated by an LFO)."""

import math
from typing import Any

import numpy as np

from dense_armor.core.audio._state import StatefulRoot


class Chorus(StatefulRoot):
    """A copy of the sound delayed by a time that a sine LFO makes swing, mixed with the original.

    The delay is ``d(t) = delay_ms + depth_ms sin(2 pi rate_hz t)`` milliseconds and the output is
    ``(1 - mix) x[n] + mix x(n - d)``, with the delayed value read between two samples by linear
    interpolation. The delay never goes below 0 (``delay_ms >= depth_ms``). The state is the last
    samples and the position of the LFO, so a signal can be given in pieces. The last output is
    in ``out``.

    Choices stated: a sine LFO, linear interpolation and the centre delay ``delay_ms`` (15 ms by
    default) are the ones of this code; the task gives only the rate, the depth and the mix.

    Args:
        rate_hz: frequency of the LFO, at least 0.
        depth_ms: swing of the delay, at least 0.
        mix: weight of the delayed copy, in ``[0, 1]``.
        sr: sampling rate in hertz, positive.
        delay_ms: centre delay, at least ``depth_ms``.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> c = Chorus(rate_hz=0.0, depth_ms=0.0, mix=1.0, sr=1000.0, delay_ms=2.0)
        >>> c.update_many([1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 0.0, 1.0, 2.0]

    References:
        Zolzer (ed.) (2011), DAFX: Digital Audio Effects, 2nd ed., Wiley, ISBN
        978-0-470-66599-2: the chorus as a delay modulated by a low-frequency oscillator.
        Bibliographic only (not in ``papers/``). Checked against a loop written out by hand (see
        ``test_the_output_equals_a_hand_written_loop``), against an exact delay (see
        ``test_without_modulation_it_is_a_delay``) and on the rate of the LFO, measured on the
        envelope of the output (see ``test_the_rate_of_the_lfo_is_measured_on_the_output``).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_hist", "_n", "out")

    def __init__(
        self,
        rate_hz: float = 0.5,
        depth_ms: float = 5.0,
        mix: float = 0.5,
        sr: float = 44100.0,
        delay_ms: float = 15.0,
    ) -> None:
        if rate_hz < 0.0 or depth_ms < 0.0 or not 0.0 <= mix <= 1.0 or sr <= 0.0:
            raise ValueError("rate_hz and depth_ms must be >= 0, mix in [0, 1], sr > 0")
        if delay_ms < depth_ms:
            raise ValueError("delay_ms must be >= depth_ms")
        self.rate_hz = float(rate_hz)
        self.depth_ms = float(depth_ms)
        self.mix = float(mix)
        self.sr = float(sr)
        self.delay_ms = float(delay_ms)
        self._hist = np.zeros(self._history_length())
        self._n = 0
        self.out = 0.0

    def _history_length(self) -> int:
        return int(math.ceil((self.delay_ms + self.depth_ms) * 1e-3 * self.sr)) + 2

    def update_many(self, values: Any) -> np.ndarray:
        """Process a block; returns the output samples."""
        x = np.asarray(values, dtype=float)
        n = len(x)
        if n == 0:
            return np.zeros(0)
        h = len(self._hist)
        buf = np.concatenate([self._hist, x, [0.0]])
        t = (self._n + np.arange(n)) / self.sr
        d = (
            (self.delay_ms + self.depth_ms * np.sin(2.0 * np.pi * self.rate_hz * t))
            * 1e-3
            * self.sr
        )
        pos = h + np.arange(n) - d
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        delayed = (1.0 - frac) * buf[i0] + frac * buf[i0 + 1]
        out = (1.0 - self.mix) * x + self.mix * delayed
        self._hist = buf[n : n + h].copy()
        self._n += n
        self.out = float(out[-1])
        return out

    def update(self, x: float) -> "Chorus":
        """Process one sample; the result is ``out``."""
        self.update_many([x])
        return self
