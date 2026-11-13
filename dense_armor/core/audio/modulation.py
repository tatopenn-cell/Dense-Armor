"""Modulation effects: a chorus (a delay modulated by an LFO) and a phaser (all-passes swept by an LFO)."""

import math
from typing import Any

import numpy as np

from dense_armor.core.audio._state import StatefulRoot

PHASER_F_MIN = 200.0
PHASER_F_MAX = 2000.0


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


class Phaser(StatefulRoot):
    """First-order all-pass stages whose corner frequency is swept by an LFO, added to the original.

    Each stage is ``y[n] = a x[n] + x[n-1] - a y[n-1]``, with ``a = (t - 1) / (t + 1)`` and
    ``t = tan(pi f_c / sr)``: gain 1 at every frequency and a phase that goes from 0 at 0 Hz to
    -180 degrees at the Nyquist frequency, -90 degrees at ``f_c``. The stages are in series and the
    output is ``(1 - mix) x + mix y``. Where the series has turned the phase by an odd number of
    half turns the two terms cancel: a notch. ``stages`` stages give ``stages // 2`` notches below
    the Nyquist frequency, at ``f_k = (sr / pi) atan(t tan((2k + 1) pi / (2 stages)))`` (the stages
    of an equal corner frequency); an LFO that moves ``f_c`` moves them all.

    The corner frequency is ``f_c = 200 (2000 / 200) ** m`` hertz, with ``m = 0.5 + 0.5 depth
    sin(2 pi rate_hz t)``: it moves on a logarithmic scale between 200 and 2000 Hz, sits at 632 Hz
    (the geometric centre) when ``depth = 0``, and uses the fraction ``depth`` of the range around
    that centre. The last output is in ``out``.

    Choices stated: the sweep range 200 to 2000 Hz, the sine LFO and the mix of 0.5 by default,
    which gives the deepest notches. The task text says that ``stages`` stages give ``stages``
    notches: they give ``stages // 2``, a notch for every 360 degrees of turn after the first 180.

    Args:
        rate_hz: frequency of the LFO, at least 0.
        depth: fraction of the sweep range that is used, in ``[0, 1]``.
        stages: number of all-pass stages, an integer at least 1.
        sr: sampling rate in hertz, above 4000 (twice the top of the sweep).
        mix: weight of the all-passed signal, in ``[0, 1]``.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> p = Phaser(rate_hz=0.0, depth=0.0, stages=2, sr=8000.0, mix=1.0)
        >>> y = p.update_many(np.r_[1.0, np.zeros(2000)])
        >>> bool(abs((y ** 2).sum() - 1.0) < 1e-9)
        True

    References:
        Zolzer (ed.) (2011), DAFX: Digital Audio Effects, 2nd ed., Wiley, ISBN
        978-0-470-66599-2: the phaser as all-pass filters with a swept corner frequency, added to
        the input. Bibliographic only (not in ``papers/``). The frequencies of the notches were
        derived from the phase of the all-pass (bilinear transform of ``(s - w_c) / (s + w_c)``) and
        checked on the impulse response (see ``test_the_notches_are_where_the_phase_says``); the
        response at a fixed corner frequency is compared with ``scipy.signal.freqz`` (see
        ``test_the_static_response_matches_scipy``).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"
    _state_attrs = ("_x1", "_y1", "_n", "out")

    def __init__(
        self,
        rate_hz: float = 0.5,
        depth: float = 0.7,
        stages: int = 4,
        sr: float = 44100.0,
        mix: float = 0.5,
    ) -> None:
        if rate_hz < 0.0 or not 0.0 <= depth <= 1.0 or not 0.0 <= mix <= 1.0:
            raise ValueError("rate_hz >= 0, depth and mix in [0, 1]")
        if stages < 1 or int(stages) != stages:
            raise ValueError("stages must be an integer >= 1")
        if sr <= 2.0 * PHASER_F_MAX:
            raise ValueError("sr must be above 4000 Hz")
        self.rate_hz = float(rate_hz)
        self.depth = float(depth)
        self.stages = int(stages)
        self.sr = float(sr)
        self.mix = float(mix)
        self._x1 = [0.0] * self.stages
        self._y1 = [0.0] * self.stages
        self._n = 0
        self.out = 0.0

    def corner_frequency(self, n: int) -> float:
        """The corner frequency ``f_c`` in hertz at sample ``n``."""
        m = 0.5 + 0.5 * self.depth * math.sin(
            2.0 * math.pi * self.rate_hz * n / self.sr
        )
        return PHASER_F_MIN * (PHASER_F_MAX / PHASER_F_MIN) ** m

    def update(self, x: float) -> "Phaser":
        """Process one sample; the result is ``out``."""
        x = float(x)
        t = math.tan(math.pi * self.corner_frequency(self._n) / self.sr)
        a = (t - 1.0) / (t + 1.0)
        v = x
        for k in range(self.stages):
            y = a * v + self._x1[k] - a * self._y1[k]
            self._x1[k] = v
            self._y1[k] = y
            v = y
        self._n += 1
        self.out = (1.0 - self.mix) * x + self.mix * v
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Process a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out
