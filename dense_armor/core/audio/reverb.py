"""Artificial reverberation: the comb and all-pass filters of Schroeder and the reverberator."""

from typing import Any

import numpy as np

from dense_armor.core.audio._state import StatefulRoot

COMB_DELAYS_S = (0.0311, 0.0359, 0.0403, 0.0447)
ALLPASS_DELAYS_S = (0.005, 0.0017)
ALLPASS_GAIN = 0.7


def comb_gain(delay_s: float, t60_s: float) -> float:
    """Feedback gain of a comb of delay ``delay_s`` that gives the reverberation time ``t60_s``.

    From ``T = 3 tau / log10(1 / g)``: ``g = 10 ** (-3 tau / T)``.

    Args:
        delay_s: the delay ``tau`` in seconds, positive.
        t60_s: the reverberation time ``T`` in seconds, positive.

    Returns:
        The gain ``g`` in ``(0, 1)``.

    Raises:
        ValueError: if a time is not positive.

    Example:
        >>> round(comb_gain(0.1, 2.0), 4)
        0.7079

    References:
        Schroeder (1962), Natural Sounding Artificial Reverberation, J. Audio Eng. Soc. 10(3),
        219-223: Eq. 15, ``T = (3 / log10(1/g)) tau`` (60 dB of decay in the time ``T``). The
        text of the PDF was read; for ``g = 0.708`` the paper says that a delay of 100 ms gives
        ``T = 2 s``, which is the value of the example.
    """
    if delay_s <= 0.0 or t60_s <= 0.0:
        raise ValueError("delay_s and t60_s must be positive")
    return 10.0 ** (-3.0 * delay_s / t60_s)


class CombFilter(StatefulRoot):
    """A delay in a feedback loop: ``y[n] = x[n - D] + g y[n - D]``, with an optional damping.

    ``H(z) = z^-D / (1 - g z^-D)``: an echo train that decays by ``g`` every ``D`` samples. With
    ``damping = d`` a one-pole low-pass ``f[n] = (1 - d) y[n] + d f[n - 1]`` sits in the loop, so
    that the high frequencies decay faster (it has unit gain at 0 Hz). ``d = 0`` is the comb of the
    paper. The last output is in ``out``.

    Args:
        delay: ``D`` in samples, at least 1.
        gain: the feedback gain ``g``, in ``[0, 1)``.
        damping: ``d`` in ``[0, 1)``.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> CombFilter(2, 0.5).update_many([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).tolist()
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.0]

    References:
        Schroeder (1962), J. Audio Eng. Soc. 10(3), 219-223, Fig. 1 and Eq. 3: the impulse
        response is ``delta(t - tau) + g delta(t - 2 tau) + ...``. The text of the PDF was read.
        Verified against ``scipy.signal.lfilter`` (see ``test_the_comb_matches_scipy``). The
        damping is not in the paper: it is the usual low-pass in the loop and it is off by default.
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_buf", "_pos", "_lp", "out")

    def __init__(self, delay: int, gain: float = 0.7, damping: float = 0.0) -> None:
        if delay < 1 or not 0.0 <= gain < 1.0 or not 0.0 <= damping < 1.0:
            raise ValueError("delay must be >= 1, gain and damping in [0, 1)")
        self.delay = int(delay)
        self.gain = float(gain)
        self.damping = float(damping)
        self._buf = [0.0] * self.delay
        self._pos = 0
        self._lp = 0.0
        self.out = 0.0

    def update(self, x: float) -> "CombFilter":
        """Filter one sample; the result is ``out``."""
        y = self._buf[self._pos]
        self._lp = (1.0 - self.damping) * y + self.damping * self._lp
        self._buf[self._pos] = float(x) + self.gain * self._lp
        self._pos = (self._pos + 1) % self.delay
        self.out = y
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Filter a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out


class AllpassFilter(StatefulRoot):
    """The all-pass of Schroeder: ``H(z) = (z^-D - g) / (1 - g z^-D)``, gain 1 at every frequency.

    Implemented with one delay line: ``v[n] = x[n] + g v[n - D]``, ``y[n] = -g v[n] + v[n - D]``.
    Its impulse response is ``-g`` at 0, then ``(1 - g^2)`` times ``g^k`` at the multiples of ``D``.
    The last output is in ``out``.

    Args:
        delay: ``D`` in samples, at least 1.
        gain: ``g`` in ``[0, 1)``.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> AllpassFilter(2, 0.5).update_many([1.0, 0.0, 0.0, 0.0, 0.0]).tolist()
        [-0.5, 0.0, 0.75, 0.0, 0.375]

    References:
        Schroeder (1962), J. Audio Eng. Soc. 10(3), 219-223, Eq. 12: ``H(w) = (e^{-jwT} - g) /
        (1 - g e^{-jwT})``; the impulse response ``-g delta(t) + (1 - g^2) [delta(t - T) +
        g delta(t - 2T) + ...]`` follows from it (and is checked below). The text of the PDF was
        read. Verified against
        ``scipy.signal.lfilter`` and ``scipy.signal.freqz`` (see ``test_the_allpass_matches_scipy``
        and ``test_the_allpass_has_unit_gain_at_every_frequency``).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_buf", "_pos", "out")

    def __init__(self, delay: int, gain: float = ALLPASS_GAIN) -> None:
        if delay < 1 or not 0.0 <= gain < 1.0:
            raise ValueError("delay must be >= 1 and gain in [0, 1)")
        self.delay = int(delay)
        self.gain = float(gain)
        self._buf = [0.0] * self.delay
        self._pos = 0
        self.out = 0.0

    def update(self, x: float) -> "AllpassFilter":
        """Filter one sample; the result is ``out``."""
        delayed = self._buf[self._pos]
        v = float(x) + self.gain * delayed
        self._buf[self._pos] = v
        self._pos = (self._pos + 1) % self.delay
        self.out = -self.gain * v + delayed
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Filter a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out


class SchroederReverb(StatefulRoot):
    """The reverberator of Schroeder: four combs in parallel, two all-passes in series.

    The four combs have the delays of ``COMB_DELAYS_S`` (31.1 to 44.7 ms) and each one has the
    gain that gives the same reverberation time ``T`` (:func:`comb_gain`); their mean goes through
    the two all-passes of ``ALLPASS_DELAYS_S`` (5 and 1.7 ms, gain 0.7); the output is
    ``dry * x + wet * reverberated``. The last output is in ``out``.

    Choices stated, because the paper gives ranges and not values: the delays of the combs lie in
    the range of the paper (about 30 to 45 ms, a ratio of about 1:1.5) and are not simply related;
    the reverberation time is ``T = 0.2 + 2.8 room_size`` seconds; ``damping`` puts a low-pass in
    each comb (``damping = 0`` is the reverberator of the paper); the mean of the combs is
    divided by 4.

    Args:
        room_size: in ``[0, 1]``, sets ``T`` from 0.2 s to 3 s.
        damping: in ``[0, 1)``, the low-pass in the loop of the combs.
        wet: gain of the reverberated sound, at least 0.
        dry: gain of the direct sound, at least 0.
        sr: sampling rate in hertz, positive.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> r = SchroederReverb(0.5, 0.0, 1.0, 0.0, 8000.0)
        >>> y = r.update_many(np.r_[1.0, np.zeros(999)])
        >>> int(np.argmax(np.abs(y) > 0)) > 0
        True

    References:
        Schroeder (1962), Natural Sounding Artificial Reverberation, J. Audio Eng. Soc. 10(3),
        219-223 (Fig. 5): four comb filters in parallel with delays of about 30 to 45 ms (a range
        of about 1:1.5) whose gains give the same ``T`` (Eq. 15), then two all-pass filters of
        about 5 and 1.7 ms with gain 0.7, then a gain for the reverberated sound. The text of the
        PDF was read. The reverberation time is measured on the impulse response (see
        ``test_the_reverberation_time_is_the_one_asked``).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_combs", "_allpasses", "out")

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet: float = 0.3,
        dry: float = 0.7,
        sr: float = 44100.0,
    ) -> None:
        if not 0.0 <= room_size <= 1.0 or not 0.0 <= damping < 1.0:
            raise ValueError("room_size must be in [0, 1] and damping in [0, 1)")
        if wet < 0.0 or dry < 0.0 or sr <= 0.0:
            raise ValueError("wet and dry must be >= 0 and sr > 0")
        self.room_size = float(room_size)
        self.damping = float(damping)
        self.wet = float(wet)
        self.dry = float(dry)
        self.sr = float(sr)
        t60 = self.t60_s
        self._combs = [
            CombFilter(max(1, round(d * self.sr)), comb_gain(d, t60), self.damping)
            for d in COMB_DELAYS_S
        ]
        self._allpasses = [
            AllpassFilter(max(1, round(d * self.sr)), ALLPASS_GAIN)
            for d in ALLPASS_DELAYS_S
        ]
        self.out = 0.0

    @property
    def t60_s(self) -> float:
        """The reverberation time asked, in seconds."""
        return 0.2 + 2.8 * self.room_size

    def update(self, x: float) -> "SchroederReverb":
        """Reverberate one sample; the result is ``out``."""
        x = float(x)
        w = sum(c.update(x).out for c in self._combs) / len(self._combs)
        for a in self._allpasses:
            w = a.update(w).out
        self.out = self.dry * x + self.wet * w
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Reverberate a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out


def reverberation_time(impulse_response: Any, sr: float) -> float:
    """Time for the decay of an impulse response to fall by 60 dB, from the Schroeder integral.

    The energy decay curve is the reversed cumulative sum of the squares, in dB. A line is fitted
    to the part between -5 and -35 dB and extended to -60 dB.

    Args:
        impulse_response: the response, shape ``(n,)``, long enough to decay by 35 dB.
        sr: sampling rate in hertz.

    Returns:
        The time in seconds.

    Raises:
        ValueError: if the response does not fall 35 dB below its start.

    Example:
        >>> sr = 1000.0
        >>> h = np.exp(-np.arange(4000) / sr * 3.0 * np.log(10) / 1.0)
        >>> bool(abs(reverberation_time(h, sr) - 1.0) < 0.05)
        True

    References:
        Schroeder (1965), New method of measuring reverberation time, J. Acoust. Soc. Am. 37(3),
        409-412, DOI 10.1121/1.1909343 (reference checked on Crossref): the backward integration
        of the squared impulse response. Bibliographic only (not in ``papers/``); checked on an
        exponential whose reverberation time is known.
    """
    h = np.asarray(impulse_response, dtype=float)
    edc = np.cumsum((h**2)[::-1])[::-1]
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-300))
    if db.min() > -35.0:
        raise ValueError("the response does not decay by 35 dB")
    idx = np.nonzero((db <= -5.0) & (db >= -35.0))[0]
    slope, _ = np.polyfit(idx / sr, db[idx], 1)
    return -60.0 / slope
