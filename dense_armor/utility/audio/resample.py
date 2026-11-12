"""Resampling by an arbitrary, possibly changing, ratio with linear interpolation.

Source: A. Falkenberg (2026), "Real-Time Arbitrary Resampling for Phase-Continuous
Microtuning in Audio Streams" (preprint by one author), Algorithm 3, PDF page 4:
``k = floor(u)``, ``alpha = u - k``, ``Y[n] = G ((1 - alpha) s0 + alpha s1)`` with
``s0 = H[k]``, ``s1 = H[k + 1]`` and the gain ``G``, then ``u <- u + q``. The interpolation of
the source is linear (Sec. 10.2, page 6), so the stopband is not that of a windowed FIR filter.

The same preprint gives the rational form of the step (PDF page 3: ``A = round(B q)``,
``g = gcd(A, B)``, ``D = A / g``, ``U = B / g``) and Algorithm 4 (page 4), the continuous
transition of the pitch ratio, ratio-linear or cents-linear. They are :func:`rational_ratio`
and :func:`pitch_ratio_transition`.
"""

import math
from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot
from dense_armor.utility.audio.signal import Audio


class Resampler(StatefulRoot):
    """Demand-driven resampler; ``q`` is the input step per output sample.

    ``q = 2`` reads the input at twice the speed (half as many output samples, an octave up);
    ``q = Fin / Fout`` converts a sample rate. ``q`` can be changed at any time with
    :meth:`set_ratio`; the read position ``u`` stays continuous. The output is multiplied by
    the gain ``G`` of Algorithm 3, set with :meth:`set_gain`.

    Args:
        q: positive input step per output sample.
        gain: the gain ``G`` of the output, finite.

    Example:
        >>> r = Resampler(2.0)
        >>> r.update_many([0.0, 1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 2.0]
        >>> Resampler(2.0, gain=0.5).update_many([0.0, 1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 1.0]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = ("_buf", "_u")

    def __init__(self, q: float = 1.0, gain: float = 1.0) -> None:
        if q <= 0.0:
            raise ValueError(f"q must be positive, got {q}")
        if not math.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain}")
        self.q = float(q)
        self.gain = float(gain)
        self._buf = np.empty(0)
        self._u = 0.0

    def set_gain(self, gain: float) -> "Resampler":
        """Change the gain ``G`` of the output."""
        if not math.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain}")
        self.gain = float(gain)
        return self

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
            out.append(self.gain * ((1.0 - a) * self._buf[k] + a * self._buf[k + 1]))
            self._u += self.q
        k = int(np.floor(self._u))
        if k > 0:
            self._buf = self._buf[k:]
            self._u -= k
        return np.array(out)


def rational_ratio(q: float, b: int) -> tuple[int, int]:
    """The step ``q`` as a ratio of integers ``D / U``, with ``A = round(b q)`` (page 3).

    ``g = gcd(A, b)``, ``D = A / g`` and ``U = b / g``, so ``D / U`` is the approximation of
    ``q``; ``b`` is a representation parameter that the algorithm does not fix.

    Args:
        q: positive finite step.
        b: the denominator, at least 1.

    Raises:
        ValueError: if ``q`` is not positive and finite, ``b < 1``, or ``round(b q)`` is 0.

    Example:
        >>> rational_ratio(1.5, 4)
        (3, 2)
    """
    if not (q > 0.0 and math.isfinite(q)):
        raise ValueError(f"q must be positive and finite, got {q}")
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    a = round(b * q)
    if a < 1:
        raise ValueError(f"round(b q) = {a} is not positive; use a larger b")
    g = math.gcd(a, b)
    return a // g, b // g


def pitch_ratio_transition(
    p_start: float,
    p_target: float,
    fin: float,
    fout: float,
    steps: int,
    mode: str = "ratio-linear",
) -> np.ndarray:
    """The steps ``q[0 .. steps]`` of a continuous change of pitch ratio (Algorithm 4, page 4).

    With ``lam = j / steps``: in ``"ratio-linear"`` mode ``p = (1 - lam) p_start + lam p_target``;
    in ``"cents-linear"`` mode the cents ``c = 1200 log2(p)`` go linearly from the start to the
    target and ``p = 2^(c / 1200)``. Then ``q[j] = (fin / fout) p``. Give the trajectory to
    :meth:`Resampler.set_ratio`, one value per output sample or per block, so the read position
    is never reset.

    Args:
        p_start: starting pitch ratio, positive.
        p_target: final pitch ratio, positive.
        fin: input sample rate, positive.
        fout: output sample rate, positive.
        steps: number of transition steps ``L``, at least 1; the result has ``L + 1`` values.
        mode: ``"ratio-linear"`` or ``"cents-linear"``.

    Raises:
        ValueError: for a non-positive ratio or rate, ``steps < 1`` or an unknown ``mode``.

    Example:
        >>> pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, 2, "cents-linear").round(4).tolist()
        [1.0, 1.4142, 2.0]
    """
    if min(p_start, p_target, fin, fout) <= 0.0:
        raise ValueError("pitch ratios and sample rates must be positive")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if mode not in ("ratio-linear", "cents-linear"):
        raise ValueError(
            f"unknown mode {mode!r}, expected 'ratio-linear' or 'cents-linear'"
        )
    lam = np.arange(steps + 1) / steps
    if mode == "ratio-linear":
        p = (1.0 - lam) * p_start + lam * p_target
    else:
        c0, c1 = 1200.0 * math.log2(p_start), 1200.0 * math.log2(p_target)
        p = 2.0 ** (((1.0 - lam) * c0 + lam * c1) / 1200.0)
    return (fin / fout) * p


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
