"""Oscillators, white noise and the Karplus-Strong plucked string.

Sources:
    * additive sawtooth and square: Hayes, Shier, Fazekas, McPherson, Saitis (2023),
      "A review of differentiable digital signal processing for music and speech
      synthesis", Eq. (10) and (11), PDF page 23, after Masuda and Saito (2021);
    * PolyBLEP residual: Eq. (1) of Du et al. (2026), "DDSP-QbE++", PDF page 2, with the sign
      of the second term of Eq. (2) corrected as the docstring of :class:`Oscillator` states;
      the two-point polynomial BLEP is introduced in Valimaki and Pekonen (2012), JASA 132(2);
    * Karplus-Strong: Karplus and Strong (1983), Computer Music Journal 7(2), pages 3 and 4
      of the PDF.
"""

from collections import deque
from typing import Any

import numpy as np

from dense_armor.roles import Root
from dense_armor.utility.signals.audio.signal import Audio

KINDS = ("sine", "sawtooth_additive", "square_additive", "sawtooth_polyblep")


def _blep_residual(t: float) -> float:
    if -1.0 <= t < 0.0:
        return t * t + 2.0 * t + 1.0
    if 0.0 <= t < 1.0:
        return -t * t + 2.0 * t - 1.0
    return 0.0


class Oscillator(Root):
    """A phase-accumulator oscillator whose frequency may change at every sample.

    The phase is the running sum of ``f / sr``, so it stays continuous when ``f`` changes.

    * ``"sine"``: ``sin(2 pi phi)``.
    * ``"sawtooth_additive"``: ``sum_{k=1..K} 2 / (pi k) sin(2 pi k phi)`` (Eq. 10), a
      falling ramp, with ``K`` the harmonics below the Nyquist frequency.
    * ``"square_additive"``: ``sum_{k=1..K} 4 / (pi (2k-1)) sin(2 pi (2k-1) phi)`` (Eq. 11).
    * ``"sawtooth_polyblep"``: the rising ramp ``2 phi - 1`` minus the residual of Eq. (1)
      of the source at the wrap, taken at ``phi / d`` and at ``(phi - 1) / d`` with
      ``d = f / sr``. Eq. (2) of the source adds the second term; with that sign the
      aliasing is not reduced (measured in the tests), so both terms are subtracted.

    Args:
        kind: one of ``KINDS``.
        sr: sample rate in Hz.

    Raises:
        ValueError: for an unknown ``kind`` or a non-positive ``sr``.

    Example:
        >>> o = Oscillator("sine", 8000)
        >>> o.update(1000.0).out == 0.0
        True
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"

    def __init__(self, kind: str = "sine", sr: int = 44100) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        self.kind = kind
        self.sr = int(sr)
        self.phase_ = 0.0
        self.out = 0.0

    def update(self, f: float) -> "Oscillator":
        """Output one sample at frequency ``f`` Hz, then advance the phase."""
        f = abs(float(f))
        p = self.phase_
        if self.kind == "sine":
            y = np.sin(2.0 * np.pi * p)
        elif self.kind in ("sawtooth_additive", "square_additive"):
            kmax = int(self.sr / 2.0 / f) if f > 0 else 0
            k = np.arange(1, kmax + 1)
            k = k[k * f < self.sr / 2.0]
            if self.kind == "sawtooth_additive":
                y = float(np.sum(2.0 / (np.pi * k) * np.sin(2.0 * np.pi * k * p)))
            else:
                k = k[k % 2 == 1]
                y = float(np.sum(4.0 / (np.pi * k) * np.sin(2.0 * np.pi * k * p)))
        else:
            d = f / self.sr
            y = 2.0 * p - 1.0
            if d > 0.0:
                y -= _blep_residual(p / d) + _blep_residual((p - 1.0) / d)
        self.out = float(y)
        self.phase_ = (p + f / self.sr) % 1.0
        return self

    def update_many(self, freqs: Any) -> np.ndarray:
        """Output one sample per entry of ``freqs``; returns the samples."""
        f = np.broadcast_to(np.asarray(freqs, dtype=float), np.shape(freqs))
        out = np.empty(len(f))
        for i, v in enumerate(f):
            out[i] = self.update(v).out
        return out


def white_noise(n: int, seed: int = 0) -> np.ndarray:
    """``n`` samples of white Gaussian noise with mean 0 and variance 1.

    Args:
        n: number of samples.
        seed: seed of the generator.
    """
    return np.random.default_rng(seed).standard_normal(n)


class KarplusStrong(Root):
    """The plucked-string algorithm of Karplus and Strong.

    A delay line of ``p`` samples is loaded with a burst of values ``+1`` or ``-1`` chosen at
    random (page 4), then every output is the mean of two successive older outputs,
    ``y[t] = (y[t-p] + y[t-p-1]) / 2`` (page 3). The tone has a period of ``p + 1/2``
    samples. Choice stated: ``y[-1]`` is 0, which the paper does not specify.

    Args:
        p: length of the delay line, at least 2.
        seed: seed of the burst.

    Example:
        >>> ks = KarplusStrong(4, seed=1)
        >>> y = ks.update_many(np.zeros(8))
        >>> len(y)
        8
    """

    budget_s: float = 2e-4
    memory_class: str = "O(p)"

    def __init__(self, p: int = 100, seed: int = 0) -> None:
        if p < 2:
            raise ValueError(f"p must be >= 2, got {p}")
        self.p = int(p)
        self.seed = seed
        self._burst = np.where(
            np.random.default_rng(seed).random(self.p) < 0.5, -1.0, 1.0
        )
        self._hist: deque = deque([0.0] * (self.p + 1), maxlen=self.p + 1)
        self._i = 0
        self.out = 0.0

    def update(self, _v: float = 0.0) -> "KarplusStrong":
        """Output the next sample of the string; the argument is ignored."""
        if self._i < self.p:
            y = float(self._burst[self._i])
        else:
            y = 0.5 * (self._hist[1] + self._hist[0])
        self._hist.append(y)
        self._i += 1
        self.out = y
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Output ``len(values)`` samples."""
        out = np.empty(len(values))
        for i in range(len(values)):
            out[i] = self.update().out
        return out


def plucked_string(freq: float, sr: int, duration: float, seed: int = 0) -> Audio:
    """A plucked string at ``freq`` Hz.

    The period of the tone is ``p + 1/2`` samples, so ``p = round(sr / freq - 1/2)``;
    this rounding is a choice stated here, the paper only says that ``p`` is an integer.

    Args:
        freq: fundamental in Hz.
        sr: sample rate.
        duration: length in seconds.
        seed: seed of the burst.
    """
    p = max(2, int(round(sr / freq - 0.5)))
    n = int(round(duration * sr))
    return Audio(KarplusStrong(p, seed).update_many(np.zeros(n)), sr)
