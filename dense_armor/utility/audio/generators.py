"""Oscillators, white noise and the Karplus-Strong plucked string.

Sources:
    * additive sawtooth and square: Hayes, Shier, Fazekas, McPherson, Saitis (2023),
      "A review of differentiable digital signal processing for music and speech
      synthesis", Eq. (10) and (11), PDF page 23, after Masuda and Saito (2021);
    * PolyBLEP residual: Eq. (1) of Du et al. (2026), "DDSP-QbE++", PDF page 2. Its Eq. (2)
      prints a ``+`` before the second residual term; the docstring of :class:`Oscillator`
      says why both terms are subtracted;
    * pulse and square waves by polyBLEP: Valimaki and Pekonen (2012), JASA 132(2), PDF page 4,
      Eq. (12) and (13): the fractional delay of a discontinuity is the value of the modulo
      counter after it, divided by the step; for the downward edge of a pulse of duty cycle
      ``P`` the counter is taken after it exceeds ``P``, with ``P`` subtracted; every second
      correction is inverted;
    * Karplus-Strong: Karplus and Strong (1983), Computer Music Journal 7(2), pages 3 and 4
      of the PDF;
    * pink noise: Timmer and Koenig (1995), "On generating power law noise", A&A 300, 707,
      Sec. 3 (PDF page 3: two Gaussian numbers per Fourier frequency, amplitude
      ``(1/w)^(beta/2)``, a real Nyquist term, conjugate negative frequencies, inverse transform);
    * ADSR: Hayes et al. (2023), Eq. (15), PDF page 27, after Masuda and Saito (2023);
    * triangle with polyBLAMP: Esqueda, Valimaki and Bilbao (2016), see ``blamp.py``.
"""

from collections import deque
from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot
from dense_armor.utility.audio.blamp import _r as _blamp_r
from dense_armor.utility.audio.signal import Audio

KINDS = (
    "sine",
    "sawtooth_additive",
    "square_additive",
    "sawtooth_polyblep",
    "square_polyblep",
    "pulse_polyblep",
    "triangle_blamp",
)


def _blep_residual(t: float) -> float:
    if -1.0 <= t < 0.0:
        return t * t + 2.0 * t + 1.0
    if 0.0 <= t < 1.0:
        return -t * t + 2.0 * t - 1.0
    return 0.0


class Oscillator(StatefulRoot):
    """A phase-accumulator oscillator whose frequency may change at every sample.

    The phase is the running sum of ``f / sr``, so it stays continuous when ``f`` changes. The
    phase of sample ``n`` is the sum of the frequencies before ``n`` (the first sample is at
    phase 0); Eq. (10) and (11) of the source sum up to and including ``n``, which is the same
    wave one sample earlier.

    * ``"sine"``: ``sin(2 pi phi)``.
    * ``"sawtooth_additive"``: ``sum_{k=1..K} 2 / (pi k) sin(2 pi k phi)`` (Eq. 10), a
      falling ramp, with ``K`` the harmonics below the Nyquist frequency.
    * ``"square_additive"``: ``sum_{k=1..K} 4 / (pi (2k-1)) sin(2 pi (2k-1) phi)`` (Eq. 11).
    * ``"sawtooth_polyblep"``: the rising ramp ``2 phi - 1`` minus the residual of Eq. (1)
      of the source at the wrap, taken at ``phi / d`` and at ``(phi - 1) / d`` with
      ``d = f / sr``. Eq. (2) of the source adds the second term; with that sign the
      aliasing is not reduced (measured in the tests), so both terms are subtracted.
    * ``"square_polyblep"`` and ``"pulse_polyblep"``: ``+1`` for ``phi < duty`` and ``-1``
      after it (``duty = 1/2`` for the square). The residual of Eq. (1) is added at the rising
      edge (``phi = 0``) and subtracted at the falling edge (``phi = duty``), the second one
      taken at ``(phi - duty) mod 1`` as in Valimaki and Pekonen (2012), Eq. (13) for a pulse
      wave; the polarity of the two corrections is opposite, as the paper says.
    * ``"triangle_blamp"``: the triangle of :func:`triangle_blamp`, from ``-1`` at
      ``phi = 0`` up to ``+1`` at ``phi = 1/2``, one sample at a time. A corner changes the
      two samples before it and the one after, so ``out`` is the sample **two calls
      earlier** (``0.0`` for the first two calls) and ``f`` must be below ``sr / 4``.

    Args:
        kind: one of ``KINDS``.
        sr: sample rate in Hz.
        duty: fraction of the period at ``+1`` for ``"pulse_polyblep"``, in ``(0, 1)``;
            ignored by the other kinds.

    Raises:
        ValueError: for an unknown ``kind``, a non-positive ``sr`` or a ``duty`` outside
            ``(0, 1)``.

    Example:
        >>> o = Oscillator("sine", 8000)
        >>> o.update(1000.0).out == 0.0
        True
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"
    _state_attrs = ("_w", "_last_step", "out")

    def __init__(self, kind: str = "sine", sr: int = 44100, duty: float = 0.5) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        if not 0.0 < duty < 1.0:
            raise ValueError(f"duty must be in (0, 1), got {duty}")
        self.kind = kind
        self.sr = int(sr)
        self.duty = float(duty)
        self.phase_ = 0.0
        self.n_ = 0
        self.out = 0.0
        self._w = [0.0, 0.0, 0.0]
        self._last_step = 0.0

    def _additive(self, f: float, p: float) -> float:
        kmax = int(self.sr / 2.0 / f) if f > 0 else 0
        k = np.arange(1, kmax + 1)
        k = k[k * f < self.sr / 2.0]
        if self.kind == "sawtooth_additive":
            return float(np.sum(2.0 / (np.pi * k) * np.sin(2.0 * np.pi * k * p)))
        k = k[k % 2 == 1]
        return float(np.sum(4.0 / (np.pi * k) * np.sin(2.0 * np.pi * k * p)))

    def _pulse(self, p: float, step: float) -> float:
        duty = 0.5 if self.kind == "square_polyblep" else self.duty
        y = 1.0 if p < duty else -1.0
        if step > 0.0:
            q = (p - duty) % 1.0
            y += _blep_residual(p / step) + _blep_residual((p - 1.0) / step)
            y -= _blep_residual(q / step) + _blep_residual((q - 1.0) / step)
        return y

    def _triangle_blamp(self, p: float, step: float) -> float:
        if step >= 0.25:
            raise ValueError("triangle_blamp needs f < sr / 4")
        if self.n_ == 0:
            self._last_step = step
        w = [self._w[0], self._w[1], self._w[2] + 1.0 - 4.0 * abs(p - 0.5), 0.0]
        ls = self._last_step
        if ls > 0.0:
            pp = p - ls
            sign = 0.0
            if pp < 0.0:
                sign, delta = 1.0, p / ls
            elif pp < 0.5 <= p:
                sign, delta = -1.0, (p - 0.5) / ls
            if sign:
                for k in (-2, -1, 0, 1):
                    if self.n_ + k >= 0:
                        w[k + 2] += sign * 8.0 * ls * _blamp_r(k + delta)
        self._w = w[1:]
        self._last_step = step
        return w[0]

    def update(self, f: float) -> "Oscillator":
        """Output one sample at frequency ``f`` Hz, then advance the phase."""
        f = abs(float(f))
        p = self.phase_
        step = f / self.sr
        if self.kind == "sine":
            y = np.sin(2.0 * np.pi * p)
        elif self.kind in ("sawtooth_additive", "square_additive"):
            y = self._additive(f, p)
        elif self.kind == "sawtooth_polyblep":
            y = 2.0 * p - 1.0
            if step > 0.0:
                y -= _blep_residual(p / step) + _blep_residual((p - 1.0) / step)
        elif self.kind == "triangle_blamp":
            y = self._triangle_blamp(p, step)
        else:
            y = self._pulse(p, step)
        self.out = float(y)
        self.phase_ = (p + step) % 1.0
        self.n_ += 1
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


class KarplusStrong(StatefulRoot):
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
    memory_class: str = "O(window)"
    _state_attrs = ("_hist", "_i", "out")

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


def pink_noise(n: int, seed: int = 0, beta: float = 1.0) -> np.ndarray:
    """Noise with a power spectrum ``S(w) ~ (1 / w)^beta`` (``beta = 1`` is pink noise).

    The algorithm of Timmer and Koenig (1995), Sec. 3: for each Fourier frequency draw two
    Gaussian numbers, multiply them by ``(1 / w)^(beta / 2)`` and use them as the real and
    imaginary part of the Fourier transform; at the Nyquist frequency of an even ``n`` only
    one number is drawn, the negative frequencies are the complex conjugates, and the signal
    is the inverse transform. Choices stated: the component at ``w = 0`` is set to 0 (a power
    law is not defined there) and the result is scaled to unit variance; the paper fixes
    neither.

    Args:
        n: number of samples, at least 2.
        seed: seed of the generator.
        beta: exponent of the spectrum, ``beta = 2`` is a random walk.

    Raises:
        ValueError: if ``n < 2``.

    Example:
        >>> pink_noise(8, 0).shape
        (8,)
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    rng = np.random.default_rng(seed)
    m = n // 2 + 1
    amp = np.zeros(m)
    amp[1:] = np.arange(1, m) ** (-beta / 2.0)
    f = rng.standard_normal(m) * amp + 1j * rng.standard_normal(m) * amp
    if n % 2 == 0:
        f[-1] = f[-1].real
    x = np.fft.irfft(f, n)
    return x / x.std()


class ADSR(StatefulRoot):
    """The attack-decay-sustain-release envelope of Masuda and Saito (2023).

    ``u(t) = |t v_max / t_atk|[0, v_max] + |(t - t_atk)(v_sus - v_max) / t_dec|[v_sus - v_max, 0]
    + |-(t - t_off) v_sus / t_rel|[-v_sus, 0]`` with ``|x|[a, b] = min(max(x, a), b)``
    (Eq. 15). The value is 0 at ``t = 0``, ``v_max`` at ``t_atk``, ``v_sus`` from
    ``t_atk + t_dec`` to ``t_off``, and 0 at ``t_off + t_rel``.

    Args:
        t_atk: attack time in seconds, positive.
        t_dec: decay time, positive.
        v_sus: sustain level, between 0 and ``v_max``.
        t_rel: release time, positive.
        t_off: note-off time, not before ``t_atk + t_dec``.
        sr: sample rate.
        v_max: peak level.

    Raises:
        ValueError: if a time is not positive, ``v_sus`` is outside ``[0, v_max]`` or
            ``t_off < t_atk + t_dec``.

    Example:
        >>> e = ADSR(0.1, 0.1, 0.5, 0.2, 0.5, 100)
        >>> [round(e.update().out, 3) for _ in range(12)][-3:]
        [0.9, 1.0, 0.95]
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"
    _state_attrs = ("out",)

    def __init__(
        self,
        t_atk: float = 0.01,
        t_dec: float = 0.1,
        v_sus: float = 0.7,
        t_rel: float = 0.2,
        t_off: float = 1.0,
        sr: int = 44100,
        v_max: float = 1.0,
    ) -> None:
        if min(t_atk, t_dec, t_rel) <= 0 or sr <= 0:
            raise ValueError("t_atk, t_dec, t_rel and sr must be positive")
        if not 0.0 <= v_sus <= v_max:
            raise ValueError("v_sus must be in [0, v_max]")
        if t_off < t_atk + t_dec:
            raise ValueError("t_off must not precede t_atk + t_dec")
        self.t_atk = t_atk
        self.t_dec = t_dec
        self.v_sus = v_sus
        self.t_rel = t_rel
        self.t_off = t_off
        self.sr = int(sr)
        self.v_max = v_max
        self.n_ = 0
        self.out = 0.0

    def value(self, t: float) -> float:
        """The envelope at time ``t`` seconds."""
        vm, vs = self.v_max, self.v_sus
        a = min(max(t * vm / self.t_atk, 0.0), vm)
        d = min(max((t - self.t_atk) * (vs - vm) / self.t_dec, vs - vm), 0.0)
        r = min(max(-(t - self.t_off) * vs / self.t_rel, -vs), 0.0)
        return float(a + d + r)

    def update(self, _v: float = 0.0) -> "ADSR":
        """Output the next sample of the envelope; the argument is ignored."""
        self.out = self.value(self.n_ / self.sr)
        self.n_ += 1
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Output ``len(values)`` samples."""
        out = np.empty(len(values))
        for i in range(len(values)):
            out[i] = self.update().out
        return out
