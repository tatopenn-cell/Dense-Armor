"""Time and pitch: a phase vocoder (time-stretch, pitch-shift), PSOLA and a harmonizer."""

import math
from typing import Any

import numpy as np

from dense_armor.core.audio.fourier import stft
from dense_armor.core.audio.resample import Resampler
from dense_armor.core.audio.windows import get_window
from dense_armor.roles import Root


def _overlap_add(frames: np.ndarray, hop: int, window: str, n_fft: int) -> np.ndarray:
    """Overlap-add of the inverse transforms, divided by the sum of the windows that overlap.

    The sum of the windows is not constant in the first and the last ``n_fft - hop`` samples, where
    fewer frames overlap; dividing by the sum itself (floored at 0.1% of its largest value, which
    only touches the first 30 samples or so) is exact there for a stationary sound.
    """
    w = get_window(window, n_fft)
    time = np.fft.irfft(frames, n=n_fft, axis=1)
    length = hop * (len(frames) - 1) + n_fft
    out = np.zeros(length)
    norm = np.zeros(length)
    for m in range(len(frames)):
        out[m * hop : m * hop + n_fft] += time[m]
        norm[m * hop : m * hop + n_fft] += w
    return out / np.maximum(norm, 0.001 * norm.max())


def time_stretch(
    x: Any,
    sr: float,
    rate: float,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Make ``x`` ``rate`` times longer without changing its pitch (a phase vocoder).

    ``rate > 1`` lengthens, ``rate < 1`` shortens; the output has ``round(len(x) * rate)``
    samples. (``librosa.effects.time_stretch`` takes the inverse: its ``rate`` speeds up.)

    The signal is cut into frames by :func:`~dense_armor.core.audio.fourier.stft` with step ``hop``,
    after ``n_fft`` zeros at the end so that the last samples are covered (nothing at the start: see
    below). Output frame ``j`` is built at the analysis
    position ``p = j / rate``, in frames, and kept at the last frame if ``p`` goes past it:

    * its magnitude is the interpolation, with ``p - floor(p)``, of the magnitudes of the
      analysis frames ``floor(p)`` and ``floor(p) + 1``;
    * its phase is accumulated: ``phase += w_k + wrap(dphi - w_k)``, where ``w_k = 2 pi hop k /
      n_fft`` is the phase that bin ``k`` turns in one hop, ``dphi`` is the difference of phase
      of the two analysis frames, and ``wrap`` takes it to ``(-pi, pi]``.

    The output frames are spaced ``hop`` apart like the input ones, so ``w_k + wrap(dphi - w_k)``
    is the phase that the true frequency of the partial in the bin turns in one hop
    (Flanagan and Golden's instantaneous frequency): a sine, even one that is not at the centre of
    a bin, goes on at its own frequency. The inverse transforms of the frames are added and divided
    by the sum of the windows that overlap, which is exact also in the first and the last
    ``n_fft - hop`` samples, where fewer frames overlap.

    Why nothing is added at the start: the phase is accumulated and never locked again, so an
    error in the phase step of one frame stays in every later frame. A signal padded at the start
    (by reflection: a kink at the join) has frames that cross the join, they give wrong phase steps,
    and the bins next to a partial lose their phase relation: the amplitude of the output falls (to
    0.37 of the input at ``rate = 0.5`` in a first version, with exact results at ``rate = 1`` and
    ``2``). At the end there is no later frame to harm, so zeros can be added there.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz; not used by the algorithm, it is there so that the functions of
            this module start with the same arguments.
        rate: the length ratio, positive.
        n_fft: frame length, a power of 2 is usual.
        hop: frame step, at most ``n_fft``; ``n_fft / hop`` of 4 is usual.
        window: a window name of :mod:`dense_armor.core.audio.windows`.

    Returns:
        Array of shape ``(round(n * rate),)``.

    Raises:
        ValueError: if ``rate`` is not positive, or ``x`` is not a non-empty 1D array.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> y = time_stretch(np.sin(2 * np.pi * 440 * t), 8000.0, 2.0)
        >>> y.shape
        (8192,)

    References:
        Flanagan and Golden (1966), Phase Vocoder, Bell System Technical Journal 45(9),
        1493-1509, DOI 10.1002/j.1538-7305.1966.tb01706.x, and Laroche and Dolson (1999), New
        phase-vocoder techniques for pitch-shifting, harmonizing and other exotic effects, IEEE
        WASPAA, 91-94, DOI 10.1109/ASPAA.1999.810857. Bibliographic only (not in ``papers/``). The
        phase rule was derived (above) and checked on a sine that is not at the centre of a bin
        (see ``test_a_sine_off_the_centre_of_a_bin_keeps_its_frequency``); the pitch of the output
        is measured with ``yin_f0`` and the result is compared with
        ``librosa.effects.time_stretch``.
    """
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or v.size == 0 or rate <= 0.0:
        raise ValueError("x must be a non-empty 1D array and rate positive")
    n_out = int(round(v.size * rate))
    spec = stft(np.concatenate([v, np.zeros(n_fft)]), n_fft, hop, window)
    n_frames, n_bins = spec.shape
    spec = np.vstack([spec, np.zeros((1, n_bins), dtype=complex)])
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    n_steps = int(math.ceil(n_out / hop)) + n_fft // hop
    positions = np.minimum(np.arange(n_steps) / rate, n_frames - 1.0)
    phase = np.angle(spec[0])
    frames = np.empty((n_steps, n_bins), dtype=complex)
    for j, p in enumerate(positions):
        k = int(p)
        frac = p - k
        a, b = spec[k], spec[k + 1]
        frames[j] = ((1.0 - frac) * np.abs(a) + frac * np.abs(b)) * np.exp(1j * phase)
        dphi = np.angle(b) - np.angle(a) - omega
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        phase = phase + omega + dphi
    y = _overlap_add(frames, hop, window, n_fft)[:n_out]
    return np.pad(y, (0, n_out - len(y)))


def pitch_shift(
    x: Any,
    sr: float,
    n_semitones: float,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Shift the pitch of ``x`` by ``n_semitones`` and keep its length.

    The method is the simplest one: :func:`time_stretch` by ``r = 2 ** (n_semitones / 12)`` and
    then a read of the result at ``r`` samples per output sample, by the existing
    :class:`~dense_armor.core.audio.resample.Resampler`. The pitch is multiplied by ``r`` and the
    length comes back to ``len(x)``.

    Choice stated: the resampler interpolates linearly and has no anti-alias filter (its
    docstring says so), so for a shift up the partials above ``sr / (2 r)`` fold back, and the
    result is best for sounds with little energy there.
    The resampler reads the next sample to interpolate, so the last sample of the output is 0.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz; not used by the algorithm.
        n_semitones: the shift in semitones; 12 is an octave up.
        n_fft: frame length of the phase vocoder.
        hop: frame step of the phase vocoder.
        window: a window name.

    Returns:
        Array of shape ``(n,)``.

    Raises:
        ValueError: as :func:`time_stretch`.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> pitch_shift(np.sin(2 * np.pi * 440 * t), 8000.0, 12.0).shape
        (4096,)

    References:
        Laroche and Dolson (1999), IEEE WASPAA, 91-94, DOI 10.1109/ASPAA.1999.810857: pitch
        shifting as time-scale modification followed by resampling. Bibliographic only (not in
        ``papers/``); verified by the frequency of the output (see
        ``test_plus_12_semitones_doubles_the_frequency``) and compared with
        ``librosa.effects.pitch_shift``.
    """
    ratio = 2.0 ** (n_semitones / 12.0)
    stretched = time_stretch(x, sr, ratio, n_fft, hop, window)
    y = Resampler(ratio).update_many(stretched)
    n = len(np.asarray(x))
    return np.pad(y[:n], (0, max(0, n - len(y))))


def _period_at(f0: Any, position: float, sr: float, fallback: float) -> float:
    if np.ndim(f0) == 0:
        value = float(f0)
    else:
        value = float(f0[min(max(int(position), 0), len(f0) - 1)])
    return sr / value if value > 0.0 and math.isfinite(value) else fallback


def psola_time_stretch(x: Any, sr: float, f0: Any, rate: float) -> np.ndarray:
    """Make ``x`` ``rate`` times longer by repeating or dropping whole periods (TD-PSOLA).

    The marks are one period apart: the first one at the largest sample of the first period, the
    next ones one period (``sr / f0``) after the previous, each moved to the largest sample within
    a quarter of a period. Each analysis segment is the samples within one period on each side of a
    mark, multiplied by a Hann window. The output has marks one period apart as well, so the
    pitch does not change; the output mark at time ``s`` takes the segment of the analysis mark
    nearest to ``s / rate``, and the segments are added. The output has ``round(len(x) * rate)``
    samples.

    A frame where ``f0`` is not positive or not finite (unvoiced) gets a period of 10 ms.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz, positive.
        f0: the fundamental frequency in hertz, a number or an array of ``n`` values (one for each
            sample); the ones of ``yin_f0`` frame by frame, repeated, will do.
        rate: the length ratio, positive.

    Returns:
        Array of shape ``(round(n * rate),)``.

    Raises:
        ValueError: if ``sr`` or ``rate`` is not positive, ``x`` is not a non-empty 1D array, or
            ``f0`` is an array that does not have one value per sample.

    Example:
        >>> t = np.arange(2000) / 8000.0
        >>> y = psola_time_stretch(np.sin(2 * np.pi * 200 * t), 8000.0, 200.0, 2.0)
        >>> y.shape
        (4000,)

    References:
        Moulines and Laroche (1995), Non-parametric techniques for pitch-scale and time-scale
        modification of speech, Speech Communication 16(2), 175-205, DOI
        10.1016/0167-6393(94)00054-E, and Moulines and Charpentier (1990), Speech Communication
        9(5-6), 453-467, DOI 10.1016/0167-6393(90)90021-Z (references checked on Crossref).
        Bibliographic only (not in ``papers/``); verified by the length, the pitch of the output
        measured with ``yin_f0``, and its periodicity (see ``test_psola_keeps_the_pitch``).
    """
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or v.size == 0 or sr <= 0.0 or rate <= 0.0:
        raise ValueError("x must be a non-empty 1D array, sr and rate positive")
    if np.ndim(f0) > 0 and len(f0) != len(v):
        raise ValueError("f0 must be a number or have one value for each sample")
    n = len(v)
    n_out = int(round(n * rate))
    fallback = 0.01 * sr
    t0 = _period_at(f0, 0, sr, fallback)
    first = int(np.argmax(np.abs(v[: max(1, int(round(t0)))])))
    marks = [float(first)]
    while marks[-1] < n:
        step = _period_at(f0, marks[-1], sr, fallback)
        nxt = marks[-1] + step
        lo, hi = int(nxt - step / 4), int(nxt + step / 4) + 1
        if lo >= n:
            marks.append(nxt)
            break
        seg = np.abs(v[max(lo, 0) : min(hi, n)])
        marks.append(float(max(lo, 0) + int(np.argmax(seg))) if len(seg) else nxt)
    marks_arr = np.array(marks)
    out = np.zeros(n_out + 4 * int(fallback) + 8)
    s = marks_arr[0] * rate
    while s < n_out + 1:
        idx = int(np.argmin(np.abs(marks_arr - s / rate)))
        m = marks_arr[idx]
        period = _period_at(f0, m, sr, fallback)
        half = max(1, int(round(period)))
        center = int(round(m))
        lo, hi = center - half, center + half + 1
        seg = np.zeros(2 * half + 1)
        src = v[max(lo, 0) : min(hi, n)]
        seg[max(0, -lo) : max(0, -lo) + len(src)] = src
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(2 * half + 1) / (2 * half)))
        at = int(round(s)) - half
        a, b = max(at, 0), min(at + 2 * half + 1, len(out))
        if b > a:
            out[a:b] += (seg * win)[a - at : b - at]
        s += period
    return out[:n_out]


class Harmonizer(Root):
    """The sound plus copies of it shifted by intervals, added with the same weight.

    ``process(x)`` returns the mean of ``pitch_shift(x, sr, i)`` over the intervals ``i`` in
    ``intervals_semitones``; an interval of 0 is the original sound. The mean (a division by the
    number of voices) keeps the level of the sum near that of the input.

    Args:
        intervals_semitones: the intervals in semitones, not empty, for example ``(-12, 0, 7)``.
        sr: sampling rate in hertz, positive.
        n_fft: frame length of the phase vocoder.
        hop: frame step of the phase vocoder.

    Raises:
        ValueError: if ``intervals_semitones`` is empty or ``sr`` is not positive.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> Harmonizer((0, 7), 8000.0).process(np.sin(2 * np.pi * 440 * t)).shape
        (4096,)

    References:
        Laroche and Dolson (1999), IEEE WASPAA, 91-94, DOI 10.1109/ASPAA.1999.810857:
        harmonizing by pitch-shifting and adding. Bibliographic only (not in ``papers/``);
        verified by the frequencies of the peaks of the spectrum (see
        ``test_the_harmonizer_gives_the_three_pitches``).
    """

    def __init__(
        self,
        intervals_semitones: Any,
        sr: float = 44100.0,
        n_fft: int = 2048,
        hop: int = 512,
    ) -> None:
        intervals = tuple(float(i) for i in intervals_semitones)
        if len(intervals) == 0 or sr <= 0.0:
            raise ValueError("intervals_semitones must not be empty and sr must be > 0")
        self.intervals_semitones = intervals
        self.sr = float(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)

    def process(self, x: Any) -> np.ndarray:
        """Return the mean of the shifted copies of ``x``; same length as ``x``."""
        v = np.asarray(x, dtype=float)
        voices = [
            v if i == 0.0 else pitch_shift(v, self.sr, i, self.n_fft, self.hop)
            for i in self.intervals_semitones
        ]
        return np.mean(voices, axis=0)
