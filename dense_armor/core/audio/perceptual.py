"""Perceptual analysis: the mel scale, the mel filterbank, the mel spectrogram and the MFCC."""

import numpy as np

from dense_armor.core.audio.fourier import stft

_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = np.log(6.4) / 27.0


def hz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    """Convert hertz to mel on the scale of Slaney's Auditory Toolbox.

    The scale is linear below 1 kHz (``200/3`` Hz per mel) and logarithmic above, with the
    step ``log(6.4) / 27``: 27 steps take 1 kHz to 6.4 kHz. The ratio between two successive
    centres of the log region is then ``6.4 ** (1/27) = 1.0711703``.

    Args:
        freq: frequency in hertz, a number or an array.

    Returns:
        The same shape in mel.

    Example:
        >>> round(float(hz_to_mel(1000.0)), 9)
        15.0

    References:
        Slaney (1998), Auditory Toolbox, Version 2, Interval Research Corporation
        Technical Report #1998-010, page 29: 13 linearly spaced filters followed by 27
        log-spaced filters "separated by a factor of 1.0711703 in frequency". The factor
        is checked in the tests (it is ``6.4 ** (1/27)``). The 1 kHz break and the linear
        step are the values of the toolbox as used by ``librosa.hz_to_mel``; the TR text
        does not state them as a formula, so they are verified against that function.
    """
    f = np.asarray(freq, dtype=float)
    lin = f / _F_SP
    log = _MIN_LOG_MEL + np.log(np.maximum(f, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOG_STEP
    out = np.where(f >= _MIN_LOG_HZ, log, lin)
    return float(out) if out.ndim == 0 else out


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel (Slaney scale) to hertz, the inverse of :func:`hz_to_mel`.

    Args:
        mel: value in mel, a number or an array.

    Returns:
        The same shape in hertz.

    Example:
        >>> round(float(mel_to_hz(hz_to_mel(440.0))), 9)
        440.0

    References:
        Same as :func:`hz_to_mel`.
    """
    m = np.asarray(mel, dtype=float)
    lin = m * _F_SP
    log = _MIN_LOG_HZ * np.exp(_LOG_STEP * (np.maximum(m, _MIN_LOG_MEL) - _MIN_LOG_MEL))
    out = np.where(m >= _MIN_LOG_MEL, log, lin)
    return float(out) if out.ndim == 0 else out


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: float,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular mel filters on the bins of an FFT, each with unit area (Slaney).

    The ``n_mels + 2`` edges are equally spaced on the Slaney mel scale between ``fmin`` and
    ``fmax``. Filter ``i`` rises from edge ``i`` to edge ``i + 1`` and falls to edge ``i + 2``,
    measured in hertz on the bins ``0 .. sr/2``. Each filter is divided by half of its width,
    ``(edge[i+2] - edge[i]) / 2``, so that every filter has the same area.

    Choice stated: this is the Slaney construction (triangles with equal area), not the HTK one
    (``2595 log10(1 + f/700)``, triangles with unit peak).

    Args:
        n_mels: number of filters, at least 1.
        n_fft: FFT length; the matrix has ``n_fft // 2 + 1`` columns.
        sr: sampling rate in hertz.
        fmin: lowest edge in hertz.
        fmax: highest edge in hertz; ``None`` means ``sr / 2``.

    Returns:
        Array of shape ``(n_mels, n_fft // 2 + 1)``.

    Raises:
        ValueError: if ``n_mels`` or ``n_fft`` is below 1, or if ``fmax`` is not above ``fmin``.

    Example:
        >>> mel_filterbank(8, 64, 8000.0).shape
        (8, 33)

    References:
        Slaney (1998), Auditory Toolbox, Version 2, Interval Research Corporation
        Technical Report #1998-010, page 29 (the mel filterbank of the MFCC). Bibliographic
        only for the area normalisation and the edge placement: the matrix is verified
        against ``librosa.filters.mel`` (see ``test_mel_filterbank_matches_librosa``).
    """
    if n_mels < 1 or n_fft < 1:
        raise ValueError("n_mels and n_fft must be >= 1")
    fmax = sr / 2.0 if fmax is None else float(fmax)
    if fmax <= fmin:
        raise ValueError("fmax must be above fmin")
    mels = np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    edges = np.asarray(mel_to_hz(mels), dtype=float)
    bins = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fdiff = np.diff(edges)
    ramps = edges[:, None] - bins[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    return weights * (2.0 / (edges[2:] - edges[:-2]))[:, None]


def mel_spectrogram(
    x: np.ndarray,
    sr: float,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    window: str = "hann",
) -> np.ndarray:
    """Power spectrogram of ``x`` projected on the mel filterbank.

    It is ``|STFT|**2`` of :func:`dense_armor.core.audio.fourier.stft` times the filterbank of
    :func:`mel_filterbank`, with no logarithm and no other scaling. The STFT has no padding:
    frame ``m`` is ``x[m hop : m hop + n_fft]`` times the window, so a signal shorter than
    ``n_fft`` gives zero frames. ``librosa.feature.melspectrogram`` centres the frames by
    padding ``n_fft // 2`` samples on each side, so the two agree with ``center=False``.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz.
        n_mels: number of mel filters.
        n_fft: frame length.
        hop: frame step.
        fmin: lowest edge of the filterbank in hertz.
        fmax: highest edge in hertz; ``None`` means ``sr / 2``.
        window: a window name of :mod:`dense_armor.core.audio.windows`.

    Returns:
        Array of shape ``(n_frames, n_mels)``.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> mel_spectrogram(np.sin(2 * np.pi * 440 * t), 8000.0, n_mels=20).shape
        (5, 20)

    References:
        Allen and Rabiner (1977), A unified approach to short-time Fourier analysis and
        synthesis, Proc. IEEE 65(11), for the STFT (see :func:`stft`). Purwins et al. (2019),
        Deep Learning for Audio Signal Processing, IEEE JSTSP, page 3, says that the mel
        filter bank is made of triangular filters and that the log-mel spectrum is a common
        feature; it gives no formula. The projection is verified against
        ``librosa.feature.melspectrogram(center=False)`` (see
        ``test_mel_spectrogram_matches_librosa``).
    """
    power = np.abs(stft(x, n_fft, hop, window)) ** 2
    return power @ mel_filterbank(n_mels, n_fft, sr, fmin, fmax).T


def _dct_ii_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape ``(n_out, n_in)`` (the first ``n_out`` rows)."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    mat = np.sqrt(2.0 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    mat[0] = np.sqrt(1.0 / n_in)
    return mat


def mfcc(
    x: np.ndarray,
    sr: float,
    n_mfcc: int = 13,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop: int = 512,
    top_db: float = 80.0,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients: the DCT-II of the log mel spectrogram.

    Steps: :func:`mel_spectrogram`, then the power in decibels
    ``10 log10(max(power, 1e-10))`` floored at ``top_db`` below its largest value (all frames
    together), then the orthonormal DCT-II along the mel axis, keeping the first ``n_mfcc``
    coefficients.

    Choices stated: the logarithm is in decibels with the floor of ``librosa`` (not
    ``log10`` of the energies as in the original paper), so that the result can be compared
    with ``librosa.feature.mfcc``; the DCT is written as a matrix so that the library does not
    need ``scipy``.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz.
        n_mfcc: number of coefficients, between 1 and ``n_mels``.
        n_mels: number of mel filters.
        n_fft: frame length.
        hop: frame step.
        top_db: the floor, in decibels below the largest value; ``None`` for no floor.

    Returns:
        Array of shape ``(n_frames, n_mfcc)``.

    Raises:
        ValueError: if ``n_mfcc`` is not between 1 and ``n_mels``.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> mfcc(np.sin(2 * np.pi * 440 * t), 8000.0, n_mfcc=5).shape
        (5, 5)

    References:
        Davis and Mermelstein (1980), Comparison of parametric representations for
        monosyllabic word recognition in continuously spoken sentences, IEEE Trans. ASSP
        28(4), 357-366, DOI 10.1109/TASSP.1980.1163420: the cosine transform of the log
        energies of a mel filterbank. Bibliographic only (the paper is not in ``papers/``);
        the formula is verified against ``librosa.feature.mfcc`` (see
        ``test_mfcc_matches_librosa``) and against ``scipy.fft.dct`` (see
        ``test_the_dct_matrix_is_the_orthonormal_dct_ii``).
    """
    if not 1 <= n_mfcc <= n_mels:
        raise ValueError("n_mfcc must be between 1 and n_mels")
    mel = mel_spectrogram(x, sr, n_mels, n_fft, hop)
    if mel.size == 0:
        return np.zeros((0, n_mfcc))
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    if top_db is not None:
        log_mel = np.maximum(log_mel, log_mel.max() - top_db)
    return log_mel @ _dct_ii_matrix(n_mfcc, n_mels).T
