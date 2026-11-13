"""Perceptual analysis: the mel scale, the mel filterbank, the mel spectrogram and the MFCC."""

import numpy as np

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
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fdiff = np.diff(edges)
    ramps = edges[:, None] - bins[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    return weights * (2.0 / (edges[2:] - edges[:-2]))[:, None]
