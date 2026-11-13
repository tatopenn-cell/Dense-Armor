"""Spectral features of a signal, one value per STFT frame."""

import numpy as np

from dense_armor.core.audio.fourier import stft


def _magnitude(x: np.ndarray, n_fft: int, hop: int, window: str) -> np.ndarray:
    return np.abs(stft(x, n_fft, hop, window))


def spectral_centroid(
    x: np.ndarray, sr: float, n_fft: int = 2048, hop: int = 512, window: str = "hann"
) -> np.ndarray:
    """Centre of mass of the magnitude spectrum of each frame, in hertz.

    ``sum(f * |X|) / sum(|X|)`` over the bins ``f = k sr / n_fft``. A silent frame gives 0.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz.
        n_fft: frame length.
        hop: frame step.
        window: a window name of :mod:`dense_armor.core.audio.windows`.

    Returns:
        Array of shape ``(n_frames,)``.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> c = spectral_centroid(np.sin(2 * np.pi * 1000 * t), 8000.0, 1024, 512)
        >>> bool(abs(c.mean() - 1000.0) < 10.0)
        True

    References:
        Peeters (2004), A large set of audio features for sound description (similarity and
        classification) in the CUIDADO project, IRCAM report. Bibliographic only (the PDF
        could not be downloaded); the values are verified against
        ``librosa.feature.spectral_centroid(center=False)``.
    """
    mag = _magnitude(x, n_fft, hop, window)
    freq = np.arange(mag.shape[1]) * sr / n_fft
    total = mag.sum(axis=1)
    return np.divide(mag @ freq, total, out=np.zeros_like(total), where=total > 0)


def spectral_rolloff(
    x: np.ndarray,
    sr: float,
    roll_percent: float = 0.85,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Frequency below which ``roll_percent`` of the magnitude of each frame lies, in hertz.

    The lowest bin ``k`` whose cumulative magnitude reaches ``roll_percent`` of the frame
    total, as a frequency ``k sr / n_fft``. A silent frame gives 0.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz.
        roll_percent: fraction of the total magnitude, in ``(0, 1)``.
        n_fft: frame length.
        hop: frame step.
        window: a window name of :mod:`dense_armor.core.audio.windows`.

    Returns:
        Array of shape ``(n_frames,)``.

    Raises:
        ValueError: if ``roll_percent`` is not in ``(0, 1)``.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> r = spectral_rolloff(np.sin(2 * np.pi * 1000 * t), 8000.0, 0.85, 1024, 512)
        >>> bool(abs(r.mean() - 1000.0) < 30.0)
        True

    References:
        Peeters (2004), as in :func:`spectral_centroid`. Bibliographic only; verified against
        ``librosa.feature.spectral_rolloff(center=False)``. The cumulative sum is of the
        magnitude, not of the power.
    """
    if not 0.0 < roll_percent < 1.0:
        raise ValueError("roll_percent must be in (0, 1)")
    mag = _magnitude(x, n_fft, hop, window)
    cum = np.cumsum(mag, axis=1)
    reached = cum >= roll_percent * cum[:, -1:] if mag.size else cum >= 0
    return reached.argmax(axis=1) * sr / n_fft


def spectral_flatness(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
    amin: float = 1e-10,
    power: float = 2.0,
) -> np.ndarray:
    """Geometric mean over arithmetic mean of the spectrum of each frame, in ``(0, 1]``.

    With ``p = max(amin, |X| ** power)``: ``exp(mean(log p)) / mean(p)``. It is close to 1 for
    noise and close to 0 for a tone.

    Args:
        x: real samples, shape ``(n,)``.
        n_fft: frame length.
        hop: frame step.
        window: a window name of :mod:`dense_armor.core.audio.windows`.
        amin: floor of ``p``, so that the logarithm is finite.
        power: exponent of the magnitude (2 is the power spectrum).

    Returns:
        Array of shape ``(n_frames,)``.

    Example:
        >>> t = np.arange(4096) / 8000.0
        >>> f = spectral_flatness(np.sin(2 * np.pi * 1000 * t), 1024, 512)
        >>> bool(f.max() < 0.05)
        True

    References:
        Peeters (2004), as in :func:`spectral_centroid`. Bibliographic only; verified against
        ``librosa.feature.spectral_flatness(center=False)``.
    """
    p = np.maximum(amin, _magnitude(x, n_fft, hop, window) ** power)
    return np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)


def spectral_flux(
    x: np.ndarray, n_fft: int = 2048, hop: int = 512, window: str = "hann"
) -> np.ndarray:
    """Positive change of the magnitude spectrum between a frame and the previous one.

    ``flux[m] = sum_k max(0, |X[m, k]| - |X[m-1, k]|)``, the half-wave rectified difference.
    The first frame has no previous one and gives 0.

    Args:
        x: real samples, shape ``(n,)``.
        n_fft: frame length.
        hop: frame step.
        window: a window name of :mod:`dense_armor.core.audio.windows`.

    Returns:
        Array of shape ``(n_frames,)``.

    Example:
        >>> x = np.concatenate([np.zeros(2048), np.ones(2048)])
        >>> f = spectral_flux(x, 512, 512)
        >>> bool(f[:3].max() == 0.0 and f.max() > 0.0)
        True

    References:
        Bello et al. (2005), A tutorial on onset detection in music signals, IEEE Trans.
        Speech and Audio Processing 13(5), 1035-1047, DOI 10.1109/TSA.2005.851998: the
        spectral flux as a detection function. Bibliographic only; no library has the same
        function, so it is verified against a hand computation (see
        ``test_flux_is_the_rectified_difference``).
    """
    mag = _magnitude(x, n_fft, hop, window)
    flux = np.zeros(len(mag))
    if len(mag) > 1:
        flux[1:] = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    return flux
