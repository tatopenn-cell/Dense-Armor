"""The analytic signal and an envelope follower with attack and release."""

import math

import numpy as np


def hilbert(x: np.ndarray) -> np.ndarray:
    """Analytic signal of a real signal: ``x + j * H(x)`` with ``H`` the Hilbert transform.

    Computed with the FFT: the negative frequencies are set to zero, the positive ones are
    doubled, and the DC term (and the Nyquist term of an even length) are kept. The real part is
    ``x`` and the imaginary part is the Hilbert transform. The FFT treats the signal as periodic.

    Args:
        x: real samples, shape ``(n,)``, ``n >= 1``.

    Returns:
        Complex array of shape ``(n,)``.

    Raises:
        ValueError: if ``x`` is not a non-empty real 1D array.

    Example:
        >>> t = np.arange(64) / 64.0
        >>> z = hilbert(np.cos(2 * np.pi * 4 * t))
        >>> bool(np.allclose(z.imag, np.sin(2 * np.pi * 4 * t)))
        True

    References:
        Oppenheim and Schafer, Discrete-Time Signal Processing (the analytic signal and the
        discrete Hilbert transform). Bibliographic only (not in ``papers/``): verified against
        ``scipy.signal.hilbert`` (see ``test_hilbert_matches_scipy``) and on a cosine, whose
        Hilbert transform is a sine (see ``test_the_hilbert_transform_of_a_cosine_is_a_sine``).
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0 or np.iscomplexobj(x):
        raise ValueError("x must be a non-empty real 1D array")
    n = x.size
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x.astype(float)) * h)


def envelope_follower(
    x: np.ndarray, sr: float, attack_s: float = 0.01, release_s: float = 0.1
) -> np.ndarray:
    """Envelope of ``x``: ``|hilbert(x)|`` smoothed by a one-pole filter with attack and release.

    ``y[n] = e[n] + a * (y[n-1] - e[n])`` with ``e = |hilbert(x)|`` and ``a = exp(-1 / (tau sr))``,
    where ``tau`` is ``attack_s`` while the envelope rises (``e[n] > y[n-1]``) and ``release_s``
    while it falls. After a step of ``e`` the output has covered ``1 - 1/e`` (63%) of the step
    after ``tau`` seconds. A time of 0 follows ``e`` at once. The state starts at 0. The loop is
    in Python, one pass over the samples.

    Args:
        x: real samples, shape ``(n,)``.
        sr: sampling rate in hertz.
        attack_s: time constant of the rise, in seconds, at least 0.
        release_s: time constant of the fall, in seconds, at least 0.

    Returns:
        Array of shape ``(n,)``.

    Raises:
        ValueError: if ``sr`` is not positive or a time is negative.

    Example:
        >>> t = np.arange(8000) / 8000.0
        >>> y = envelope_follower(np.sin(2 * np.pi * 440 * t), 8000.0)
        >>> bool(abs(y[-1] - 1.0) < 0.05)
        True

    References:
        The rise and fall of a one-pole smoother with separate time constants is the usual
        envelope detector of dynamics processors; see Giannoulis, Massberg and Reiss (2012),
        Digital Dynamic Range Compressor Design - A Tutorial and Analysis, J. Audio Eng. Soc.
        60(6), 399-408. Bibliographic only (not in ``papers/``): verified on a burst, where the
        output reaches 63% of the step after ``attack_s`` and falls to 37% after ``release_s``
        (see ``test_a_burst_rises_and_falls_with_the_time_constants``).
    """
    if sr <= 0 or attack_s < 0 or release_s < 0:
        raise ValueError("sr must be positive and the times must be >= 0")
    env = np.abs(hilbert(x))
    a_att = math.exp(-1.0 / (attack_s * sr)) if attack_s > 0 else 0.0
    a_rel = math.exp(-1.0 / (release_s * sr)) if release_s > 0 else 0.0
    out = np.empty_like(env)
    y = 0.0
    for i, e in enumerate(env.tolist()):
        a = a_att if e > y else a_rel
        y = e + a * (y - e)
        out[i] = y
    return out
