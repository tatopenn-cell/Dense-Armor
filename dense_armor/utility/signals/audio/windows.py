"""Window functions: rectangular, Hann, Hamming and Blackman.

All of them are sums of cosines, ``w[m] = sum_k (-1)^k a_k cos(2 pi k m / M)``, which is the
form of Harris (1978), Eq. (31a) for the Blackman family, with ``M = n`` for the periodic
(DFT-even) window and ``M = n - 1`` for the symmetric one.
"""

import numpy as np


def _cosine_sum(n: int, coefs: tuple[float, ...], periodic: bool) -> np.ndarray:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return np.ones(1)
    m = np.arange(n)
    period = n if periodic else n - 1
    w = np.zeros(n)
    for k, a in enumerate(coefs):
        w += (-1) ** k * a * np.cos(2.0 * np.pi * k * m / period)
    return w


def rectangular(n: int, periodic: bool = True) -> np.ndarray:
    """Rectangular window of ``n`` ones.

    Args:
        n: length, at least 1.
        periodic: accepted for a uniform signature; has no effect.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.ones(n)


def hann(n: int, periodic: bool = True) -> np.ndarray:
    """Hann window, ``0.5 - 0.5 cos(2 pi m / M)``.

    Harris (1978), Eq. (28a): the Hanning window is ``0.5 + 0.5 cos`` on samples centred at
    zero; shifted to start at ``m = 0`` the cosine changes sign.

    Args:
        n: length, at least 1.
        periodic: DFT-even window if True, symmetric if False.

    Example:
        >>> [round(float(v), 3) for v in hann(4)]
        [0.0, 0.5, 1.0, 0.5]
    """
    return _cosine_sum(n, (0.5, 0.5), periodic)


def hamming(n: int, periodic: bool = True, a: float = 0.54) -> np.ndarray:
    """Hamming window, ``a - (1 - a) cos(2 pi m / M)``.

    Harris (1978), Hamming section (PDF page 12): ``w = a + (1 - a) cos`` on centred samples,
    with ``a = 25/46``
    (about 0.543478) cancelling the first sidelobe exactly and ``a = 0.54`` its two-place
    approximation, which is the default.

    Args:
        n: length, at least 1.
        periodic: DFT-even window if True, symmetric if False.
        a: the constant term.
    """
    return _cosine_sum(n, (a, 1.0 - a), periodic)


def blackman(n: int, periodic: bool = True) -> np.ndarray:
    """Blackman window with the two-place coefficients 0.42, 0.50, 0.08.

    Harris (1978), Eq. (31b): the exact coefficients are 7938/18608, 9240/18608 and
    1430/18608; the two-place approximation is the Blackman window.

    Args:
        n: length, at least 1.
        periodic: DFT-even window if True, symmetric if False.
    """
    return _cosine_sum(n, (0.42, 0.50, 0.08), periodic)


WINDOWS = {
    "rectangular": rectangular,
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
}


def get_window(name: str, n: int, periodic: bool = True) -> np.ndarray:
    """Return the window ``name`` of length ``n``.

    Raises:
        ValueError: if ``name`` is not one of ``rectangular``, ``hann``, ``hamming``,
            ``blackman``.
    """
    if name not in WINDOWS:
        raise ValueError(f"unknown window {name!r}, expected one of {sorted(WINDOWS)}")
    return WINDOWS[name](n, periodic)
