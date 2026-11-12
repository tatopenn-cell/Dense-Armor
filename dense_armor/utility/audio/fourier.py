"""Short-time Fourier transform, its inverse by overlap-add, and a streaming form.

Source: Allen and Rabiner (1977), "A unified approach to short-time Fourier analysis and
synthesis", Proc. IEEE 65(11): the overlap-add (OLA) synthesis rebuilds the signal by adding
the inverse transforms of the frames, which is exact when the shifted analysis windows sum to a
constant (PDF pages 3 to 6).
"""

from typing import Any

import numpy as np

from dense_armor.utility.audio._state import StatefulRoot
from dense_armor.utility.audio.windows import get_window


def _win(window: str | np.ndarray, n_fft: int) -> np.ndarray:
    if isinstance(window, str):
        return get_window(window, n_fft)
    w = np.asarray(window, dtype=float)
    if w.shape != (n_fft,):
        raise ValueError(f"window must have shape ({n_fft},), got {w.shape}")
    return w


def cola_constant(window: np.ndarray, hop: int, tol: float = 1e-9) -> float | None:
    """Return the constant ``C`` with ``sum_m w[n - m hop] = C``, or None.

    The sum is evaluated over one hop of the interior, where every shift that touches
    the sample is present.

    Args:
        window: analysis window.
        hop: distance in samples between frames.
        tol: largest allowed spread of the sum.
    """
    n = len(window)
    if hop < 1:
        raise ValueError(f"hop must be >= 1, got {hop}")
    reps = -(-n // hop)
    pad = np.zeros((reps + 1) * hop)
    for r in range(reps):
        seg = window[r * hop : (r + 1) * hop]
        pad[: len(seg)] += seg
    s = pad[:hop]
    return (
        float(s[0])
        if float(s.max() - s.min()) <= tol * max(1.0, abs(float(s[0])))
        else None
    )


def cola_ok(window: np.ndarray, hop: int, tol: float = 1e-9) -> bool:
    """True when the shifted copies of ``window`` at ``hop`` sum to a constant."""
    return cola_constant(np.asarray(window, dtype=float), hop, tol) is not None


def stft(
    x: np.ndarray, n_fft: int, hop: int, window: str | np.ndarray = "hann"
) -> np.ndarray:
    """Short-time Fourier transform of a real signal, without padding.

    Frame ``m`` is ``x[m hop : m hop + n_fft]`` times the window. A signal shorter than
    ``n_fft`` gives zero frames.

    Args:
        x: real samples, shape ``(n,)``.
        n_fft: frame length.
        hop: frame step.
        window: a window name or an array of length ``n_fft``.

    Returns:
        Complex array of shape ``(frames, n_fft // 2 + 1)``.

    Example:
        >>> stft(np.ones(16), 8, 4, "rectangular").shape
        (3, 5)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if n_fft < 1 or hop < 1:
        raise ValueError("n_fft and hop must be >= 1")
    w = _win(window, n_fft)
    frames = 0 if len(x) < n_fft else 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(x[idx] * w, axis=1)


def istft(
    S: np.ndarray, hop: int, window: str | np.ndarray = "hann", n_fft: int | None = None
) -> np.ndarray:
    """Inverse of :func:`stft` by overlap-add.

    The result is exact (up to rounding) in the interior, where ``n_fft / hop`` frames
    overlap; the first and last ``n_fft - hop`` samples lack some overlaps and are
    scaled by the partial sum of the window.

    Args:
        S: spectra from :func:`stft`.
        hop: frame step used by :func:`stft`.
        window: the analysis window.
        n_fft: frame length; defaults to ``2 * (S.shape[1] - 1)``.

    Raises:
        ValueError: if the window and hop do not satisfy the overlap-add condition.
    """
    S = np.asarray(S)
    n = 2 * (S.shape[1] - 1) if n_fft is None else int(n_fft)
    w = _win(window, n)
    c = cola_constant(w, hop)
    if c is None or c == 0.0:
        raise ValueError("window and hop do not satisfy the overlap-add condition")
    frames = np.fft.irfft(S, n=n, axis=1)
    out = np.zeros(hop * (S.shape[0] - 1) + n)
    for m in range(S.shape[0]):
        out[m * hop : m * hop + n] += frames[m]
    return out / c


class StreamingSTFT(StatefulRoot):
    """The :func:`stft` one sample at a time.

    ``update(v)`` pushes a sample; after the first ``n_fft`` samples, and then every ``hop``
    samples, a new spectrum is ready and ``spectrum`` returns it. The spectra equal the
    rows of :func:`stft` on the same signal.

    Args:
        n_fft: frame length.
        hop: frame step.
        window: a window name.

    Example:
        >>> s = StreamingSTFT(4, 2, "rectangular")
        >>> _ = s.update_many([1.0, 2.0, 3.0, 4.0])
        >>> s.frames_
        1
    """

    budget_s: float = 1e-3
    memory_class: str = "O(window)"
    _state_attrs = ("_buf", "_S")

    def __init__(self, n_fft: int = 256, hop: int = 128, window: str = "hann") -> None:
        if n_fft < 1 or hop < 1:
            raise ValueError("n_fft and hop must be >= 1")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.window = window
        self._w = get_window(window, self.n_fft)
        self._buf = np.zeros(self.n_fft)
        self._S: np.ndarray | None = None
        self.n_seen_ = 0
        self.frames_ = 0

    @property
    def spectrum(self) -> np.ndarray | None:
        """The last spectrum, or None before the first frame."""
        return self._S

    def update(self, v: float) -> "StreamingSTFT":
        """Push one sample; returns self."""
        self._buf = np.roll(self._buf, -1)
        self._buf[-1] = float(v)
        self.n_seen_ += 1
        k = self.n_seen_ - self.n_fft
        if k >= 0 and k % self.hop == 0:
            self._S = np.fft.rfft(self._buf * self._w)
            self.frames_ += 1
        return self

    def update_many(self, values: Any) -> list[np.ndarray]:
        """Push several samples in order; return the spectra that became ready."""
        out = []
        for v in values:
            before = self.frames_
            self.update(v)
            if self.frames_ != before and self._S is not None:
                out.append(self._S.copy())
        return out
