"""Sliding discrete Fourier transform, from Jacobsen and Lyons (2003)."""

from collections import deque
from typing import Any

import numpy as np

from dense_armor.roles import Root


class SDFT(Root):
    """Sliding DFT: the N-point DFT of the last N samples, O(N) per update.

    A batch DFT recomputes the sum over N samples every time a new sample
    arrives, so N samples cost ``O(N^2)`` in total. The sliding DFT
    updates the previous spectrum in place, in ``O(N)`` per sample, with
    the recursion::

        X_new[k] = (X_old[k] - x_old + x_new) * exp(+j * 2 * pi * k / N)

    where `x_old` is the sample leaving the window, `x_new` the one
    entering, and `k = 0, ..., N-1`. The sign of the exponential is the
    one that makes the value match ``numpy.fft.fft`` on the same window
    (the paper writes the same sign; the case ``[1, 2, 3, 4]`` moving to
    ``[2, 3, 4, 5]`` gives the same values as ``numpy.fft.fft`` in the
    test of this module).

    The first call after ``__init__`` computes the DFT of the buffer
    with ``numpy.fft.fft`` (the recursion needs a starting value).

    Args:
        n: window length, strictly positive.

    Raises:
        ValueError: if ``n < 1``.

    Examples:
        >>> from dense_armor.utility.misc.sdft import SDFT
        >>> s = SDFT(n=4)
        >>> for v in [1.0, 2.0, 3.0, 4.0]:
        ...     _ = s.update(v)
        >>> s.is_ready
        True
        >>> float(round(s.X[0].real, 6))
        10.0

    References:
        Jacobsen, E., Lyons, R. (2003). The Sliding DFT. IEEE Signal
        Processing Magazine 20(2), 74-80. A course tutorial on the
        sliding DFT is the PDF in the task folder; the recursion above
        is the one in the tutorial, Equation (1), page 4 of that copy
        (Section "SDFT as", MATLAB code).
    """

    budget_s: float = 8e-3
    memory_class: str = "O(N)"

    def __init__(self, n: int = 8) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self._buffer: deque = deque(maxlen=self.n)
        self._X: np.ndarray | None = None
        self.n_seen_: int = 0

    @property
    def is_ready(self) -> bool:
        """True once the buffer holds ``n`` samples."""
        return len(self._buffer) == self.n

    @property
    def X(self) -> np.ndarray:
        """The last computed spectrum, shape ``(n,)``."""
        if self._X is None:
            return np.zeros(self.n, dtype=complex)
        return self._X

    def update(self, v: float) -> "SDFT":
        """Push one sample and update the spectrum.

        Before the buffer is full, the spectrum is the DFT of the
        samples seen so far (zero-padded to length ``n``). Once it is
        full, the update uses the sliding recursion above.

        Args:
            v: the new sample.

        Returns:
            self.
        """
        fv = float(v)
        if not np.isfinite(fv):
            return self
        was_full = self.is_ready
        x_old = self._buffer[0] if was_full else 0.0
        self._buffer.append(fv)
        self.n_seen_ += 1
        if not was_full:
            self._X = np.fft.fft(np.asarray(self._buffer, dtype=float), n=self.n)
            return self
        k = np.arange(self.n)
        rot = np.exp(1j * 2.0 * np.pi * k / self.n)
        assert self._X is not None
        self._X = (self._X - x_old + fv) * rot
        return self

    def update_many(self, values: Any) -> "SDFT":
        """Push several samples in order."""
        for v in values:
            _ = self.update(v)
        return self

    def spectrum(self) -> np.ndarray:
        """Return a copy of the current spectrum."""
        return self.X.copy()
