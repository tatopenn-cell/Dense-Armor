"""The value type for sound: samples, sample rate and start time."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    """A block of sound.

    Args:
        x: samples, shape ``(n,)`` for one channel or ``(n, channels)``.
        sr: sample rate in Hz, positive.
        t0: time of the first sample in seconds.

    Raises:
        ValueError: if ``sr`` is not positive or ``x`` is not 1D or 2D.

    Example:
        >>> a = Audio(np.zeros(8000), 8000)
        >>> a.duration, a.channels
        (1.0, 1)
    """

    x: np.ndarray
    sr: int
    t0: float = 0.0

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        if self.sr <= 0:
            raise ValueError(f"sr must be positive, got {self.sr}")
        if self.x.ndim not in (1, 2):
            raise ValueError(f"x must be 1D or 2D, got {self.x.ndim}D")

    @property
    def n(self) -> int:
        """Number of samples per channel."""
        return int(self.x.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels."""
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n / self.sr

    def channel(self, k: int) -> "Audio":
        """Return channel ``k`` as a one-channel ``Audio`` (a view, not a copy).

        Raises:
            IndexError: if ``k`` is not a channel of this sound.

        Example:
            >>> a = Audio(np.arange(6.0).reshape(3, 2), 8000)
            >>> a.channel(1).x.tolist()
            [1.0, 3.0, 5.0]
        """
        if not 0 <= k < self.channels:
            raise IndexError(f"channel {k} is not in 0..{self.channels - 1}")
        return self if self.x.ndim == 1 else Audio(self.x[:, k], self.sr, self.t0)

    def to_mono(self) -> "Audio":
        """Return the mean of the channels as a one-channel ``Audio``."""
        if self.x.ndim == 1:
            return self
        return Audio(self.x.mean(axis=1), self.sr, self.t0)
