"""Frame features: the magnitude spectrum of a block in a few bands."""

import numpy as np

from dense_armor.roles import Root
from dense_armor.utility.audio.signal import Audio
from dense_armor.utility.audio.windows import get_window


class AudioFeatures(Root):
    """The magnitude spectrum of a block, averaged in ``n_bands`` groups of bins.

    The magnitude is ``|rfft(w x)|`` of the windowed block, the short-time Fourier transform
    of Allen and Rabiner (1977). Choice stated: the bins are split in ``n_bands`` groups of
    nearly equal size with ``numpy.array_split`` and each group is replaced by the mean
    magnitude; no source defines this grouping. The keys are ``b0 .. b{n_bands-1}``, fixed, so
    the dict goes straight into ``learn_one`` of a Dense-Armor estimator.

    Args:
        n_bands: number of groups, at least 1.
        window: a window name.

    Raises:
        ValueError: from ``transform``, if a block has fewer frequency bins
            (``len(block) // 2 + 1``) than ``n_bands``.

    Example:
        >>> f = AudioFeatures(n_bands=2)
        >>> sorted(f.transform(Audio(np.ones(16), 16)))
        ['b0', 'b1']
    """

    budget_s: float = 1e-3
    memory_class: str = "O(1)"

    def __init__(self, n_bands: int = 8, window: str = "hann") -> None:
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        self.n_bands = int(n_bands)
        self.window = window

    def transform(self, block: Audio) -> dict[str, float]:
        """Features of one mono block."""
        x: np.ndarray = np.asarray(block.to_mono().x, dtype=float)
        mag = np.abs(np.fft.rfft(x * get_window(self.window, len(x))))
        if self.n_bands > len(mag):
            raise ValueError(
                f"n_bands is {self.n_bands} but the block has only {len(mag)} frequency bins"
            )
        parts = np.array_split(mag, self.n_bands)
        return {f"b{i}": float(p.mean()) for i, p in enumerate(parts)}
