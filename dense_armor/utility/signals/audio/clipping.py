"""Hard clipping.

Source: Esqueda, Valimaki and Bilbao (2016), "Rounding corners with BLAMP", DAFx-16, Eq. (14),
PDF page 5. The polyBLAMP-corrected version is ``hard_clip_blamp`` in ``blamp.py``.
"""

import numpy as np


def hard_clip(x: np.ndarray, level: float) -> np.ndarray:
    """``sgn(x) min(|x|, L)`` (Eq. 14).

    Args:
        x: samples.
        level: the threshold ``L`` in ``(0, 1]``.

    Raises:
        ValueError: if ``level`` is outside ``(0, 1]``.

    Example:
        >>> hard_clip(np.array([-2.0, 0.5, 2.0]), 1.0).tolist()
        [-1.0, 0.5, 1.0]
    """
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level}")
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.minimum(np.abs(x), level)
