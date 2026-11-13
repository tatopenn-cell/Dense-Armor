"""Graph Fourier Transform.

References:
    Liu et al. (2024/2025), arXiv:2405.13806, Section 2:
    ``f_hat = U^T f`` and ``f = U f_hat``. Shuman et al. (2013):
    the GFT definition (bibliographic only).
"""

from __future__ import annotations

import numpy as np


def gft(x, U: np.ndarray) -> np.ndarray:
    """Graph Fourier Transform: ``U^T @ x``.

    Args:
        x: signal of shape ``(n,)`` or ``(n, m)`` (columns = signals).
        U: eigenvectors of the Laplacian, shape ``(n, n)``.

    Returns:
        Transformed signal, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    return U.T @ x


def igft(x_hat, U: np.ndarray) -> np.ndarray:
    """Inverse GFT: ``U @ x_hat``."""
    x_hat = np.asarray(x_hat, dtype=float)
    return U @ x_hat
