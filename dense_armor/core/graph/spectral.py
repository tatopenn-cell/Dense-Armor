"""Spectral decomposition of the graph Laplacian.

References:
    Liu, He, Laurent, Di Giovanni, Bronstein, Bresson (2024/2025),
    "A General Graph Spectral Wavelet Convolution via Chebyshev Order
    Decomposition", arXiv:2405.13806, Section 2: eigendecomposition
    ``L_hat = U Lambda U^T``. Shuman et al. (2013): the Fiedler vector
    (bibliographic only).
"""

from __future__ import annotations

import numpy as np


def eigendecomposition(
    L, *, dense_threshold: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors)`` of the symmetric ``L``.

    Eigenvalues ascending; ``eigenvectors[:, k]`` is the eigenvector of
    ``eigenvalues[k]``. The eigenvectors are orthonormal (up to the
    sign, which is fixed by ``fix_sign`` when needed).

    Args:
        L: symmetric matrix, dense or scipy sparse.

    Returns:
        ``(eigenvalues, eigenvectors)`` with shapes ``(n,)`` and
        ``(n, n)``.
    """
    L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    w, U = np.linalg.eigh(L_dense)
    idx = np.argsort(w)
    return w[idx], U[:, idx]


def fix_sign(U: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Flip each column so that the first non-negligible entry is positive."""
    U = U.copy()
    for k in range(U.shape[1]):
        col = U[:, k]
        nz = np.where(np.abs(col) > tol)[0]
        if len(nz) and col[nz[0]] < 0:
            U[:, k] = -col
    return U


def fiedler_vector(L) -> np.ndarray:
    """Second eigenvector of ``L``, unit norm, sign fixed.

    Args:
        L: graph Laplacian (symmetric).

    Returns:
        ``v`` with ``||v||_2 = 1`` and ``v^T L v = lambda_2``.
    """
    w, U = eigendecomposition(L)
    v = U[:, 1] if U.shape[1] > 1 else U[:, 0]
    v = v / np.linalg.norm(v)
    # fissa il segno
    nz = np.where(np.abs(v) > 1e-9)[0]
    if len(nz) and v[nz[0]] < 0:
        v = -v
    return v
