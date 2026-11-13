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


class GraphFourier:
    """The Graph Fourier Transform of one graph, with the basis found once.

    Args:
        graph: a :class:`Graph`.
        normalized: use the normalised Laplacian (eigenvalues in
            ``[0, 2]``) instead of the combinatorial one.

    Attributes:
        eigenvalues: the frequencies, ascending, shape ``(n,)``.
        basis: the eigenvectors as columns, shape ``(n, n)``.

    Example:
        >>> from dense_armor.core.graph.core import Graph
        >>> f = GraphFourier(Graph(3, [(0, 1, 1.0), (1, 2, 1.0)]))
        >>> bool(np.allclose(f.inverse(f.transform([1.0, 2.0, 4.0])), [1.0, 2.0, 4.0]))
        True
    """

    def __init__(self, graph, normalized: bool = False) -> None:
        from dense_armor.core.graph.spectral import eigendecomposition

        L = graph.normalized_laplacian if normalized else graph.laplacian
        self.eigenvalues, self.basis = eigendecomposition(L)
        self.n_nodes = graph.n_nodes

    def transform(self, x) -> np.ndarray:
        """``U^T x``."""
        return gft(x, self.basis)

    def inverse(self, x_hat) -> np.ndarray:
        """``U x_hat``."""
        return igft(x_hat, self.basis)
