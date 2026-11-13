"""Spectral filters on a graph (heat kernel, low-pass) and the
Chebyshev polynomial approximation.

References:
    Liu et al. (2024/2025), arXiv:2405.13806, Section 2 for the
    spectral filter ``g(L) = U g(Lambda) U^T``, and Section 3.1 for
    the Chebyshev recurrence ``T_k(y) = 2 y T_{k-1}(y) - T_{k-2}(y)``
    with ``T_0 = 1``, ``T_1 = y``. Shuman et al. (2013) for the heat
    kernel (bibliographic only). Hammond, Vandergheynst, Gribonval
    (2011) for the graph Chebyshev filter (bibliographic only).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _as_dense(L) -> np.ndarray:
    return L.toarray() if hasattr(L, "toarray") else np.asarray(L, dtype=float)


def heat_kernel(L, tau: float) -> np.ndarray:
    """Heat kernel ``exp(-tau L)`` as a dense ``(n, n)`` matrix."""
    L_d = _as_dense(L)
    return expm(-float(tau) * L_d)


def low_pass(L, cutoff: float) -> np.ndarray:
    """Low-pass spectral filter: ``U g(Lambda) U^T`` with
    ``g(lambda) = 1`` for ``lambda <= cutoff``, ``0`` otherwise.

    Choices stated explicitly:
        The sharp cutoff (an indicator function) is a design choice of this
        module. The papers cited in the module describe the general
        spectral filter ``g(L) = U g(Lambda) U^T`` but do not prescribe
        a low-pass shape; a smooth cutoff (heat kernel or Chebyshev)
        would be a different choice.
    """
    L_d = _as_dense(L)
    w, U = np.linalg.eigh(L_d)
    g = (w <= float(cutoff)).astype(float)
    return (U * g) @ U.T


def apply_filter(x, L, filter_fn) -> np.ndarray:
    """Apply the spectral filter ``filter_fn`` (a function of the
    eigenvalues) to ``x`` via ``U g(Lambda) U^T x``."""
    L_d = _as_dense(L)
    w, U = np.linalg.eigh(L_d)
    g = np.asarray(filter_fn(w), dtype=float)
    return U @ (g * (U.T @ np.asarray(x, dtype=float)))


def chebyshev_coefficients(g, lmax: float, order: int) -> np.ndarray:
    """Chebyshev coefficients of ``g`` on ``[0, lmax]``.

    Returns ``c_0 .. c_order`` in the expansion
    ``g(lambda) ~ sum_k c_k T_k(2 lambda / lmax - 1)``.
    """
    if not (lmax > 0 and np.isfinite(lmax)):
        raise ValueError(f"lmax must be positive and finite, got {lmax}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    # interpola g sulla variabile y in [-1, 1]
    nodes = np.cos(np.pi * (np.arange(order + 1) + 0.5) / (order + 1))
    vals = g((nodes + 1.0) * lmax / 2.0)
    # Chebyshev-Gauss quadrature: Chebyshev coefficients
    N = order + 1
    c = np.zeros(N)
    for k in range(N):
        c[k] = (2.0 / N) * np.sum(vals * np.cos(np.pi * k * (np.arange(N) + 0.5) / N))
    c[0] *= 0.5
    return c


def chebyshev_filter(x, L, coeffs, lmax: float) -> np.ndarray:
    """Apply a Chebyshev polynomial filter to ``x`` without
    diagonalizing ``L``.

    The shifted operator ``A = (2 / lmax) L - I`` maps the spectrum of
    ``L`` to ``[-1, 1]``. The recurrence
    ``T_0 x = x``, ``T_1 x = A x``, ``T_k x = 2 A T_{k-1} x - T_{k-2} x``
    is summed with ``coeffs``.

    Args:
        x: input signal, ``(n,)``.
        L: Laplacian, dense or sparse.
        coeffs: coefficients from :func:`chebyshev_coefficients`.
        lmax: majorant of the largest eigenvalue of ``L``; must be the
            same value used for ``coeffs``.

    Raises:
        ValueError: if ``lmax <= 0`` or ``coeffs`` is empty.
    """
    if not (lmax > 0 and np.isfinite(lmax)):
        raise ValueError(f"lmax must be positive and finite, got {lmax}")
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.size == 0:
        raise ValueError("coeffs must be non-empty")
    x = np.asarray(x, dtype=float)
    if hasattr(L, "toarray"):
        import scipy.sparse as sp

        n = L.shape[0]
        A = (2.0 / lmax) * L - sp.identity(n, format="csr")
    else:
        n = L.shape[0]
        A = (2.0 / lmax) * np.asarray(L, dtype=float) - np.eye(n)
    T0 = x
    out = coeffs[0] * T0
    if coeffs.size == 1:
        return out
    T1 = A @ x
    out = out + coeffs[1] * T1
    for k in range(2, coeffs.size):
        T2 = 2.0 * (A @ T1) - T0
        out = out + coeffs[k] * T2
        T0, T1 = T1, T2
    return out


class SpectralFilter:
    """A spectral filter of one graph, with the basis found once.

    The filter keeps each frequency ``lambda`` of the graph with the weight
    ``response(lambda)``: ``U g(Lambda) U^T x``.

    Args:
        graph: a :class:`Graph`.
        response: a function of the eigenvalues, returning an array.
        normalized: use the normalised Laplacian.

    Raises:
        ValueError: if ``response`` does not return one value per node.

    Example:
        >>> from dense_armor.core.graph.core import Graph
        >>> g = Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
        >>> f = SpectralFilter.heat(g, 1.0)
        >>> round(float(f.apply([0.0, 3.0, 0.0]).sum()), 6)
        3.0
    """

    def __init__(self, graph, response, normalized: bool = False) -> None:
        from dense_armor.core.graph.fourier import GraphFourier

        self._fourier = GraphFourier(graph, normalized)
        g = np.asarray(response(self._fourier.eigenvalues), dtype=float)
        if g.shape != (graph.n_nodes,):
            raise ValueError(
                f"response must return one value per node, got shape {g.shape}"
            )
        self.response = g

    @classmethod
    def heat(cls, graph, tau: float, normalized: bool = False) -> SpectralFilter:
        """The heat kernel ``exp(-tau lambda)``."""
        return cls(graph, lambda w: np.exp(-float(tau) * w), normalized)

    @classmethod
    def low_pass(
        cls, graph, cutoff: float, normalized: bool = False
    ) -> SpectralFilter:
        """1 for the frequencies up to ``cutoff``, 0 above."""
        return cls(graph, lambda w: (w <= float(cutoff)).astype(float), normalized)

    def apply(self, x) -> np.ndarray:
        """Filter the signal ``x``."""
        return self._fourier.inverse(self.response * self._fourier.transform(x))


class ChebyshevFilter:
    """A Chebyshev polynomial filter of one graph, with no eigenvectors.

    Args:
        graph: a :class:`Graph`.
        response: a function of the eigenvalues, vectorised.
        lmax: a bound of the largest eigenvalue of the Laplacian; required.
        order: the degree of the polynomial, at least 0.
        normalized: use the normalised Laplacian.

    Example:
        >>> from dense_armor.core.graph.core import Graph
        >>> g = Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
        >>> f = ChebyshevFilter(g, lambda w: np.exp(-w), lmax=4.0, order=12)
        >>> exact = SpectralFilter.heat(g, 1.0).apply([0.0, 3.0, 0.0])
        >>> bool(np.allclose(f.apply([0.0, 3.0, 0.0]), exact, atol=1e-8))
        True
    """

    def __init__(
        self, graph, response, lmax: float, order: int, normalized: bool = False
    ) -> None:
        self._L = graph.normalized_laplacian if normalized else graph.laplacian
        self.lmax = float(lmax)
        self.order = int(order)
        self.coefficients = chebyshev_coefficients(response, self.lmax, self.order)

    def apply(self, x) -> np.ndarray:
        """Filter the signal ``x``."""
        return chebyshev_filter(x, self._L, self.coefficients, self.lmax)
