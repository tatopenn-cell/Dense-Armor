"""Graph spectral wavelets from Chebyshev polynomials of even and odd order.

References:
    Liu, He, Laurent, Di Giovanni, Bronstein, Bresson (2024/2025), "A
    General Graph Spectral Wavelet Convolution via Chebyshev Order
    Decomposition", arXiv:2405.13806, Section 2 (the spectral graph
    wavelet transform, Eq. (1)-(2)) and Section 3.1 (Eq. (4)-(5)): the
    transform ``T_k(y) -> (1 - T_k(y - 1)) / 2`` turns the even-order
    polynomials into band-pass wavelet functions with ``g(0) = 0`` and the
    odd-order ones into scaling functions with ``h(0) = 1``. The
    coefficients are learned in the paper by a network (Eq. (6)-(7)); here
    they are given.
"""

from __future__ import annotations

import numpy as np

from dense_armor.core.graph.fourier import GraphFourier


def _chebyshev(y: np.ndarray, k: int) -> np.ndarray:
    return np.cos(k * np.arccos(np.clip(y, -1.0, 1.0)))


def chebyshev_wavelet_terms(lam, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """The transformed Chebyshev terms on the eigenvalues ``lam`` in ``[0, 2]``.

    The transformed term of order ``k`` is ``(1 - T_k(lam - 1)) / 2`` (Eq.
    (4)). The even orders ``2, 4, .., 2 rho`` give the wavelet terms, which
    are 0 at ``lam = 0``; the odd orders ``1, 3, .., 2 rho - 1`` give the
    scaling terms, which are 1 at ``lam = 0``. ``n_terms`` is ``rho``.

    Returns:
        ``(even, odd)``, each of shape ``(n_terms, len(lam))``.

    Example:
        >>> even, odd = chebyshev_wavelet_terms([0.0, 1.0, 2.0], 2)
        >>> even[:, 0].tolist(), odd[:, 0].tolist()
        ([0.0, 0.0], [1.0, 1.0])
    """
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    y = np.asarray(lam, dtype=float) - 1.0
    even = np.stack([(1.0 - _chebyshev(y, 2 * i)) / 2.0 for i in range(1, n_terms + 1)])
    odd = np.stack(
        [(1.0 - _chebyshev(y, 2 * i - 1)) / 2.0 for i in range(1, n_terms + 1)]
    )
    return even, odd


class SpectralWavelet:
    """The scaling function and the wavelets of a graph.

    With the eigenvalues ``lambda`` of the normalised Laplacian, the scaling
    function is ``h(lambda) = sum_i b_i T^o_i(lambda)`` and the wavelet of
    scale ``s`` is ``g(s lambda) = sum_i a_i T^e_i(s lambda)``, cut to 0
    where ``s lambda`` leaves ``[0, 2]`` (Section 3.1). The operators are
    ``U h(Lambda) U^T`` and ``U g(s Lambda) U^T``.

    Args:
        graph: a :class:`Graph`.
        a: the ``rho`` coefficients of the wavelet.
        b: the ``rho`` coefficients of the scaling function.
        scales: the scales ``s_j``, positive.

    Raises:
        ValueError: if ``a`` and ``b`` differ in length, or a scale is not
            positive.

    Example:
        >>> from dense_armor.core.graph.core import Graph
        >>> w = SpectralWavelet(Graph(3, [(0, 1, 1.0), (1, 2, 1.0)]), [1.0], [1.0], [1.0])
        >>> w.scaling_response.round(3).tolist()
        [1.0, 0.5, 0.0]
    """

    def __init__(self, graph, a, b, scales) -> None:
        a = np.atleast_1d(np.asarray(a, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        s = np.atleast_1d(np.asarray(scales, dtype=float))
        if a.shape != b.shape or a.ndim != 1:
            raise ValueError("a and b must be 1-D arrays of the same length")
        if np.any(s <= 0):
            raise ValueError("scales must be positive")
        self._fourier = GraphFourier(graph, normalized=True)
        lam = self._fourier.eigenvalues
        rho = a.size
        _, odd = chebyshev_wavelet_terms(lam, rho)
        self.scaling_response = b @ odd
        rows = []
        for sj in s:
            sl = sj * lam
            even, _ = chebyshev_wavelet_terms(np.clip(sl, 0.0, 2.0), rho)
            rows.append(np.where(sl <= 2.0 + 1e-12, a @ even, 0.0))
        self.wavelet_responses = np.stack(rows)
        self.scales = s

    def scaling(self, x) -> np.ndarray:
        """The scaling-function coefficients ``U h(Lambda) U^T x``."""
        f = self._fourier
        return f.inverse(self.scaling_response * f.transform(x))

    def wavelets(self, x) -> np.ndarray:
        """The wavelet coefficients, one row per scale: ``U g(s Lambda) U^T x``."""
        f = self._fourier
        xh = f.transform(x)
        return np.stack([f.inverse(g * xh) for g in self.wavelet_responses])

    def frame_sum(self) -> np.ndarray:
        """``h(lambda)^2 + sum_j g(s_j lambda)^2`` for each eigenvalue.

        The wavelets form a tight frame if it is 1 everywhere (Section 2),
        and then the energy of the signal is kept.
        """
        return self.scaling_response**2 + (self.wavelet_responses**2).sum(axis=0)
