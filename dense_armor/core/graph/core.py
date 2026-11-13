"""Graph structure: weighted undirected graph, Laplacians.

A :class:`Graph` wraps a weighted undirected graph on ``n_nodes``
nodes ``0 .. n-1``. ``adjacency`` is the symmetric weight matrix
``W``, ``laplacian`` is ``L = D - W`` and ``normalized_laplacian`` is
``I - D^{-1/2} W D^{-1/2}`` (a node with no edges has row and column
of zeros in the normalized Laplacian, the convention of
``networkx.normalized_laplacian_matrix``).

References:
    Niresi, Kuhn, Frusque, Fink (2024), "Informed Graph Learning By
    Domain Knowledge Injection and Smooth Graph Signal
    Representation", arXiv:2406.03898, Section II.A, Eq. (1)-(3):
    ``L = D - W`` and ``S_2(x) = x^T L x``.
    Shuman et al. (2013): the normalized Laplacian (bibliographic
    only, the PDF is not in ``papers/``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as _csgraph_laplacian


class Graph:
    """Weighted undirected graph on ``n_nodes`` nodes.

    Args:
        n_nodes: number of nodes, at least 1.
        edges: list of ``(i, j, w)`` triples with ``0 <= i, j < n``,
            ``w > 0`` finite. No self-loops, no repeated pair
            (``(i, j)`` and ``(j, i)`` are the same).

    Raises:
        ValueError: on a bad node index, a self-loop, a repeated pair,
            a non-positive or non-finite weight, or ``n_nodes < 1``.

    Example:
        >>> g = Graph(3, [(0, 1, 1.0), (1, 2, 2.0)])
        >>> g.n_nodes, g.n_edges
        (3, 2)
        >>> float(g.laplacian[0, 0])
        1.0
    """

    def __init__(self, n_nodes: int, edges: list[tuple[int, int, float]]) -> None:
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
        n = int(n_nodes)
        rows, cols, data = [], [], []
        seen: set[tuple[int, int]] = set()
        clean_edges = []
        for triple in edges:
            if len(triple) != 3:
                raise ValueError(f"edge must be (i, j, w), got {triple!r}")
            i, j, w = triple
            if not (
                isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))
            ):
                raise ValueError(f"node indices must be ints, got {triple!r}")
            i, j, w = int(i), int(j), float(w)
            if not (0 <= i < n) or not (0 <= j < n):
                raise ValueError(f"node index out of range: {triple!r}")
            if i == j:
                raise ValueError(f"self-loop not allowed: {triple!r}")
            if not np.isfinite(w) or w <= 0:
                raise ValueError(f"weight must be > 0 finite, got {w}")
            key = (min(i, j), max(i, j))
            if key in seen:
                raise ValueError(f"repeated pair {key}")
            seen.add(key)
            clean_edges.append((i, j, w))
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
        self.n_nodes = n
        self.n_edges = len(clean_edges)
        self._edges = clean_edges
        W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        self.adjacency = W
        self.degrees = np.asarray(W.sum(axis=1)).ravel()
        self.laplacian = _csgraph_laplacian(W, normed=False, return_diag=False).tocsr()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(self.degrees)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        I_mat = sp.identity(n, format="csr")
        L_norm = I_mat - D_inv_sqrt @ W @ D_inv_sqrt
        isolated = self.degrees == 0
        if np.any(isolated):
            L_norm = L_norm.tolil()
            for k in np.where(isolated)[0]:
                L_norm[k, :] = 0
                L_norm[:, k] = 0
            L_norm = L_norm.tocsr()
        self.normalized_laplacian = L_norm

    @property
    def edges(self) -> list[tuple[int, int, float]]:
        """The edges as ``(i, j, w)`` triples."""
        return list(self._edges)

    def degree(self, i: int) -> float:
        """Weighted degree of node ``i``."""
        return float(self.degrees[i])

    def neighbors(self, i: int) -> list[int]:
        """Neighbors of node ``i``."""
        return [int(k) for k in self.adjacency[i].indices]

    def to_networkx(self):
        """Return a ``networkx.Graph`` with the same weights."""
        import networkx as nx  # type: ignore[import-untyped]

        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        for i, j, w in self._edges:
            G.add_edge(int(i), int(j), weight=float(w))
        return G

    def to_pygsp(self):
        """Return a ``pygsp.graphs.Graph`` with the same weights."""
        import pygsp  # type: ignore[import-untyped]

        return pygsp.graphs.Graph(self.adjacency, lap_type="combinatorial")
