"""Tests for eigendecomposition and fiedler_vector."""

import numpy as np
import scipy.sparse.linalg as spla

from dense_armor.core.graph import Graph, eigendecomposition, fiedler_vector


def _path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def test_path_eigenvalues_closed_form():
    n = 10
    g = _path(n)
    w, U = eigendecomposition(g.laplacian)
    expected = np.array([2.0 - 2.0 * np.cos(np.pi * k / n) for k in range(n)])
    assert np.allclose(np.sort(w), np.sort(expected), atol=1e-10)


def test_cycle_eigenvalues_closed_form():
    n = 10
    # ciclo: n archi (0-1, 1-2, ..., n-2, n-1, n-1, 0)
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    edges.append((0, n - 1, 1.0))
    g = Graph(n, edges)
    w, _ = eigendecomposition(g.laplacian)
    expected = np.array([2.0 - 2.0 * np.cos(2.0 * np.pi * k / n) for k in range(n)])
    assert np.allclose(np.sort(w), np.sort(expected), atol=1e-10)


def test_residual_and_orthonormal():
    g = _path(10)
    w, U = eigendecomposition(g.laplacian)
    L = g.laplacian.toarray()
    resid = L @ U - U * w
    assert np.max(np.abs(resid)) < 1e-8
    assert np.max(np.abs(U.T @ U - np.eye(len(w)))) < 1e-10


def test_eigsh_k_n_minus_1():
    n = 10
    g = _path(n)
    w, _ = eigendecomposition(g.laplacian)
    w_partial = spla.eigsh(
        g.laplacian.astype(float), k=n - 1, which="SA", return_eigenvectors=False
    )
    assert np.allclose(np.sort(w_partial), w[:-1], atol=1e-8)


def test_fiedler_path_closed_form():
    n = 10
    g = _path(n)
    v = fiedler_vector(g.laplacian)
    expected = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    expected = expected / np.linalg.norm(expected)
    if v[0] < 0:
        v = -v
    if expected[0] < 0:
        expected = -expected
    assert np.allclose(v, expected, atol=1e-10)


def test_eigenvalues_sum_to_trace():
    g = _path(10)
    w, _ = eigendecomposition(g.laplacian)
    assert abs(w.sum() - np.trace(g.laplacian.toarray())) < 1e-10


def test_disconnected_two_zero_eigenvalues():
    g = Graph(4, [(0, 1, 1.0), (2, 3, 1.0)])
    w, _ = eigendecomposition(g.laplacian)
    assert abs(w[0]) < 1e-10
    assert abs(w[1]) < 1e-10
