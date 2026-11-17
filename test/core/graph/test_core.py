"""Tests for Graph: adjacency, Laplacian, normalized Laplacian."""

import networkx as nx
import numpy as np
import pytest

from dense_armor.core.graph import Graph


def _graphs():
    yield "path3", Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
    yield "star3", Graph(3, [(0, 1, 1.0), (0, 2, 1.0)])
    yield "complete3", Graph(3, [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)])
    yield "weighted", Graph(4, [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0)])
    g = Graph(4, [(0, 1, 1.0)])
    yield "isolated", g  # nodo 2, 3 isolati


@pytest.mark.parametrize("name,g", list(_graphs()))
def test_adjacency_matches_networkx(name, g):
    G_nx = g.to_networkx()
    A = nx.to_numpy_array(G_nx, nodelist=range(g.n_nodes), weight="weight")
    assert np.allclose(g.adjacency.toarray(), A, atol=1e-12)


@pytest.mark.parametrize("name,g", list(_graphs()))
def test_degrees_matches_networkx_weighted(name, g):
    G_nx = g.to_networkx()
    deg = np.array([d for _, d in G_nx.degree(weight="weight")])
    assert np.allclose(g.degrees, deg, atol=1e-12)


@pytest.mark.parametrize("name,g", list(_graphs()))
def test_laplacian_matches_networkx(name, g):
    G_nx = g.to_networkx()
    L_nx = nx.laplacian_matrix(G_nx, nodelist=range(g.n_nodes)).toarray()
    assert np.allclose(g.laplacian.toarray(), L_nx, atol=1e-12)


@pytest.mark.parametrize("name,g", list(_graphs()))
def test_normalized_laplacian_matches_networkx(name, g):
    G_nx = g.to_networkx()
    L_nx = nx.normalized_laplacian_matrix(G_nx, nodelist=range(g.n_nodes)).toarray()
    assert np.allclose(g.normalized_laplacian.toarray(), L_nx, atol=1e-12)


def test_quadratic_form_matches_edges():
    g = Graph(4, [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0)])
    x = np.array([1.0, -2.0, 0.5, 3.0])
    Lx = g.laplacian @ x
    quad = float(x @ Lx)
    edge_sum = sum(
        w * (x[i] - x[j]) ** 2 for i, j, w in [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0)]
    )
    assert abs(quad - edge_sum) < 1e-10


def test_errors():
    with pytest.raises(ValueError):
        Graph(0, [])
    with pytest.raises(ValueError):
        Graph(3, [(0, 0, 1.0)])
    with pytest.raises(ValueError):
        Graph(3, [(0, 1, -1.0)])
    with pytest.raises(ValueError):
        Graph(3, [(0, 1, 1.0), (1, 0, 2.0)])
    with pytest.raises(ValueError):
        Graph(3, [(0, 5, 1.0)])


def test_laplacian_is_indexable():
    g = Graph(3, [(0, 1, 1.0), (1, 2, 2.0)])
    assert float(g.laplacian[0, 0]) == 1.0
    assert float(g.laplacian[1, 1]) == 3.0
