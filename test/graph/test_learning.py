"""Tests for learn_graph and weights_from_drops."""

import numpy as np
import pytest
from scipy.optimize import minimize

from dense_armor.core.graph import learn_graph, weights_from_drops
from dense_armor.core.graph.learning import (
    _degree_operator,
    _pairs,
    learning_objective,
)


def test_weights_from_drops_by_hand():
    w = weights_from_drops([0.0, 1.0, 2.0, 20.0])
    assert np.allclose(w, [0.0, 1.0, 0.5, 0.0])


def test_weights_from_drops_cut_and_infinite_drop():
    assert weights_from_drops([1.0, 5.0], cut=0.1).tolist() == [1.0, 0.2]
    assert weights_from_drops([1.0, 5.0], cut=0.5).tolist() == [1.0, 0.0]
    assert weights_from_drops([1.0, np.inf]).tolist() == [1.0, 0.0]


def _signals(n=5, m=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, m))


def _objective_args(X, alpha, beta, upsilon, known):
    n = X.shape[0]
    i, j = _pairs(n)
    z = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)[i, j]
    if known is None:
        return z, np.zeros_like(z), np.zeros_like(z)
    w_pi = known[i, j]
    return z, (w_pi != 0).astype(float), w_pi


def test_the_solution_matches_a_generic_solver():
    X = _signals()
    n = X.shape[0]
    alpha, beta = 1.0, 0.1
    _, W = learn_graph(X, alpha=alpha, beta=beta, return_weights=True, max_iter=20000, tol=1e-12)
    z, m, w_pi = _objective_args(X, alpha, beta, 0.0, None)
    S = _degree_operator(n)
    i, j = _pairs(n)
    ours = learning_objective(W[i, j], z, alpha, beta, 0.0, m, w_pi, S)
    res = minimize(
        lambda w: learning_objective(w, z, alpha, beta, 0.0, m, w_pi, S),
        np.full(z.size, 0.5),
        method="L-BFGS-B",
        bounds=[(1e-9, None)] * z.size,
    )
    assert ours <= res.fun + 1e-3 * max(1.0, abs(res.fun))


def test_the_weights_are_symmetric_and_not_negative():
    G, W = learn_graph(_signals(6, 10, 1), return_weights=True)
    assert np.allclose(W, W.T) and (W >= 0).all() and (np.diag(W) == 0).all()
    assert G.n_nodes == 6


def test_a_prior_pulls_the_weights_towards_the_known_graph():
    X = _signals(6, 10, 2)
    known = np.zeros((6, 6))
    for a in range(5):
        known[a, a + 1] = known[a + 1, a] = 1.0
    _, free = learn_graph(X, return_weights=True, max_iter=8000)
    _, informed = learn_graph(X, known=known, upsilon=5.0, return_weights=True, max_iter=8000)
    on = known > 0
    assert np.abs(informed[on] - 1.0).mean() < np.abs(free[on] - 1.0).mean()


def test_similar_signals_get_a_stronger_edge_than_different_ones():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(12)
    X = np.stack([base, base + 0.01 * rng.standard_normal(12), rng.standard_normal(12) * 3])
    _, W = learn_graph(X, return_weights=True, max_iter=8000)
    assert W[0, 1] > W[0, 2]


def test_bad_arguments_are_refused():
    X = _signals()
    with pytest.raises(ValueError):
        learn_graph(X[:1])
    with pytest.raises(ValueError):
        learn_graph(X, alpha=0.0)
    with pytest.raises(ValueError):
        learn_graph(X, upsilon=-1.0)
    with pytest.raises(ValueError):
        learn_graph(X, known=np.ones((5, 5)))
    with pytest.raises(ValueError):
        learn_graph(X, known=np.zeros((3, 3)))
