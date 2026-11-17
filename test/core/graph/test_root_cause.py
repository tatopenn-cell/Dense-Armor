"""Tests for root_cause_walk (eIRWR, arXiv:2608.08073)."""

import numpy as np
import pytest
import scipy.sparse as sp

from dense_armor.core.graph import Graph, root_cause_walk
from dense_armor.core.graph.anomaly import _adaptive_exponent, _enhanced_transition


def _chain():
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])


def test_basic_walk_is_the_solution_of_the_fixed_point_equation():
    rng = np.random.default_rng(0)
    W = rng.random((6, 6)) * (rng.random((6, 6)) < 0.5)
    np.fill_diagonal(W, 0.0)
    s = rng.random(6)
    alpha, R = 0.2, 0.3
    r = root_cause_walk(W, s, resilience=R, alpha=alpha, enhanced=False, eps=1e-12)
    rs = W.sum(axis=1)
    Wn = np.divide(W, rs[:, None], out=np.zeros_like(W), where=rs[:, None] > 0)
    M = (1 - R) * Wn
    expected = alpha * np.linalg.solve(np.eye(6) - (1 - alpha) * M, s / s.sum())
    assert np.allclose(r, expected, atol=1e-9)


def test_restart_probability_one_returns_the_normalised_anomaly():
    s = np.array([2.0, 1.0, 1.0])
    r = root_cause_walk(_chain(), s, alpha=1.0, enhanced=False)
    assert np.allclose(r, s / s.sum())


def test_the_score_of_a_caller_collects_the_score_of_its_callee():
    s = np.array([0.0, 0.0, 1.0])
    r = root_cause_walk(_chain(), s, alpha=0.5, enhanced=False, eps=1e-12)
    assert r[1] > 0 and r[0] > 0 and r[2] > r[1] > r[0]


def test_enhanced_transition_by_hand_on_a_chain():
    W = sp.csr_matrix(_chain())
    C = np.array([0.4, 0.6, 1.0])
    M = _enhanced_transition(W, W, C, 0.5).toarray()
    expected = np.array([[0.0, 1.0, 0.0], [1 / 6, 0.0, 5 / 6], [0.0, 3 / 13, 10 / 13]])
    assert np.allclose(M, expected)


def test_the_enhanced_transition_is_row_stochastic():
    rng = np.random.default_rng(1)
    W = sp.csr_matrix(rng.random((8, 8)) * (rng.random((8, 8)) < 0.4))
    W = W - sp.diags(W.diagonal())
    C = rng.random(8) + 0.1
    M = _enhanced_transition(W, W, C / C.max(), 0.3)
    sums = np.asarray(M.sum(axis=1)).ravel()
    assert np.allclose(sums[sums > 0], 1.0)


def test_adaptive_exponent_by_hand():
    b = np.array([2.0, 1.0] + [0.0] * 8)
    sigma = 2.0 / 0.3
    assert _adaptive_exponent(b, 4.0) == pytest.approx(1 + 3 * (sigma - 5) / 15)
    assert _adaptive_exponent(np.ones(10), 4.0) == 1.0
    assert _adaptive_exponent(np.array([1.0] + [0.0] * 49), 4.0) == 4.0


def test_enhanced_walk_returns_finite_scores_for_every_node():
    rng = np.random.default_rng(2)
    W = rng.random((10, 10)) * (rng.random((10, 10)) < 0.3)
    np.fill_diagonal(W, 0.0)
    r = root_cause_walk(
        W,
        rng.random(10),
        resilience=0.2,
        beta=1.0,
        rho=0.5,
        q=3.0,
        outer=3,
    )
    assert r.shape == (10,) and np.isfinite(r).all() and (r >= 0).all()


def test_a_graph_object_is_read_as_symmetric_weights():
    g = Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
    r = root_cause_walk(g, [0.1, 0.1, 0.8], enhanced=False)
    assert r.shape == (3,) and r.argmax() == 2


def test_bad_arguments_are_refused():
    W, s = _chain(), [0.1, 0.2, 0.7]
    with pytest.raises(ValueError):
        root_cause_walk(np.ones((2, 3)), s)
    with pytest.raises(ValueError):
        root_cause_walk(W, [0.1, 0.2])
    with pytest.raises(ValueError):
        root_cause_walk(W, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        root_cause_walk(W, [-1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        root_cause_walk(W, s, alpha=0.0)
    with pytest.raises(ValueError):
        root_cause_walk(W, s, rho=1.0)
    with pytest.raises(ValueError):
        root_cause_walk(W, s, q=0.5)
