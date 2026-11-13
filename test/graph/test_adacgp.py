"""Tests for AdaCGP."""

import numpy as np
import pytest

from dense_armor.core.graph import AdaCGP


def test_first_filter_step_from_zero_by_hand():
    m = AdaCGP(3, order=1, forgetting=1.0, mu=0.0, alpha=0.5)
    x1, x2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    m.update(x1).update(x2)
    assert np.allclose(m.psi, 0.5 * np.outer(x2, x1))


def test_the_split_parts_are_never_negative():
    m = AdaCGP(3, order=2, forgetting=0.99, mu=0.01, gamma=0.1, alpha=0.01)
    rng = np.random.default_rng(0)
    for _ in range(50):
        m.update(rng.standard_normal(3))
    assert (m._pp >= 0).all() and (m._pm >= 0).all()


def _ring(n=4):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1.0
    return W


def test_the_filter_of_a_graph_process_is_recovered():
    W = _ring()
    H = 0.2 * np.eye(4) + 0.3 * W
    rng = np.random.default_rng(0)
    x = np.zeros(4)
    m = AdaCGP(4, order=1, forgetting=0.995, mu=0.0, alpha=0.001)
    for _ in range(3000):
        x = H @ x + rng.standard_normal(4)
        m.update(x)
    assert np.linalg.norm(m.psi - H) / np.linalg.norm(H) < 0.3


def test_path_one_gives_weights_of_the_right_shape():
    m = AdaCGP(
        3, order=2, forgetting=0.99, mu=0.01, gamma=0.1, alpha=0.01, beta=0.01, path=1
    )
    rng = np.random.default_rng(0)
    for _ in range(30):
        m.update(rng.standard_normal(3))
    assert m.weights.shape == (3, 3) and np.isfinite(m.weights).all()
    assert len(m.filters) == 2


def test_debias_keeps_the_zeros_of_the_filters():
    m = AdaCGP(3, order=1, forgetting=0.99, mu=0.5, alpha=0.01)
    rng = np.random.default_rng(1)
    for _ in range(100):
        m.update(rng.standard_normal(3))
    zeros = m.psi == 0
    for _ in range(20):
        m.debias(rng.standard_normal(3))
    assert (m.psi[zeros] == 0).all()


def test_filter_coefficients_have_p_times_p_plus_three_over_two_entries():
    m = AdaCGP(3, order=2, alpha=0.01)
    rng = np.random.default_rng(2)
    for _ in range(5):
        x = rng.standard_normal(3)
        m.update_filter(x)
        m.update(x)
    assert m.filter_coefficients.shape == (5,)


def test_filter_coefficients_of_a_known_process():
    W = _ring()
    h0, h1 = 0.2, 0.3
    H = h0 * np.eye(4) + h1 * W
    rng = np.random.default_rng(3)
    m = AdaCGP(4, order=1, forgetting=0.995, mu=0.0, alpha=0.001)
    m._wp = np.maximum(W, 0.0)
    x = np.zeros(4)
    m.path = 1
    for _ in range(4000):
        x = H @ x + rng.standard_normal(4)
        m.update_filter(x, rho=0.002)
        m.update(x)
        m._wp, m._wm = np.maximum(W, 0.0), np.zeros_like(W)
    assert np.allclose(m.filter_coefficients, [h0, h1], atol=0.15)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        AdaCGP(0)
    with pytest.raises(ValueError):
        AdaCGP(3, forgetting=0.0)
    with pytest.raises(ValueError):
        AdaCGP(3, path=3)
    with pytest.raises(ValueError):
        AdaCGP(3, order=2, mu=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        AdaCGP(3).update([1.0, 2.0])
