"""Tests for AdaCGP."""

import numpy as np
import pytest

from dense_armor.checks.check_estimator import check_estimator
from dense_armor.core.graph import AdaCGP
from dense_armor.roles import Signal


def test_first_filter_step_from_zero_by_hand():
    m = AdaCGP(3, order=1, forgetting=1.0, mu=0.0, alpha=0.5)
    x1, x2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    m.learn_one(x1).learn_one(x2)
    assert np.allclose(m.psi, 0.5 * np.outer(x2, x1))


def test_the_split_parts_are_never_negative():
    m = AdaCGP(3, order=2, forgetting=0.99, mu=0.01, gamma=0.1, alpha=0.01)
    rng = np.random.default_rng(0)
    for _ in range(50):
        m.learn_one(rng.standard_normal(3))
    assert (m.psi_plus_ >= 0).all() and (m.psi_minus_ >= 0).all()


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
        m.learn_one(rng.standard_normal(3))
    assert m.weights.shape == (3, 3) and np.isfinite(m.weights).all()
    assert len(m.filters) == 2


def test_debias_keeps_the_zeros_of_the_filters():
    m = AdaCGP(3, order=1, forgetting=0.99, mu=0.5, alpha=0.01)
    rng = np.random.default_rng(1)
    for _ in range(100):
        m.learn_one(rng.standard_normal(3))
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
        m.learn_one(x)
    assert m.filter_coefficients.shape == (5,)


def test_filter_coefficients_of_a_known_process():
    W = _ring()
    h0, h1 = 0.2, 0.3
    H = h0 * np.eye(4) + h1 * W
    rng = np.random.default_rng(3)
    m = AdaCGP(4, order=1, forgetting=0.995, mu=0.0, alpha=0.001)
    m.path = 1
    x = np.zeros(4)
    for _ in range(4000):
        x = H @ x + rng.standard_normal(4)
        m.update_filter(x, rho=0.002)
        m.learn_one(x)
        m.w_plus_, m.w_minus_ = np.maximum(W, 0.0), np.zeros_like(W)
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
        AdaCGP(3).learn_one([1.0, 2.0])


@pytest.mark.parametrize("kwargs", list(AdaCGP._unit_test_params()))
def test_check_estimator(kwargs):
    check_estimator(AdaCGP(**kwargs))


def test_the_number_of_nodes_is_the_one_of_the_first_sample():
    m = AdaCGP()
    assert m.n_nodes_ is None
    m.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert m.n_nodes_ == 3 and m.psi.shape == (3, 3)
    with pytest.raises(ValueError):
        m.learn_one([1.0, 2.0])


def test_a_signal_and_a_dict_give_the_same_estimate():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((20, 3))
    a, b = AdaCGP(alpha=0.01), AdaCGP(alpha=0.01)
    for x in xs:
        a.learn_one(Signal(x.tolist(), names=["u", "v", "w"], units=["", "", ""]))
        b.learn_one(dict(zip("uvw", x.tolist())))
    assert np.allclose(a.psi, b.psi, atol=1e-3)


def test_the_input_is_not_modified():
    x = {"a": 1.0, "b": 2.0}
    AdaCGP().learn_one(x)
    assert x == {"a": 1.0, "b": 2.0}


def test_clone_forgets_and_state_dict_restores():
    m = AdaCGP(alpha=0.01)
    rng = np.random.default_rng(0)
    for _ in range(20):
        m.learn_one(rng.standard_normal(3))
    c = m.clone()
    assert c.n_nodes_ is None and c.steps_ == 0
    c.learn_one(rng.standard_normal(3))
    c.load_state_dict(m.state_dict())
    assert np.allclose(c.psi, m.psi) and c.steps_ == 20


def test_time_stamps_must_increase():
    m = AdaCGP()
    m.learn_one([1.0, 2.0], t=0.0)
    m.learn_one([1.0, 2.0], t=0.1)
    with pytest.raises(ValueError):
        m.learn_one([1.0, 2.0], t=0.05)


def test_the_declared_budget_and_memory_class():
    m = AdaCGP(4, order=2)
    assert m.memory_class == "O(1)"
    assert m.budget_s == pytest.approx(1e-2)
    assert AdaCGP(64, order=2).budget_s > 1e-2


def test_large_inputs_do_not_make_the_estimate_diverge():
    m = AdaCGP(alpha=0.01)
    rng = np.random.default_rng(0)
    for _ in range(300):
        m.learn_one(1000.0 * rng.standard_normal(3))
    assert np.isfinite(m.psi).all()


def test_a_step_below_the_cap_is_not_changed():
    m = AdaCGP(3, alpha=0.001)
    m.learn_one([1.0, 0.0, 0.0])
    m.learn_one([0.0, 1.0, 0.0])
    assert np.allclose(m._step(), 0.001)
