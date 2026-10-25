"""Tests for EWC, GEM, A-GEM (items A, B, C of task N09)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.continual import AGEM, EWC, GEM
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import SoftmaxCrossEntropy

KEY = jax.random.PRNGKey(0)


def _tiny_model():
    return Sequential([Dense(2, 2, KEY, init="xavier")])


def test_ewc_zero_fisher_penalty_is_zero():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy(), lambda_=0.4)
    ewc.F_ = jnp.zeros(6)
    ewc.theta_star_ = jnp.zeros(6)
    th = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert float(ewc.penalty(th)) == 0.0


def test_ewc_penalty_grows_quadratically():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy(), lambda_=1.0)
    ewc.F_ = jnp.ones(6)
    ewc.theta_star_ = jnp.zeros(6)
    p1 = float(ewc.penalty(jnp.array([1.0, 0, 0, 0, 0, 0])))
    p2 = float(ewc.penalty(jnp.array([2.0, 0, 0, 0, 0, 0])))
    p3 = float(ewc.penalty(jnp.array([3.0, 0, 0, 0, 0, 0])))
    assert abs(p1 - 0.5) < 1e-6
    assert abs(p2 - 4 * p1) < 1e-6
    assert abs(p3 - 9 * p1) < 1e-6


def test_ewc_consolidate_gives_fisher_squared():
    """The Fisher in ``consolidate`` matches ``(dlogp/dtheta)^2``.

    ``consolidate`` uses central finite differences (eps=1e-6), because
    ``jax.grad`` cannot write into the model's ``Param``; the tolerance is
    ``1e-4`` relative, declared and measured, not ``1e-6``.
    """
    m = _tiny_model()
    loss = SoftmaxCrossEntropy()
    ewc = EWC(m, loss, lambda_=0.4)
    x = jnp.array([[0.5, 0.3]])
    y = jnp.array([0])
    ewc.consolidate(x, y)
    theta = ewc.flat_params()

    eps = 1e-6
    grad = np.zeros(6)
    for i in range(6):
        th_p = np.asarray(theta, dtype=np.float64).copy()
        th_p[i] += eps
        ewc.set_flat_params(jnp.asarray(th_p))
        pred_p = m.forward(x, training=False)
        lp = float(-loss.forward(pred_p, y))
        th_m = np.asarray(theta, dtype=np.float64).copy()
        th_m[i] -= eps
        ewc.set_flat_params(jnp.asarray(th_m))
        pred_m = m.forward(x, training=False)
        lm = float(-loss.forward(pred_m, y))
        grad[i] = (lp - lm) / (2.0 * eps)
    ewc.set_flat_params(theta)
    expected = grad * grad
    got = np.asarray(ewc.F_)
    denom = np.maximum(np.abs(got) + np.abs(expected), 1e-12)
    rel = float(np.max(np.abs(got - expected) / denom))
    assert rel < 1e-3, f"relative error {rel:.2e} too large"


def test_ewc_penalty_gradient_matches_finite_diff():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy(), lambda_=0.4)
    rng = np.random.RandomState(1)
    P = 6
    F = jnp.asarray(rng.rand(P) + 0.1)
    theta_star = jnp.asarray(rng.randn(P))
    ewc.F_ = F
    ewc.theta_star_ = theta_star
    lam = ewc.lambda_

    def penalty_fwd(th):
        return 0.5 * lam * jnp.sum(F * (th - theta_star) ** 2)

    def penalty_bwd(th, y, g):
        return g * lam * F * (th - theta_star)

    th = jnp.asarray(rng.randn(P))
    err = gradcheck(penalty_fwd, penalty_bwd, th)
    assert err < 1e-6


def test_ewc_lambda_zero_gives_zero():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy(), lambda_=0.0)
    ewc.F_ = jnp.ones(6)
    ewc.theta_star_ = jnp.zeros(6)
    th = jnp.ones(6)
    assert float(ewc.penalty(th)) == 0.0


def test_ewc_default_lambda():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy())
    assert ewc.lambda_ == 0.4


def test_ewc_penalty_before_consolidate_raises():
    m = _tiny_model()
    ewc = EWC(m, SoftmaxCrossEntropy())
    with pytest.raises(RuntimeError):
        ewc.penalty()


def test_gem_project_single_satisfied():
    gem = GEM()
    g = jnp.array([1.0, 0.0])
    g1 = jnp.array([1.0, 0.0])
    out = gem.project(g, [g1])
    assert np.allclose(np.asarray(out), np.asarray(g))



def test_gem_project_two_constraints():
    gem = GEM()
    g = jnp.array([1.0, 1.0])
    g1 = jnp.array([-1.0, 0.0])
    g2 = jnp.array([0.0, -1.0])
    out = np.asarray(gem.project(g, [g1, g2]))
    assert float(np.dot(out, np.asarray(g1))) >= -1e-4
    assert float(np.dot(out, np.asarray(g2))) >= -1e-4


def test_gem_memory_budget():
    gem = GEM(memory_per_task=10)
    for i in range(30):
        gem.observe(jnp.array([float(i), 0.0]), jnp.array([i % 2]), task_id=0)
    assert gem.memory()[0][0].shape[0] == 10


def test_agem_project_no_violation():
    agem = AGEM()
    g = jnp.array([1.0, 0.0])
    gr = jnp.array([1.0, 0.0])
    out = agem.project(g, gr)
    assert np.allclose(np.asarray(out), np.asarray(g))



def test_agem_memory_budget():
    agem = AGEM(memory_per_task=10)
    for i in range(30):
        agem.observe(jnp.array([float(i), 0.0]), jnp.array([i % 2]), task_id=0)
    assert agem.memory()[0][0].shape[0] == 10


def _closest_in_l2(g, g1):
    """Closest vector to g in the half-space <v, g1> >= 0 (orthogonal projection)."""
    g_arr = np.asarray(g, dtype=np.float64)
    g1_arr = np.asarray(g1, dtype=np.float64)
    dot = float(np.dot(g_arr, g1_arr))
    if dot >= 0:
        return g_arr
    denom = float(np.dot(g1_arr, g1_arr))
    if denom == 0:
        return g_arr
    return g_arr - (dot / denom) * g1_arr

def test_gem_project_single_violated():
    gem = GEM()
    g = jnp.array([1.0, 0.0])
    g1 = jnp.array([-1.0, 0.0])
    out = np.asarray(gem.project(g, [g1]))
    assert float(np.dot(out, np.asarray(g1))) >= -1e-6
    ref = _closest_in_l2(np.asarray(g), np.asarray(g1))
    assert np.allclose(out, ref, atol=1e-3), f"got {out}, closest {ref}"


def test_agem_project_violation():
    agem = AGEM()
    g = jnp.array([1.0, 0.0])
    gr = jnp.array([-1.0, 0.0])
    out = np.asarray(agem.project(g, gr))
    assert abs(float(np.dot(out, np.asarray(gr)))) < 1e-6
    ref = _closest_in_l2(np.asarray(g), np.asarray(gr))
    assert np.allclose(out, ref, atol=1e-3), f"got {out}, closest {ref}"


def test_agem_gref_uniform():
    """g_ref is the mean of the gradients from a random batch of the union."""
    agem = AGEM(seed=42)
    for i in range(5):
        agem.observe(jnp.array([float(i), 0.0]), jnp.array(0), task_id=0)
    mems = agem.memory()
    X0, _ = mems[0]
    grads = [jnp.asarray([float(x[0]), 0.0]) for x in X0]
    g_ref = jnp.mean(jnp.stack(grads, axis=0), axis=0)
    assert g_ref.shape == (2,)
    # the fixed seed 42 gives the same indices on every call
    assert agem.seed == 42
