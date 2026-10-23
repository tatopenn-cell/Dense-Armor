"""Tests for activations_more.py and the PRelu.params() fix."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.activations import PRelu
from dense_armor.utility.neural.activations_more import (
    ELU,
    GELU,
    LeakyReLU,
    SiLU,
    Softplus,
    elu_forward,
    gelu_approx_forward,
    gelu_forward,
    leaky_relu_forward,
    silu_forward,
    softplus_forward,
)
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD


def test_gelu_exact_hand_case():
    f = GELU(approximate=False)
    for x, expected in [(1.0, 0.8413447), (-1.0, -0.1586553), (0.0, 0.0)]:
        y = float(f.forward(jnp.array(x)))
        assert abs(y - expected) < 1e-6, f"GELU({x}) = {y}, expected {expected}"
    for x, expected in [(1.0, 1.0833155), (-1.0, -0.0833155), (0.0, 0.5)]:
        f.forward(jnp.array(x))
        dx = float(f.backward(jnp.array(1.0)))
        assert abs(dx - expected) < 1e-5, f"GELU'({x}) = {dx}, expected {expected}"


def test_gelu_approx_hand_case():
    f = GELU(approximate=True)
    for x, expected in [(1.0, 0.841192), (-1.0, -0.158808)]:
        y = float(f.forward(jnp.array(x)))
        assert abs(y - expected) < 1e-4, f"GELU_approx({x}) = {y}, expected {expected}"


def test_silu_hand_case():
    f = SiLU()
    for x, expected in [(1.0, 0.7310586), (-1.0, -0.2689414)]:
        y = float(f.forward(jnp.array(x)))
        assert abs(y - expected) < 1e-6, f"SiLU({x}) = {y}, expected {expected}"
    for x, expected in [(1.0, 0.9276705), (-1.0, 0.0723295)]:
        f.forward(jnp.array(x))
        dx = float(f.backward(jnp.array(1.0)))
        assert abs(dx - expected) < 1e-6, f"SiLU'({x}) = {dx}, expected {expected}"


def test_elu_hand_case():
    f = ELU(1.0)
    y = float(f.forward(jnp.array(-1.0)))
    assert abs(y - (-0.6321206)) < 1e-6, f"ELU(-1) = {y}"
    y = float(f.forward(jnp.array(2.0)))
    assert abs(y - 2.0) < 1e-6, f"ELU(2) = {y}"
    f.forward(jnp.array(-1.0))
    dx = float(f.backward(jnp.array(1.0)))
    assert abs(dx - 0.3678794) < 1e-6, f"ELU'(-1) = {dx}"


def test_leaky_relu_hand_case():
    f = LeakyReLU(0.01)
    y = float(f.forward(jnp.array(-2.0)))
    assert abs(y - (-0.02)) < 1e-6, f"LeakyReLU(-2) = {y}"
    dx = float(f.backward(jnp.array(1.0)))
    assert abs(dx - 0.01) < 1e-6, f"LeakyReLU'(-2) = {dx}"


def test_softplus_hand_case():
    f = Softplus()
    for x, expected in [(0.0, 0.6931472), (1.0, 1.3132617)]:
        y = float(f.forward(jnp.array(x)))
        assert abs(y - expected) < 1e-6, f"Softplus({x}) = {y}, expected {expected}"


def test_gradients_against_jax_grad():
    xs = jnp.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    g = jnp.ones_like(xs)
    cases = [
        (GELU(), gelu_forward),
        (GELU(approximate=True), gelu_approx_forward),
        (SiLU(), silu_forward),
        (ELU(), lambda v: elu_forward(v, 1.0)),
        (LeakyReLU(0.01), lambda v: leaky_relu_forward(v, 0.01)),
        (Softplus(), lambda v: softplus_forward(v, 1.0)),
    ]
    for inst, fn in cases:
        _ = inst.forward(xs)
        dx = np.asarray(inst.backward(g))
        jac = np.asarray(jax.vmap(jax.grad(fn))(xs))
        assert np.allclose(dx, jac, atol=1e-5), f"{type(inst).__name__}: {dx} vs {jac}"


def test_stateless_activations_have_no_params_method():
    for inst in [GELU(), GELU(True), SiLU(), ELU(), LeakyReLU(), Softplus()]:
        assert not hasattr(inst, "params"), (
            f"{type(inst).__name__} unexpectedly has params()"
        )


def test_sequential_no_extra_param_for_stateless():
    m1 = Sequential([Dense(2, 3, jax.random.PRNGKey(0))])
    m2 = Sequential([Dense(2, 3, jax.random.PRNGKey(0)), LeakyReLU(0.01)])
    m3 = Sequential([Dense(2, 3, jax.random.PRNGKey(0)), GELU()])
    m4 = Sequential([Dense(2, 3, jax.random.PRNGKey(0)), SiLU()])
    n = len(m1.params())
    assert len(m2.params()) == n
    assert len(m3.params()) == n
    assert len(m4.params()) == n


def test_prelu_defect_fixed():
    m = Sequential(
        [
            Dense(2, 3, jax.random.PRNGKey(0)),
            PRelu(0.25),
            Dense(3, 1, jax.random.PRNGKey(1)),
        ]
    )
    ps = m.params()
    print(f"Sequential([Dense, PRelu, Dense]).params() = {len(ps)}")
    assert len(ps) == 5, f"expected 5, got {len(ps)}"

    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (30, 2)).astype(np.float32)
    y = (X[:, 0:1] + 2 * X[:, 1:2]).astype(np.float32)

    a_before = float(m.layers[1].a.value[0])
    Trainer(m, MSE(), SGD(0.05)).fit(
        X, y, epochs=20, batch_size=10, key=jax.random.PRNGKey(0)
    )
    a_after = float(m.layers[1].a.value[0])
    delta = a_after - a_before
    print(f"PRelu a: before={a_before:.6f} after={a_after:.6f} delta={delta:+.6f}")
    assert abs(delta) > 1e-6, f"PRelu.a not trained: delta={delta}"


def test_value_errors():
    with pytest.raises(ValueError):
        ELU(0.0)
    with pytest.raises(ValueError):
        ELU(-1.0)
    with pytest.raises(ValueError):
        LeakyReLU(-0.01)
    with pytest.raises(ValueError):
        Softplus(0.0)
    with pytest.raises(ValueError):
        Softplus(-1.0)


def test_backward_before_forward_raises():
    for inst in [GELU(), SiLU(), ELU(), LeakyReLU(), Softplus()]:
        with pytest.raises(RuntimeError):
            inst.backward(jnp.array(1.0))
