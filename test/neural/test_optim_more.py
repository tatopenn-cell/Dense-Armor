"""Tests for SGD momentum/Nesterov and RMSProp."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, RMSProp


def test_momentum_hand_case():
    opt = SGD(lr=0.1, momentum=0.9)
    w = np.array(1.0)
    for expected in [0.9, 0.71]:
        w = opt.step(w, np.array(1.0))
        assert abs(float(w) - expected) < 1e-6, f"got {float(w)} expected {expected}"


def test_nesterov_hand_case():
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
    w = np.array(1.0)
    for expected in [0.81, 0.539]:
        w = opt.step(w, np.array(1.0))
        assert abs(float(w) - expected) < 1e-6, f"got {float(w)} expected {expected}"


def test_nesterov_on_quadratic():
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
    w = np.array(1.0)
    for expected in [0.81, 0.5751, 0.327321]:
        g = w.copy()
        w = opt.step(w, g)
        assert abs(float(w) - expected) < 1e-6, f"got {float(w)} expected {expected}"


def test_rmsprop_hand_case():
    opt = RMSProp(lr=0.01)
    w = np.array(1.0)
    for expected in [0.96837722, 0.94543565]:
        w = opt.step(w, np.array(1.0))
        assert abs(float(w) - expected) < 1e-7, f"got {float(w)} expected {expected}"


def test_sgd_zero_momentum_unchanged():
    opt = SGD(0.1)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = weights.copy()
    for _ in range(5):
        got = opt.step(got, np.ones_like(got))
    expected = weights - 0.5
    assert np.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize(
    "opt",
    [
        SGD(0.1, momentum=0.9),
        SGD(0.1, momentum=0.9, nesterov=True),
        RMSProp(0.05),
    ],
)
def test_pure_update_matches_step(opt):
    w_obj = np.zeros(3)
    w_pure = jnp.zeros(3)
    state = opt.init_state(3)
    for i in range(10):
        g = np.array([1.0, 2.0, -1.0]) * (i + 1) / 10.0
        w_obj = opt.step(w_obj, g)
        state, w_pure = opt.update(state, w_pure, jnp.asarray(g))
        assert np.allclose(w_obj, np.asarray(w_pure), atol=1e-6)


@pytest.mark.parametrize(
    "opt",
    [
        SGD(0.05, momentum=0.9),
        SGD(0.05, momentum=0.9, nesterov=True),
        RMSProp(0.05),
    ],
)
@pytest.mark.training
def test_learning_with_trainer(opt):
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (200, 2)).astype(np.float32)
    y = (X[:, 0:1] + 2 * X[:, 1:2]).astype(np.float32)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    before = float(
        np.mean((np.asarray(m.forward(jnp.asarray(X), training=False)) - y) ** 2)
    )
    t = Trainer(m, MSE(), opt)
    t.fit(X, y, epochs=100, batch_size=20, key=jax.random.PRNGKey(1))
    after = float(
        np.mean((np.asarray(m.forward(jnp.asarray(X), training=False)) - y) ** 2)
    )
    print(f"{type(opt).__name__}: before={before:.4f} after={after:.4f}")
    assert after < before / 10.0, f"{after} not < {before / 10.0}"


def test_sgd_value_errors():
    with pytest.raises(ValueError):
        SGD(lr=0.0)
    with pytest.raises(ValueError):
        SGD(lr=0.1, momentum=-0.1)
    with pytest.raises(ValueError):
        SGD(lr=0.1, momentum=1.0)
    with pytest.raises(ValueError):
        SGD(lr=0.1, momentum=0.0, nesterov=True)


def test_rmsprop_value_errors():
    with pytest.raises(ValueError):
        RMSProp(lr=0.0)
    with pytest.raises(ValueError):
        RMSProp(rho=-0.1)
    with pytest.raises(ValueError):
        RMSProp(rho=1.0)
    with pytest.raises(ValueError):
        RMSProp(eps=0.0)


def test_sgd_eq_and_repr():
    a = SGD(0.1)
    b = SGD(0.1)
    c = SGD(0.1, momentum=0.9)
    assert a == b
    assert a != c
    assert a != "SGD(0.1)"
    assert repr(a) == "SGD(lr=0.1)"
    assert "momentum=0.9" in repr(c)
    assert "nesterov=True" in repr(SGD(0.1, momentum=0.9, nesterov=True))


def test_rmsprop_eq_and_repr():
    a = RMSProp(0.01)
    b = RMSProp(0.01)
    c = RMSProp(0.01, rho=0.5)
    assert a == b
    assert a != c
    assert a != "RMSProp"
    assert repr(a) == "RMSProp(lr=0.01, rho=0.9, eps=1e-08)"


def test_sgd_init_state_without_momentum_is_empty():
    assert SGD(0.1).init_state(3) == {}
    state = SGD(0.1, momentum=0.9).init_state(3)
    assert "u" in state
    assert state["u"].shape == (3,)


def test_sgd_pure_update_without_momentum():
    opt = SGD(0.1)
    w = jnp.array([1.0, 2.0])
    g = jnp.array([0.5, -0.5])
    state, w_new = opt.update({}, w, g)
    assert state == {}
    assert np.allclose(np.asarray(w_new), np.asarray(w - 0.1 * g), atol=1e-7)
    w_step = opt.step(np.asarray(w), np.asarray(g))
    assert np.allclose(np.asarray(w_new), w_step, atol=1e-7)
