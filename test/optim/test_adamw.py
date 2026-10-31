"""Tests for AdamW."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.schedules import StepLR
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam, AdamW


def test_first_two_steps_by_hand():
    opt = AdamW(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.5)
    w1 = opt.step(np.array([1.0, -2.0]), np.array([0.5, 1.0]))
    np.testing.assert_allclose(w1, [0.40000000199999997, -1.099999999], rtol=1e-9)
    w2 = opt.step(w1, np.array([0.25, -1.0]))
    np.testing.assert_allclose(w2, [0.1067820394774624, -0.5447368416578948], rtol=1e-9)


def test_without_decay_it_is_adam():
    adam = Adam(lr=0.05)
    adamw = AdamW(lr=0.05, weight_decay=0.0)
    w_a = np.array([0.3, -0.7])
    w_b = np.array([0.3, -0.7])
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = rng.normal(size=2)
        w_a = adam.step(w_a, g)
        w_b = adamw.step(w_b, g)
    np.testing.assert_allclose(w_a, w_b, rtol=1e-12)


def test_the_decay_does_not_go_through_the_moments():
    opt = AdamW(lr=0.1, weight_decay=0.5)
    ref = Adam(lr=0.1)
    w = np.array([1.0, -2.0])
    g = np.array([0.5, 1.0])
    out = opt.step(w, g)
    np.testing.assert_allclose(out, ref.step(w, g) - 0.5 * w, rtol=1e-12)
    np.testing.assert_allclose(opt._m, ref._m, rtol=1e-12)
    np.testing.assert_allclose(opt._v, ref._v, rtol=1e-12)


def test_the_decay_is_scaled_by_eta_and_not_by_lr():
    opt = AdamW(lr=0.1, weight_decay=0.5)
    opt.eta = 0.25
    out = opt.step(np.array([1.0]), np.array([0.0]))
    np.testing.assert_allclose(out, [1.0 - 0.25 * 0.5 * 1.0], rtol=1e-12)
    other = AdamW(lr=0.001, weight_decay=0.5)
    out2 = other.step(np.array([1.0]), np.array([0.0]))
    np.testing.assert_allclose(out2, [0.5], rtol=1e-12)


def test_pure_form_equals_the_object_form():
    opt = AdamW(lr=0.05, weight_decay=0.3)
    state = opt.init_state(2)
    w_pure = jnp.array([0.3, -0.7])
    w_obj = np.array([0.3, -0.7])
    rng = np.random.default_rng(1)
    for _ in range(4):
        g = rng.normal(size=2)
        state, w_pure = opt.update(state, w_pure, jnp.asarray(g))
        w_obj = opt.step(w_obj, g)
    np.testing.assert_allclose(np.asarray(w_pure), w_obj, rtol=1e-5, atol=1e-6)


def test_repr_equality_and_validation():
    assert repr(AdamW(lr=0.1, weight_decay=0.2)).startswith("AdamW(lr=0.1")
    assert AdamW(weight_decay=0.1) == AdamW(weight_decay=0.1)
    assert AdamW(weight_decay=0.1) != AdamW(weight_decay=0.2)
    with pytest.raises(ValueError):
        AdamW(weight_decay=-0.1)
    with pytest.raises(ValueError):
        AdamW(lr=0.0)


class _Matrix:
    """A linear map with one matrix: every parameter has two dimensions."""

    def __init__(self, key):
        self.W = Param(jax.random.normal(key, (3, 2)))
        self._x = None

    def params(self):
        return [self.W]

    def forward(self, x, training=True):
        del training
        self._x = x
        return x @ self.W.value

    def backward(self, g):
        self.W.grad = self.W.grad + self._x.T @ g
        return g @ self.W.value.T


def _data():
    kx, ky = jax.random.split(jax.random.PRNGKey(7))
    return jax.random.normal(kx, (24, 3)), jax.random.normal(ky, (24, 2))


def test_same_trajectory_as_adam_with_decoupled_decay_when_all_params_are_matrices():
    X, y = _data()
    a = _Matrix(jax.random.PRNGKey(0))
    b = _Matrix(jax.random.PRNGKey(0))
    ta = Trainer(a, MSE(), Adam(0.05), weight_decay=0.2, decay="decoupled")
    tb = Trainer(b, MSE(), AdamW(0.05, weight_decay=0.2))
    ta.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    tb.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        ta.history_["train_loss"], tb.history_["train_loss"], rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(a.W.value), np.asarray(b.W.value), atol=1e-6)


def _trained_mlp_difference(schedule=None):
    X, y = _data()
    a = mlp([3, 4, 2], jax.random.PRNGKey(0))
    b = mlp([3, 4, 2], jax.random.PRNGKey(0))
    ta = Trainer(
        a, MSE(), Adam(0.05), schedule=schedule, weight_decay=0.2, decay="decoupled"
    )
    tb = Trainer(b, MSE(), AdamW(0.05, weight_decay=0.2), schedule=schedule)
    ta.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    tb.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    return max(
        float(jnp.max(jnp.abs(p.value - q.value)))
        for p, q in zip(a.params(), b.params())
    )


def test_they_differ_when_the_model_has_biases():
    assert _trained_mlp_difference() > 1e-3


def test_they_differ_when_the_schedule_decays_and_eta_is_not_set():
    schedule = StepLR(0.05, step=1.0, gamma=0.3)
    X, y = _data()
    a = _Matrix(jax.random.PRNGKey(0))
    b = _Matrix(jax.random.PRNGKey(0))
    ta = Trainer(
        a, MSE(), Adam(0.05), schedule=schedule, weight_decay=0.2, decay="decoupled"
    )
    tb = Trainer(
        b,
        MSE(),
        AdamW(0.05, weight_decay=0.2),
        schedule=StepLR(0.05, step=1.0, gamma=0.3),
    )
    ta.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    tb.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    assert float(jnp.max(jnp.abs(a.W.value - b.W.value))) > 1e-3


def test_eta_must_not_be_negative_or_nan():
    opt = AdamW(lr=0.1, weight_decay=0.5)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError):
            opt.eta = bad
    assert opt.eta == 1.0


def test_eta_zero_switches_the_decay_off():
    opt = AdamW(lr=0.1, weight_decay=0.5)
    opt.eta = 0.0
    ref = Adam(lr=0.1)
    w = np.array([1.0, -2.0])
    g = np.array([0.5, 1.0])
    np.testing.assert_allclose(opt.step(w, g), ref.step(w, g), rtol=1e-12)


def test_equality_and_copy_keep_eta():
    a = AdamW(weight_decay=0.1)
    b = AdamW(weight_decay=0.1)
    b.eta = 0.5
    assert a != b
    c = copy.deepcopy(b)
    assert c == b and c.eta == 0.5


def test_equality_against_non_adamw_returns_not_implemented():
    opt = AdamW(weight_decay=0.1)
    other = Adam(lr=0.1)
    result = opt.__eq__(other)
    assert result is NotImplemented or result is False
    assert opt != other
