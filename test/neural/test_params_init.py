"""Tests for Param, he_normal, xavier_normal, zeros_bias, fan_in."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.init import (
    fan_in,
    he_normal,
    xavier_normal,
    zeros_bias,
)
from dense_armor.utility.neural.params import Param
from dense_armor.utility.optim.sgd import SGD


def test_param_starts_with_zero_grad_of_value_shape():
    v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    p = Param(v)
    assert p.value.shape == v.shape
    assert p.grad.shape == v.shape
    assert p.grad.dtype == v.dtype
    assert bool(jnp.all(p.grad == 0.0))


def test_zero_grad_sets_grad_to_zero():
    p = Param(jnp.array([1.0, 2.0]))
    p.grad = jnp.array([5.0, 6.0])
    p.zero_grad()
    assert bool(jnp.all(p.grad == 0.0))


def test_update_with_sgd_moves_value_by_minus_lr_grad():
    p = Param(jnp.array([1.0, 2.0]))
    p.grad = jnp.array([0.5, -0.3])
    opt = SGD(lr=0.1)
    p.update(opt)
    expected = np.array([1.0, 2.0]) - 0.1 * np.array([0.5, -0.3])
    assert np.allclose(np.asarray(p.value), expected, atol=1e-6)


def test_update_with_wrong_shape_from_optimizer_raises():
    class BadOpt:
        def step(self, weights, gradient):
            return jnp.zeros((3,))

    p = Param(jnp.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="optimizer.step returned shape"):
        p.update(BadOpt())


def test_param_rejects_scalar():
    with pytest.raises(ValueError, match="must be an array"):
        Param(3.0)


def test_he_normal_std_matches_formula():
    key = jax.random.PRNGKey(0)
    w = he_normal(key, (1024, 1024))
    std = float(jnp.std(w))
    expected = (2.0 / 1024.0) ** 0.5
    assert abs(std - expected) / expected < 0.02


def test_he_normal_with_a_one_equals_xavier_within_two_percent():
    key = jax.random.PRNGKey(1)
    wh = he_normal(key, (1024, 1024), a=1.0)
    wx = xavier_normal(key, (1024, 1024))
    sh = float(jnp.std(wh))
    sx = float(jnp.std(wx))
    assert abs(sh - sx) / sx < 0.02


def test_xavier_normal_std_matches_formula():
    key = jax.random.PRNGKey(2)
    w = xavier_normal(key, (1024, 1024))
    std = float(jnp.std(w))
    expected = (1.0 / 1024.0) ** 0.5
    assert abs(std - expected) / expected < 0.02


def test_fan_in_is_shape_zero():
    assert fan_in((8, 4)) == 8
    assert fan_in((1,)) == 1
    assert fan_in((256, 128, 3)) == 256


def test_fan_in_empty_shape_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        fan_in(())


def test_zeros_bias_shape_and_values():
    b = zeros_bias(3)
    assert b.shape == (3,)
    assert bool(jnp.all(b == 0.0))
