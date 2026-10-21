"""Tests for BatchNorm and LayerNorm."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.norm import BatchNorm, LayerNorm


def test_batchnorm_training_mean_and_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (64, 4)).astype(np.float32)
    bn = BatchNorm(4)
    y = np.asarray(bn.forward(jnp.asarray(x)))
    mu = y.mean(axis=0)
    var = y.var(axis=0)
    assert np.allclose(mu, 0.0, atol=1e-4)
    var_x = x.var(axis=0)
    expected = var_x / (var_x + bn.eps)
    assert np.allclose(var, expected, atol=1e-3)


def test_batchnorm_running_var_matches_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 2.0, (16, 3)).astype(np.float32)
    bn = BatchNorm(3)
    bn.forward(jnp.asarray(x))
    m = 16
    var = x.var(axis=0)
    unbiased = var * m / (m - 1)
    expected = (1.0 - bn.momentum) * 1.0 + bn.momentum * unbiased
    assert np.allclose(np.asarray(bn.running_var_), expected, atol=1e-6)


def test_batchnorm_eval_single_sample_equals_batch_row():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, (8, 3)).astype(np.float32)
    bn = BatchNorm(3)
    bn.forward(jnp.asarray(x))
    y_batch = np.asarray(bn.forward(jnp.asarray(x), training=False))
    y_single = np.asarray(bn.forward(jnp.asarray(x[3]), training=False))
    assert np.allclose(y_single, y_batch[3], atol=1e-6)


def test_batchnorm_backward_matches_finite_differences():
    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 1.2, (5, 3)).astype(np.float32)
    BatchNorm(3)

    def f(xx):
        bn2 = BatchNorm(3)
        return bn2.forward(xx, training=True)

    def b(xx, y, g):
        bn3 = BatchNorm(3)
        bn3.forward(xx, training=True)
        return bn3.backward(g)

    assert gradcheck(f, b, jnp.asarray(x)) < 1e-3


def test_layernorm_mean_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(3.0, 5.0, (6, 4)).astype(np.float32)
    ln = LayerNorm(4)
    y = np.asarray(ln.forward(jnp.asarray(x)))
    mu = y.mean(axis=-1)
    assert np.allclose(mu, 0.0, atol=1e-5)


def test_layernorm_single_sample_equals_batch_row():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, (8, 3)).astype(np.float32)
    ln = LayerNorm(3)
    y_batch = np.asarray(ln.forward(jnp.asarray(x)))
    y_single = np.asarray(ln.forward(jnp.asarray(x[2])))
    assert np.allclose(y_single, y_batch[2], atol=1e-6)


def test_layernorm_backward_matches_finite_differences():
    rng = np.random.default_rng(6)
    x = rng.normal(0.5, 1.2, (4, 3)).astype(np.float32)
    LayerNorm(3)

    def f(xx):
        ln2 = LayerNorm(3)
        return ln2.forward(xx)

    def b(xx, y, g):
        ln3 = LayerNorm(3)
        ln3.forward(xx)
        return ln3.backward(g)

    assert gradcheck(f, b, jnp.asarray(x)) < 1e-3


def test_batchnorm_single_sample_training_raises():
    bn = BatchNorm(3)
    with pytest.raises(ValueError, match="at least two samples"):
        bn.forward(jnp.ones((1, 3)), training=True)


def test_batchnorm_grad_accumulates():
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 3)).astype(np.float32))
    bn = BatchNorm(3)
    g = jnp.ones((4, 3))
    bn.forward(x, training=True)
    bn.backward(g)
    first = np.array(bn.gamma.grad)
    bn.forward(x, training=True)
    bn.backward(g)
    second = np.array(bn.gamma.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)


def test_layernorm_grad_accumulates():
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 3)).astype(np.float32))
    ln = LayerNorm(3)
    g = jnp.ones((4, 3))
    ln.forward(x)
    ln.backward(g)
    first = np.array(ln.gain.grad)
    ln.forward(x)
    ln.backward(g)
    second = np.array(ln.gain.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)
