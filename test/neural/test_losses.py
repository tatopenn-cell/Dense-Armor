"""Tests for the losses: gradient check, known values, stability, edge cases."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.losses import (
    MSE,
    BCEWithLogits,
    SoftmaxCrossEntropy,
    bce_with_logits_backward,
    bce_with_logits_forward,
    mse_backward,
    mse_forward,
    softmax_cross_entropy_backward,
    softmax_cross_entropy_forward,
)


def _batch(seed, shape):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape) * 2.0


def test_mse_backward_matches_finite_differences():
    x = _batch(0, (4, 3))
    t = _batch(1, (4, 3))

    def f(xx):
        return mse_forward(xx, t)

    def b(xx, y, g):
        del y
        return g * mse_backward(xx, t)

    assert gradcheck(f, b, x) < 1e-4


def test_softmax_cross_entropy_backward_matches_finite_differences():
    x = _batch(2, (4, 3))
    labels = jnp.array([0, 1, 2, 1])

    def f(xx):
        return softmax_cross_entropy_forward(xx, labels)

    def b(xx, y, g):
        del y
        return g * softmax_cross_entropy_backward(xx, labels)

    assert gradcheck(f, b, x) < 1e-4


def test_bce_backward_matches_finite_differences():
    x = _batch(3, (4,))
    t = (jax.random.uniform(jax.random.PRNGKey(4), (4,)) > 0.5).astype(jnp.float32)

    def f(xx):
        return bce_with_logits_forward(xx, t)

    def b(xx, y, g):
        del y
        return g * bce_with_logits_backward(xx, t)

    assert gradcheck(f, b, x) < 1e-4


def test_bce_backward_is_sigmoid_minus_target():
    z = jnp.array([-1.0, 0.0, 1.0, 2.0])
    t = jnp.array([0.0, 0.5, 1.0, 0.0])
    expected = (jax.nn.sigmoid(z) - t) / 4.0
    got = bce_with_logits_backward(z, t)
    assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-6)


def test_softmax_cross_entropy_gradient_matches_hand_computation():
    logits = jnp.array([[2.0, 0.0, -1.0]])
    labels = jnp.array([0])
    s = math.exp(2.0) + math.exp(0.0) + math.exp(-1.0)
    sm = np.array([[math.exp(2.0) / s, math.exp(0.0) / s, math.exp(-1.0) / s]])
    onehot = np.array([[1.0, 0.0, 0.0]])
    expected = (sm - onehot) / 1.0
    got = np.asarray(softmax_cross_entropy_backward(logits, labels))
    assert np.allclose(got, expected, atol=1e-6)


def test_softmax_cross_entropy_is_finite_on_1000():
    logits = jnp.array([[1000.0, 0.0, -1000.0]])
    labels = jnp.array([0])
    val = softmax_cross_entropy_forward(logits, labels)
    assert np.isfinite(np.asarray(val))


def test_bce_is_finite_on_pm1000():
    z = jnp.array([1000.0, -1000.0])
    t = jnp.array([1.0, 0.0])
    val = bce_with_logits_forward(z, t)
    assert np.isfinite(np.asarray(val))


def test_mse_perfect_prediction_is_zero():
    x = _batch(5, (3, 2))
    assert float(mse_forward(x, x)) == 0.0


def test_classes_cache_and_backward():
    x = _batch(6, (3, 2))
    t = _batch(7, (3, 2))
    layer = MSE()
    _ = layer.forward(x, t)
    dx = layer.backward()
    assert dx.shape == x.shape

    logits = _batch(8, (4, 3))
    labels = jnp.array([0, 1, 2, 1])
    layer = SoftmaxCrossEntropy()
    _ = layer.forward(logits, labels)
    d = layer.backward()
    assert d.shape == logits.shape

    z = _batch(9, (4,))
    tt = jnp.ones((4,))
    layer = BCEWithLogits()
    _ = layer.forward(z, tt)
    d = layer.backward()
    assert d.shape == z.shape


def test_classes_backward_before_forward_raises():
    for layer in (MSE(), SoftmaxCrossEntropy(), BCEWithLogits()):
        with pytest.raises(RuntimeError, match="call forward"):
            layer.backward()
