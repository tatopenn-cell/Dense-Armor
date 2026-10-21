"""Tests for the five activations: gradients, edge cases, batching, jit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import (
    PRelu,
    Relu,
    Sigmoid,
    Softmax,
    Tanh,
    prelu_backward,
    prelu_forward,
    relu_backward,
    relu_forward,
    sigmoid_backward,
    sigmoid_forward,
    softmax_backward,
    softmax_forward,
    tanh_backward,
    tanh_forward,
)
from dense_armor.utility.neural.gradcheck import gradcheck


def _batch(seed=0, shape=(4, 3)):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape) * 2.0


def test_relu_backward_matches_finite_differences():
    x = _batch(0)
    assert gradcheck(relu_forward, relu_backward, x) < 1e-4


def test_prelu_backward_matches_finite_differences():
    x = _batch(1)

    def f(xx):
        return prelu_forward(xx, 0.25)

    def b(xx, y, g):
        return prelu_backward(xx, y, g, 0.25)

    assert gradcheck(f, b, x) < 1e-4


def test_sigmoid_backward_matches_finite_differences():
    x = _batch(2)
    assert gradcheck(sigmoid_forward, sigmoid_backward, x) < 1e-5


def test_tanh_backward_matches_finite_differences():
    x = _batch(3)
    assert gradcheck(tanh_forward, tanh_backward, x) < 1e-5


def test_softmax_backward_matches_finite_differences():
    x = _batch(4)
    assert gradcheck(softmax_forward, softmax_backward, x) < 1e-4


def test_relu_gradient_at_negative_zero_positive():
    x = jnp.array([-1.0, 0.0, 1.0])
    y = relu_forward(x)
    g = jnp.ones(3)
    dx = relu_backward(x, y, g)
    assert np.allclose(np.asarray(dx), np.array([0.0, 0.0, 1.0]))


def test_softmax_rows_sum_to_one():
    x = _batch(5, (8, 5))
    y = softmax_forward(x)
    sums = np.asarray(jnp.sum(y, axis=-1))
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_softmax_does_not_overflow_on_1000():
    x = jnp.array([[1000.0, 0.0, -1000.0]])
    y = softmax_forward(x)
    assert np.all(np.isfinite(np.asarray(y)))
    assert abs(float(jnp.sum(y)) - 1.0) < 1e-6


def test_sigmoid_stable_at_large_negative():
    x = jnp.array([-1000.0, 1000.0])
    y = sigmoid_forward(x)
    assert np.all(np.isfinite(np.asarray(y)))


def test_tanh_forward_matches_numpy():
    x = _batch(6)
    y = tanh_forward(x)
    assert np.allclose(np.asarray(y), np.tanh(np.asarray(x)), atol=1e-6)


def test_batch_of_one_equals_row_of_eight():
    x8 = _batch(7, (8, 3))
    y8 = softmax_forward(x8)
    x1 = x8[3:4]
    y1 = softmax_forward(x1)
    assert np.allclose(np.asarray(y1[0]), np.asarray(y8[3]), atol=1e-6)


def test_jit_pure_functions_equal_eager():
    x = _batch(8)
    for f in (relu_forward, sigmoid_forward, tanh_forward, softmax_forward):
        eager = np.asarray(f(x))
        jitted = np.asarray(jax.jit(f)(x))
        assert np.allclose(eager, jitted, atol=1e-6)


def test_classes_cache_and_backward():
    x = _batch(9)
    g = jnp.ones_like(x)
    for cls in (Relu, Sigmoid, Tanh, Softmax):
        layer = cls()
        _ = layer.forward(x)
        dx = layer.backward(g)
        assert dx.shape == x.shape
        assert np.all(np.isfinite(np.asarray(dx)))


def test_prelu_class_grad_on_a():
    x = jnp.array([-2.0, 1.0, -0.5])
    layer = PRelu(a=0.25)
    _ = layer.forward(x)
    g = jnp.ones_like(x)
    dx = layer.backward(g)
    assert dx.shape == x.shape
    assert np.all(np.isfinite(np.asarray(dx)))
    assert np.all(np.isfinite(np.asarray(layer.a.grad)))
    assert layer.a.grad.shape == layer.a.value.shape


def test_backward_before_forward_raises_for_all_classes():
    import pytest

    for cls in (Relu, Sigmoid, Tanh, Softmax, PRelu):
        layer = cls()
        with pytest.raises(RuntimeError, match="call forward"):
            layer.backward(jnp.ones((2, 2)))


def test_prelu_accumulates_a_grad():
    x = jnp.array([-2.0, 1.0, -0.5])
    layer = PRelu(a=0.25)
    g = jnp.ones_like(x)
    _ = layer.forward(x)
    _ = layer.backward(g)
    first = float(layer.a.grad[0])
    _ = layer.forward(x)
    _ = layer.backward(g)
    second = float(layer.a.grad[0])
    assert abs(second - 2.0 * first) < 1e-5, (
        f"first={first}, second={second}, expected second = 2 * first"
    )


def test_prelu_zero_grad_after_accumulation():
    x = jnp.array([-2.0, 1.0, -0.5])
    layer = PRelu(a=0.25)
    _ = layer.forward(x)
    _ = layer.backward(jnp.ones_like(x))
    layer.a.zero_grad()
    assert float(layer.a.grad[0]) == 0.0
