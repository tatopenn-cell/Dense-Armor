"""Tests for Dense, Embedding, Dropout, Residual."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.activations import Sigmoid
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import (
    Dense,
    Dropout,
    Embedding,
    Residual,
    dense_backward,
    dense_forward,
)
from dense_armor.utility.neural.losses import (
    bce_with_logits_backward,
)
from dense_armor.utility.optim.sgd import SGD


def test_dense_backward_matches_finite_differences_for_x():
    k = jax.random.PRNGKey(0)
    d = Dense(3, 2, k)
    x = jnp.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])

    def f(xx):
        return dense_forward(d.W.value, d.b.value, xx)

    def b(xx, y, g):
        dx, _, _ = dense_backward(d.W.value, xx, g)
        return dx

    assert gradcheck(f, b, x) < 1e-4


def test_dense_backward_matches_finite_differences_for_w_and_b():
    k = jax.random.PRNGKey(1)
    d = Dense(3, 2, k)
    x = jnp.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])

    def f_w(w):
        return dense_forward(w, d.b.value, x)

    def b_w(w, y, g):
        _, dw, _ = dense_backward(w, x, g)
        return dw

    assert gradcheck(f_w, b_w, d.W.value) < 1e-4

    def f_b(bv):
        return dense_forward(d.W.value, bv, x)

    def b_b(bv, y, g):
        _, _, db = dense_backward(d.W.value, x, g)
        return db

    assert gradcheck(f_b, b_b, d.b.value) < 1e-4


def test_dense_single_sample_equals_one_row_of_a_batch():
    d = Dense(3, 2, jax.random.PRNGKey(2))
    xb = jnp.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.0]])
    yb = d.forward(xb)
    y1 = d.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]), atol=1e-6)


def test_dense_grad_accumulates_and_zero_grad_resets():
    d = Dense(3, 2, jax.random.PRNGKey(3))
    x = jnp.array([1.0, 2.0, 3.0])
    g = jnp.ones(2)
    d.forward(x)
    d.backward(g)
    first = np.array(d.W.grad)
    d.forward(x)
    d.backward(g)
    second = np.array(d.W.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)
    d.W.zero_grad()
    d.b.zero_grad()
    assert bool(jnp.all(d.W.grad == 0.0)) and bool(jnp.all(d.b.grad == 0.0))


def test_dense_and_sigmoid_and_bce_reach_accuracy_one():
    k = jax.random.PRNGKey(4)
    d = Dense(2, 1, k)
    sig = Sigmoid()
    opt = SGD(lr=0.5)
    rng = np.random.default_rng(0)
    c0 = rng.normal(-2.0, 0.5, (40, 2))
    c1 = rng.normal(2.0, 0.5, (40, 2))
    X = np.concatenate([c0, c1]).astype(np.float32)
    y = np.concatenate([np.zeros(40), np.ones(40)]).astype(np.float32)

    for _ in range(30):
        d.W.zero_grad()
        d.b.zero_grad()
        for xi, yi in zip(X, y):
            logit = d.forward(xi)[0]
            sig.forward(jnp.array([logit]))
            sig.forward(jnp.array([logit]))[0]
            g_logit = bce_with_logits_backward(jnp.array([logit]), jnp.array([yi]))[0]
            d.backward(jnp.array([g_logit]))
        d.W.update(opt)
        d.b.update(opt)

    ok = 0
    for xi, yi in zip(X, y):
        logit = d.forward(xi)[0]
        pred = 1.0 if float(logit) > 0.0 else 0.0
        if pred == float(yi):
            ok += 1
    acc = ok / len(X)
    assert acc == 1.0, f"accuracy {acc}"


def test_embedding_grad_repeated_ids():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    e.forward(jnp.array([1, 1, 3]))
    g = jnp.ones((3, 4))
    e.backward(g)
    assert bool(jnp.all(e.E.grad[1] == 2.0))
    assert bool(jnp.all(e.E.grad[3] == 1.0))
    assert bool(jnp.all(e.E.grad[0] == 0.0))


def test_embedding_out_of_range_raises():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="ids must lie"):
        e.forward(jnp.array([10]))


def test_embedding_single_id_returns_a_vector():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    y = e.forward(jnp.array(3))
    assert y.shape == (4,)


def test_dropout_mean_and_zeros_fraction():
    k = jax.random.PRNGKey(0)
    d = Dropout(0.5, k)
    x = jnp.ones(20000)
    y = np.asarray(d.forward(x, training=True))
    assert abs(y.mean() - 1.0) < 0.02
    zero_frac = float((y == 0.0).mean())
    assert abs(zero_frac - 0.5) < 0.02


def test_dropout_eval_is_identity():
    d = Dropout(0.5, jax.random.PRNGKey(0))
    x = jnp.array([1.0, 2.0, -3.0])
    assert np.array_equal(np.asarray(d.forward(x, training=False)), np.asarray(x))


def test_dropout_two_training_calls_differ():
    d = Dropout(0.5, jax.random.PRNGKey(0))
    x = jnp.ones(1000)
    y1 = np.asarray(d.forward(x, training=True))
    y2 = np.asarray(d.forward(x, training=True))
    assert not np.array_equal(y1, y2)


def test_dropout_backward_zeroes_exactly_forward_zeroes():
    d = Dropout(0.5, jax.random.PRNGKey(1))
    x = jnp.ones(5000)
    y = np.asarray(d.forward(x, training=True))
    g = np.asarray(d.backward(jnp.ones(5000)))
    zeros_y = y == 0.0
    zeros_g = g == 0.0
    assert np.array_equal(zeros_y, zeros_g)


def test_dropout_bad_p_raises():
    with pytest.raises(ValueError, match="p must be in"):
        Dropout(1.0, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="p must be in"):
        Dropout(-0.1, jax.random.PRNGKey(0))


def test_residual_of_zero_dense_is_identity():
    d = Dense(4, 4, jax.random.PRNGKey(0))
    d.W.value = jnp.zeros_like(d.W.value)
    d.b.value = jnp.zeros_like(d.b.value)
    r = Residual(d)
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(np.asarray(r.forward(x)), np.asarray(x))


def test_residual_matches_finite_differences():
    k = jax.random.PRNGKey(2)
    r = Residual(Dense(3, 3, k))

    def f(xx):
        return r.forward(xx)

    def b(xx, y, g):
        return r.backward(g)

    x = jnp.array([0.5, -1.0, 0.3])
    assert gradcheck(f, b, x) < 1e-4


def test_residual_shape_mismatch_raises():
    r = Residual(Dense(3, 2, jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match="expected the input shape"):
        r.forward(jnp.ones(3))
