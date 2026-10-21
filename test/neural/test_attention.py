"""Tests for attention.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.attention import (
    MultiHeadSelfAttention,
    TransformerBlock,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    sinusoidal_positions,
)
from dense_armor.utility.neural.gradcheck import gradcheck


def _x():
    return jax.random.normal(jax.random.PRNGKey(1), (4, 8))


def test_sdpa_q_gradcheck():
    q = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    k = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    v = jax.random.normal(jax.random.PRNGKey(2), (5, 6))

    def f(qq):
        return scaled_dot_product_attention(qq, k, v)

    def b(qq, y, g):
        dq, _, _ = scaled_dot_product_attention_backward(qq, k, v, g)
        return dq

    assert gradcheck(f, b, q) < 1e-4


def test_sdpa_k_gradcheck():
    q = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    k = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    v = jax.random.normal(jax.random.PRNGKey(2), (5, 6))

    def f(kk):
        return scaled_dot_product_attention(q, kk, v)

    def b(kk, y, g):
        _, dk, _ = scaled_dot_product_attention_backward(q, kk, v, g)
        return dk

    assert gradcheck(f, b, k) < 1e-4


def test_sdpa_weights_sum_to_one():
    q = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    k = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    v = jax.random.normal(jax.random.PRNGKey(2), (5, 6))
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    w = np.asarray(jax.nn.softmax(scores, axis=-1))
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_sdpa_two_keys_by_hand():
    q = jnp.array([[1.0, 0.0]])
    k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    v = jnp.array([[10.0], [20.0]])
    y = np.asarray(scaled_dot_product_attention(q, k, v))
    d_k = 2
    s = np.array([[1.0 / np.sqrt(d_k), 0.0]])
    w = np.exp(s) / np.exp(s).sum(axis=-1, keepdims=True)
    expected = w @ np.array([[10.0], [20.0]])
    assert np.allclose(y, expected, atol=1e-6)


def test_sdpa_mask_all_but_one():
    q = jnp.array([[1.0, 0.0]])
    k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    v = jnp.array([[10.0], [20.0]])
    mask = jnp.array([[True, False]])
    y = np.asarray(scaled_dot_product_attention(q, k, v, mask))
    assert np.allclose(y, np.array([[10.0]]), atol=1e-6)


def test_sinusoidal_positions_hand():
    pe = np.asarray(sinusoidal_positions(4, 4))
    assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert np.allclose(
        pe[1], np.array([np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)]), atol=1e-6
    )


def test_sinusoidal_odd_d_model_raises():
    with pytest.raises(ValueError, match="d_model must be even"):
        sinusoidal_positions(3, 7)


def test_mhsa_d_model_not_divisible_raises():
    with pytest.raises(ValueError, match="not divisible"):
        MultiHeadSelfAttention(7, 2, jax.random.PRNGKey(0))


def test_mhsa_x_gradcheck():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    x = _x()

    def f(xx):
        m._cache = None
        return m.forward(xx)

    def b(xx, y, g):
        m._cache = None
        m.forward(xx)
        return m.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_mhsa_param_gradcheck():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    x = _x()
    for pname in ("W_Q", "W_K", "W_V", "W_O"):
        W0 = getattr(m, pname).value

        def f(w, pn=pname, W0=W0):
            getattr(m, pn).value = w.reshape(W0.shape)
            m._cache = None
            try:
                return m.forward(x)
            finally:
                getattr(m, pn).value = W0

        def b(w, y, g, pn=pname, W0=W0):
            getattr(m, pn).value = w.reshape(W0.shape)
            for p in m.params():
                p.grad = jnp.zeros_like(p.value)
            m._cache = None
            try:
                m.forward(x)
                m.backward(g)
                return getattr(m, pn).grad.reshape(-1)
            finally:
                getattr(m, pn).value = W0

        assert gradcheck(f, b, W0.reshape(-1)) < 1e-4, pname


def test_mhsa_causal_step_equals_forward():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=True)
    seq = jax.random.normal(jax.random.PRNGKey(2), (6, 8))
    yf = m.forward(seq)
    cache = m.init_cache()
    ys = []
    for t in range(6):
        yt, cache = m.step(seq[t], cache)
        ys.append(yt)
    ys = jnp.stack(ys)
    assert np.allclose(np.asarray(ys), np.asarray(yf), atol=1e-5)


def test_mhsa_causal_future_does_not_change_past():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=True)
    s1 = jax.random.normal(jax.random.PRNGKey(3), (6, 8))
    s2 = s1.at[4:].set(s1[4:] + 1.0)
    y1 = m.forward(s1)
    y2 = m.forward(s2)
    assert np.allclose(np.asarray(y1[:4]), np.asarray(y2[:4]))


def test_mhsa_batch_equals_single():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    xb = jax.random.normal(jax.random.PRNGKey(4), (2, 4, 8))
    yb = m.forward(xb)
    y1 = m.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_mhsa_grad_accumulates():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    x = _x()
    m.forward(x)
    m.backward(jnp.ones((4, 8)))
    first = np.asarray(m.W_Q.grad)
    m.forward(x)
    m.backward(jnp.ones((4, 8)))
    second = np.asarray(m.W_Q.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)


def test_mhsa_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0)).backward(jnp.ones((4, 8)))


def test_block_x_gradcheck():
    blk = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
    x = _x()

    def f(xx):
        blk._cache = None
        return blk.forward(xx)

    def b(xx, y, g):
        blk._cache = None
        blk.forward(xx)
        return blk.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_block_param_gradcheck_w_q():
    blk = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
    x = _x()
    W0 = blk.attn.W_Q.value

    def f(w):
        blk.attn.W_Q.value = w.reshape(W0.shape)
        blk._cache = None
        try:
            return blk.forward(x)
        finally:
            blk.attn.W_Q.value = W0

    def b(w, y, g):
        blk.attn.W_Q.value = w.reshape(W0.shape)
        for p in blk.params():
            p.grad = jnp.zeros_like(p.value)
        blk._cache = None
        try:
            blk.forward(x)
            blk.backward(g)
            return blk.attn.W_Q.grad.reshape(-1)
        finally:
            blk.attn.W_Q.value = W0

    assert gradcheck(f, b, W0.reshape(-1)) < 1e-4


def test_block_backward_before_forward_raises():
    blk = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        blk.backward(jnp.ones((4, 8)))


def test_sdpa_backward_with_mask():
    q = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    k = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    v = jax.random.normal(jax.random.PRNGKey(2), (3, 5))
    mask = jnp.array([[True, True, False], [True, False, False]])

    def f(qq):
        return scaled_dot_product_attention(qq, k, v, mask)

    def b(qq, y, g):
        dq, _, _ = scaled_dot_product_attention_backward(qq, k, v, g, mask)
        return dq

    assert gradcheck(f, b, q) < 1e-4


def test_repr_and_block_with_dropout():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    assert "MultiHeadSelfAttention" in repr(m)
    blk = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), dropout=0.1)
    assert "TransformerBlock" in repr(blk)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    y = blk.forward(x, training=True)
    assert y.shape == (4, 8)
    blk.backward(jnp.ones((4, 8)))


def test_block_step_equals_forward():
    blk = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), causal=True)
    seq = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
    yf = blk.forward(seq)
    cache = blk.init_cache()
    ys = []
    for t in range(5):
        yt, cache = blk.step(seq[t], cache)
        ys.append(yt)
    ys = jnp.stack(ys)
    assert np.allclose(np.asarray(ys), np.asarray(yf), atol=1e-4)
