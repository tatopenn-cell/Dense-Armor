"""Tests for LatentCrossAttention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.latent_attention import LatentCrossAttention

KEY = jax.random.PRNGKey(0)


def _identity_model(n_latents: int = 1) -> LatentCrossAttention:
    m = LatentCrossAttention(n_latents, 2, 1, KEY, d_context=2)
    m.latents.value = (
        jnp.array([[1.0, 0.0]]) if n_latents == 1 else jnp.zeros((n_latents, 2))
    )
    for p in m.attention.params():
        p.value = jnp.eye(2)
    return m


def test_hand_case():
    m = _identity_model()
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = m.forward(xc)
    expected_w = np.array([0.40111, 0.19778, 0.40111])
    expected_y = np.array([[0.80222, 0.59889]])
    assert np.allclose(np.asarray(m.attention.weights_[0, 0]), expected_w, atol=1e-5)
    assert np.allclose(np.asarray(out), expected_y, atol=1e-5)

    xc6 = jnp.concatenate([xc, xc], axis=0)
    out6 = m.forward(xc6)
    assert np.allclose(np.asarray(out6), expected_y, atol=1e-5)


def test_equals_cross_attention_with_latents_as_queries():
    key = jax.random.PRNGKey(2)
    m = LatentCrossAttention(3, 4, 2, key, d_context=4)
    cross = MultiHeadCrossAttention(4, 2, key, d_context=4)
    cross.W_Q.value = m.attention.W_Q.value
    cross.W_K.value = m.attention.W_K.value
    cross.W_V.value = m.attention.W_V.value
    cross.W_O.value = m.attention.W_O.value
    xc = jax.random.normal(key, (5, 4))
    y1 = np.asarray(m.forward(xc))
    y2 = np.asarray(cross.forward(m.latents.value, xc))
    assert np.allclose(y1, y2, atol=1e-6)


def test_shape_independent_of_m():
    m = _identity_model(n_latents=2)
    xc7 = jax.random.normal(KEY, (7, 2))
    xc70 = jax.random.normal(KEY, (70, 2))
    y7 = np.asarray(m.forward(xc7))
    y70 = np.asarray(m.forward(xc70))
    assert y7.shape == (2, 2)
    assert y70.shape == (2, 2)


def test_bound_m_100_and_1000():
    m = _identity_model(n_latents=2)
    for M in (100, 1000):
        xc = jax.random.normal(KEY, (M, 2))
        y = m.forward(xc)
        assert np.all(np.isfinite(np.asarray(y)))
        assert np.asarray(y).shape == (2, 2)
        w = np.asarray(m.attention.weights_[0, 0])
        assert np.allclose(w.sum(axis=-1), np.ones(w.shape[0]), atol=1e-6)
        print(f"M={M}: weights shape {w.shape}")


def test_gradcheck_xc():
    m = _identity_model()

    def f(xc_):
        return m.forward(xc_)

    def b(xc_, y, g):
        m.forward(xc_)
        return m.backward(g)

    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert gradcheck(f, b, xc) < 1e-6


def test_single_and_batch():
    m = _identity_model()
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y1 = np.asarray(m.forward(xc))
    m2 = _identity_model()
    y2 = np.asarray(m2.forward(xc[None])[0])
    assert np.allclose(y1, y2, atol=1e-6)


def test_errors():
    with pytest.raises(ValueError):
        LatentCrossAttention(0, 2, 1, KEY)
    m = _identity_model()
    with pytest.raises(RuntimeError):
        m.backward(jnp.ones((1, 2)))


def test_params():
    m = _identity_model()
    ps = m.params()
    assert len(ps) == 5
    assert ps[0] is m.latents


def test_repr():
    m = _identity_model()
    s = repr(m)
    assert "LatentCrossAttention" in s
    assert "n_latents=1" in s


def test_d_context_none_defaults_to_d_model():
    m = LatentCrossAttention(1, 4, 2, KEY)
    assert m.d_context == 4
    y = m.forward(jnp.ones((3, 4)))
    assert np.asarray(y).shape == (1, 4)


def test_gradcheck_latents():
    m = _identity_model()
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def f(lat_):
        m.latents.value = lat_
        return m.forward(xc)

    def b(lat_, y, gg):
        m.latents.value = lat_
        m.forward(xc)
        m.latents.zero_grad()
        for pp in m.attention.params():
            pp.zero_grad()
        m.backward(gg)
        return m.latents.grad

    lat0 = jnp.array([[1.0, 0.0]])
    assert gradcheck(f, b, lat0) < 1e-5
