"""Tests for LayerNorm with stable=True."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dense_armor.core.deep.neural.layers.gradcheck import gradcheck
from dense_armor.core.deep.neural.layers.norm import LayerNorm


def StableLayerNorm(n):
    return LayerNorm(n, stable=True)


def test_same_output_as_layernorm_on_well_scaled_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 16)).astype(np.float32))
    a = np.asarray(LayerNorm(16).forward(x))
    b = np.asarray(StableLayerNorm(16).forward(x))
    assert np.allclose(a, b, atol=1e-5)


def test_backward_matches_gradcheck():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.2, (4, 3)).astype(np.float32)

    def f(xx):
        return StableLayerNorm(3).forward(xx)

    def b(xx, y, g):
        ln = StableLayerNorm(3)
        ln.forward(xx)
        return ln.backward(g)

    assert gradcheck(f, b, jnp.asarray(x)) < 1e-3


def test_large_offset_small_spread_is_closer_to_float64_than_layernorm():
    rng = np.random.default_rng(2)
    x32 = (1000.0 + 0.004 * rng.normal(0.0, 1.0, (64, 512))).astype(np.float32)
    x64 = x32.astype(np.float64)
    ref = (x64 - x64.mean(axis=-1, keepdims=True)) / np.sqrt(
        x64.var(axis=-1, keepdims=True) + 1e-5
    )
    usual = np.asarray(LayerNorm(512).forward(jnp.asarray(x32)))
    stable = np.asarray(StableLayerNorm(512).forward(jnp.asarray(x32)))
    assert np.max(np.abs(stable - ref)) < np.max(np.abs(usual - ref))


def test_bfloat16_input_is_accepted():
    x = jnp.asarray(np.linspace(-1.0, 1.0, 32).reshape(2, 16), jnp.bfloat16)
    y = StableLayerNorm(16).forward(x)
    assert y.shape == (2, 16)
    assert bool(jnp.all(jnp.isfinite(y)))
