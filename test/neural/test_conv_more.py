"""Tests for conv_more.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.conv import Conv2D
from dense_armor.utility.neural.conv_more import (
    Conv3D,
    DepthwiseConv2D,
    SeparableConv2D,
)
from dense_armor.utility.neural.layers import Dense


def test_conv3d_hand_case_ones():
    c = Conv3D(1, 1, 3, jax.random.PRNGKey(0))
    c.W.value = jnp.ones_like(c.W.value)
    c.b.value = jnp.zeros_like(c.b.value)
    y = np.asarray(c.forward(jnp.ones((4, 4, 4, 1))))
    print(f"Conv3D: shape {y.shape}, all 27? {np.allclose(y, 27.0)}")
    assert y.shape == (2, 2, 2, 1)
    assert np.allclose(y, 27.0)


def test_conv3d_hand_case_padding1():
    c = Conv3D(1, 1, 3, jax.random.PRNGKey(0), padding=1)
    c.W.value = jnp.ones_like(c.W.value)
    c.b.value = jnp.zeros_like(c.b.value)
    y = np.asarray(c.forward(jnp.ones((4, 4, 4, 1))))
    print(f"Conv3D pad=1: corner={y[0, 0, 0, 0]}, center={y[2, 2, 2, 0]}")
    assert y.shape == (4, 4, 4, 1)
    assert np.allclose(y[0, 0, 0, 0], 8.0)
    assert np.allclose(y[2, 2, 2, 0], 27.0)


def test_depthwise_hand_case():
    x = jnp.zeros((2, 2, 2))
    x = x.at[:, :, 0].set(1.0)
    x = x.at[:, :, 1].set(2.0)
    dw = DepthwiseConv2D(2, 2, jax.random.PRNGKey(0))
    dw.W.value = jnp.ones_like(dw.W.value)
    dw.b.value = jnp.zeros_like(dw.b.value)
    y = np.asarray(dw.forward(x))
    print(f"Depthwise: shape {y.shape}, values {y.reshape(-1).tolist()}")
    assert y.shape == (1, 1, 2)
    assert np.allclose(y.reshape(-1), [4.0, 8.0])


def test_depthwise_channels_independent():
    x = jnp.zeros((2, 2, 2))
    x = x.at[:, :, 0].set(1.0)
    x = x.at[:, :, 1].set(2.0)
    dw = DepthwiseConv2D(2, 2, jax.random.PRNGKey(0))
    dw.W.value = jnp.ones_like(dw.W.value)
    dw.b.value = jnp.zeros_like(dw.b.value)
    y1 = np.asarray(dw.forward(x))
    x2 = x.at[:, :, 1].set(5.0)
    y2 = np.asarray(dw.forward(x2))
    print(f"channel 0 out: y1={y1[0, 0, 0]:.2f}, y2={y2[0, 0, 0]:.2f}")
    assert np.allclose(y1[0, 0, 0], y2[0, 0, 0])


def test_separable_param_count():
    sep = SeparableConv2D(2, 3, 2, jax.random.PRNGKey(0))
    n = sum(p.value.size for p in sep.params())
    print(f"SeparableConv2D(2,3,2): {n} parametri")
    assert n == 19


def test_separable_matches_manual_composition():
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 4, 2)).astype(np.float32))

    sep = SeparableConv2D(2, 3, 3, key)
    sep_w = sep.depthwise.W.value
    sep_b = sep.depthwise.b.value
    sep_pw_w = sep.pointwise.W.value
    sep_pw_b = sep.pointwise.b.value

    dw = DepthwiseConv2D(2, 3, key)
    dw.W.value = sep_w
    dw.b.value = sep_b
    conv = Conv2D(2, 3, 1, jax.random.PRNGKey(1))
    conv.W.value = sep_pw_w
    conv.b.value = sep_pw_b

    y_sep = np.asarray(sep.forward(x))
    y_dw = dw.forward(x)
    y_pw = np.asarray(conv.forward(y_dw))
    print(f"sep vs dw+pw: max diff {np.max(np.abs(y_sep - y_pw)):.2e}")
    assert np.allclose(y_sep, y_pw, atol=1e-6)


def test_conv3d_kernel1_equals_dense():
    c = Conv3D(2, 3, 1, jax.random.PRNGKey(0))
    d = Dense(2, 3, jax.random.PRNGKey(0))
    d.W.value = c.W.value.reshape(2, 3)
    d.b.value = c.b.value
    xt = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    yc = np.asarray(c.forward(xt))
    yd = np.asarray(d.forward(xt.reshape(1, 4, 2))).reshape(1, 1, 2, 3)
    print(f"Conv3D(k=1) vs Dense: max diff {np.max(np.abs(yc - yd)):.2e}")
    assert np.allclose(yc, yd, atol=1e-6)


def test_depthwise_gradients_against_jax_grad():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(0.0, 1.0, (3, 3, 2)).astype(np.float32))
    dw = DepthwiseConv2D(2, 2, jax.random.PRNGKey(0))
    y = dw.forward(x)
    g = jnp.asarray(rng.normal(0.0, 1.0, y.shape).astype(np.float32)).astype(y.dtype)
    dx = np.asarray(dw.backward(g))

    def loss(xv):
        return jnp.sum(DepthwiseConv2D(2, 2, jax.random.PRNGKey(0)).forward(xv) * g)

    dx_ref = np.asarray(jax.grad(loss)(x))
    print(f"Depthwise dx max diff: {np.max(np.abs(dx - dx_ref)):.2e}")
    assert np.allclose(dx, dx_ref, atol=1e-4)


def test_conv3d_gradients_against_jax_grad():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(0.0, 1.0, (3, 3, 3, 2)).astype(np.float32))
    c = Conv3D(2, 4, 3, jax.random.PRNGKey(0))
    y = c.forward(x)
    g = jnp.asarray(rng.normal(0.0, 1.0, y.shape).astype(np.float32)).astype(y.dtype)
    dx = np.asarray(c.backward(g))
    assert c.W.grad.shape == c.W.value.shape
    assert c.b.grad.shape == c.b.value.shape

    def loss(xv):
        return jnp.sum(Conv3D(2, 4, 3, jax.random.PRNGKey(0)).forward(xv) * g)

    dx_ref = np.asarray(jax.grad(loss)(x))
    print(f"Conv3D dx max diff: {np.max(np.abs(dx - dx_ref)):.2e}")
    assert np.allclose(dx, dx_ref, atol=1e-4)


def test_separable_backward():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 4, 2)).astype(np.float32))
    sep = SeparableConv2D(2, 3, 3, jax.random.PRNGKey(0))
    y = sep.forward(x)
    g = jnp.asarray(rng.normal(0.0, 1.0, y.shape).astype(np.float32)).astype(y.dtype)
    dx = sep.backward(g)
    print(f"Separable dx shape: {dx.shape}")
    assert dx.shape == x.shape


def test_value_errors():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        Conv3D(1, 1, 0, key)
    with pytest.raises(ValueError):
        DepthwiseConv2D(1, 0, key)
    with pytest.raises(ValueError):
        SeparableConv2D(1, 1, 0, key)


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        Conv3D(1, 1, 3, jax.random.PRNGKey(0)).backward(jnp.ones((2, 2, 2, 1)))
    with pytest.raises(RuntimeError):
        DepthwiseConv2D(1, 3, jax.random.PRNGKey(0)).backward(jnp.ones((1, 1, 1)))
    with pytest.raises(RuntimeError):
        SeparableConv2D(1, 1, 3, jax.random.PRNGKey(0)).backward(jnp.ones((1, 1, 1)))


def test_repr_params():
    c = Conv3D(1, 1, 3, jax.random.PRNGKey(0))
    assert "Conv3D(1, 1, 3" in repr(c)
    assert len(c.params()) == 2
    dw = DepthwiseConv2D(2, 2, jax.random.PRNGKey(0))
    assert "DepthwiseConv2D(2, 2" in repr(dw)
    assert len(dw.params()) == 2
    sep = SeparableConv2D(2, 3, 2, jax.random.PRNGKey(0))
    assert "SeparableConv2D(2, 3, 2" in repr(sep)
    assert len(sep.params()) == 4
