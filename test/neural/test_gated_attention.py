"""Tests for GatedCrossAttention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.gated_attention import GatedCrossAttention

KEY = jax.random.PRNGKey(0)


def _identity_model() -> MultiHeadCrossAttention:
    m = MultiHeadCrossAttention(2, 1, KEY, d_context=2)
    for p in m.params():
        p.value = jnp.eye(2)
    return m


def test_hand_case():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    signal = jnp.array([0.2, 0.9])
    out = g.forward(xq, xc, signal)
    expected_gate = np.array([0.0, 1.0])
    expected_y = np.array([[0.0, 0.0], [0.59889, 0.80222]])
    assert np.allclose(np.asarray(g.gate_), expected_gate)
    assert np.allclose(np.asarray(out), expected_y, atol=1e-5)

    dxq, dxc = g.backward(jnp.ones((2, 2)))
    expected_dxq = np.array([[0.0, 0.0], [0.11377, 0.05609]])
    expected_dxc = np.array(
        [[0.19778, 0.14168], [0.40111, 0.28735], [0.40111, 0.57097]]
    )
    assert np.allclose(np.asarray(dxq), expected_dxq, atol=1e-5)
    assert np.allclose(np.asarray(dxc), expected_dxc, atol=1e-5)


def test_boundary():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    g.forward(xq, xc, jnp.array([0.5, 0.5]))
    assert np.allclose(np.asarray(g.gate_), np.zeros(2))
    g.forward(xq, xc, jnp.array([float("nan"), 0.9]))
    assert np.allclose(np.asarray(g.gate_), np.array([0.0, 1.0]))


def test_all_open_equals_attention_alone():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.1)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    signal = jnp.array([0.9, 0.9])
    y_gated = np.asarray(g.forward(xq, xc, signal))
    m2 = _identity_model()
    y_plain = np.asarray(m2.forward(xq, xc))
    assert np.allclose(y_gated, y_plain, atol=1e-6)


def test_closed_step_no_grad():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    g.forward(xq, xc, jnp.array([0.0, 0.0]))
    for p in m.params():
        p.zero_grad()
    g.backward(jnp.ones((2, 2)))
    for p in m.params():
        assert np.allclose(np.asarray(p.grad), np.zeros_like(np.asarray(p.grad)))


def test_single_and_batch():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    signal = jnp.array([0.2, 0.9])
    y1 = np.asarray(g.forward(xq, xc, signal))
    m2 = _identity_model()
    g2 = GatedCrossAttention(m2, 0.5)
    y2 = np.asarray(g2.forward(xq[None], xc[None], signal[None])[0])
    assert np.allclose(y1, y2, atol=1e-6)


def test_errors():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.ones((2, 2))
    xc = jnp.ones((3, 2))
    with pytest.raises(ValueError):
        g.forward(xq, xc, jnp.ones(3))
    with pytest.raises(RuntimeError):
        g.backward(jnp.ones((2, 2)))


def test_repr():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    s = repr(g)
    assert "GatedCrossAttention" in s
    assert "threshold=0.5" in s


def test_repr_contains_threshold():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    assert "threshold=0.5" in repr(g)
    assert "MultiHeadCrossAttention" in repr(g)


def test_signal_batch_shape_mismatch():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.ones((2, 3, 2))
    xc = jnp.ones((2, 3, 2))
    with pytest.raises(ValueError):
        g.forward(xq, xc, jnp.ones((3, 3)))


def test_params_returns_inner_four():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    ps = g.params()
    assert ps == m.params()
    assert len(ps) == 4


def test_backward_batch():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
    xc = jnp.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    signal = jnp.array([[0.2, 0.9]])
    g.forward(xq, xc, signal)
    dxq, dxc = g.backward(jnp.ones((1, 2, 2)))
    assert dxq.shape == xq.shape
    assert dxc.shape == xc.shape


def test_residual_composition_preserves_static_part():
    m = _identity_model()
    g = GatedCrossAttention(m, 0.5)
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    signal = jnp.array([0.2, 0.9])
    y_attn = g.forward(xq, xc, signal)
    y = xq + y_attn
    assert np.allclose(np.asarray(y[0]), np.asarray(xq[0]))
    assert not np.allclose(np.asarray(y[1]), np.asarray(xq[1]))
    expected = np.array([0.59889, 1.80222])
    assert np.allclose(np.asarray(y[1]), expected, atol=1e-5)
