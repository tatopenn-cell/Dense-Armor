"""Tests for RMSNorm, the GLU feed-forward layers and the rotary embedding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.attention import MultiHeadSelfAttention
from dense_armor.utility.neural.glu import GLUFeedForward
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.norm_rms import RMSNorm
from dense_armor.utility.neural.rotary import (
    apply_rotary,
    apply_rotary_transpose,
    rotary_angles,
)


def _grad_errors(layer, x):
    """Relative error of ``backward`` on the input and on every ``Param``."""

    def f(xx):
        return layer.forward(xx)

    def b(xx, y, g):
        layer.forward(xx)
        for p in layer.params():
            p.zero_grad()
        return layer.backward(g)

    errors = {"x": gradcheck(f, b, x)}
    for i, p in enumerate(layer.params()):
        w0 = p.value

        def fp(w, p=p, w0=w0):
            p.value = w.reshape(w0.shape)
            try:
                return layer.forward(x)
            finally:
                p.value = w0

        def bp(w, y, g, p=p, w0=w0):
            p.value = w.reshape(w0.shape)
            for q in layer.params():
                q.zero_grad()
            try:
                layer.forward(x)
                layer.backward(g)
                return p.grad.reshape(-1)
            finally:
                p.value = w0

        errors[i] = gradcheck(fp, bp, w0.reshape(-1))
    return errors


def test_rmsnorm_hand_case():
    n = RMSNorm(2, eps=1e-12)
    n.gain.value = jnp.array([1.0, 2.0])
    y = np.asarray(n.forward(jnp.array([3.0, 4.0])))
    np.testing.assert_allclose(y, [0.8485281374, 2.2627416998], rtol=1e-5)


def test_rmsnorm_does_not_subtract_the_mean():
    n = RMSNorm(3, eps=1e-12)
    y = np.asarray(n.forward(jnp.array([1.0, 1.0, 1.0])))
    np.testing.assert_allclose(y, [1.0, 1.0, 1.0], rtol=1e-5)


def test_rmsnorm_is_invariant_to_a_positive_rescaling_of_the_input():
    n = RMSNorm(5, eps=1e-12)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5))
    np.testing.assert_allclose(
        np.asarray(n.forward(x)), np.asarray(n.forward(7.5 * x)), atol=1e-5
    )


def test_rmsnorm_gradcheck():
    n = RMSNorm(6)
    n.gain.value = jax.random.normal(jax.random.PRNGKey(3), (6,))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 6))
    errors = _grad_errors(n, x)
    assert max(errors.values()) < 1e-4, errors


def test_rmsnorm_batch_equals_single_sample():
    n = RMSNorm(6)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 6))
    batch = np.asarray(n.forward(x))
    single = np.stack([np.asarray(n.forward(x[i])) for i in range(4)])
    np.testing.assert_allclose(batch, single, atol=1e-6)


def test_rmsnorm_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        RMSNorm(3).backward(jnp.ones(3))


GLU_CASES = [
    ("sigmoid", [0.3655292893, 0.8807970780]),
    ("gelu", [0.4206723730, 1.9544997361]),
    ("silu", [0.3655292893, 1.7615941560]),
]


@pytest.mark.parametrize("activation,expected", GLU_CASES)
def test_glu_hand_case(activation, expected):
    f = GLUFeedForward(2, 2, activation, jax.random.PRNGKey(0))
    f.W.value = jnp.eye(2)
    f.V.value = 0.5 * jnp.eye(2)
    f.W2.value = jnp.eye(2)
    y = np.asarray(f.forward(jnp.array([1.0, 2.0])))
    np.testing.assert_allclose(y, expected, rtol=1e-5)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ["sigmoid", "gelu", "silu"])
def test_glu_gradcheck(activation, bias):
    f = GLUFeedForward(4, 6, activation, jax.random.PRNGKey(0), bias=bias)
    if bias:
        keys = jax.random.split(jax.random.PRNGKey(5), 3)
        f.b.value = 0.3 * jax.random.normal(keys[0], f.b.value.shape)
        f.c.value = 0.3 * jax.random.normal(keys[1], f.c.value.shape)
        f.b2.value = 0.3 * jax.random.normal(keys[2], f.b2.value.shape)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    errors = _grad_errors(f, x)
    assert max(errors.values()) < 1e-4, errors


def test_glu_gate_changes_the_output():
    f = GLUFeedForward(2, 2, "sigmoid", jax.random.PRNGKey(0))
    f.W.value = jnp.eye(2)
    f.V.value = jnp.eye(2)
    f.W2.value = jnp.eye(2)
    x = jnp.array([2.0, -3.0])
    gated = np.asarray(f.forward(x))
    ungated = 1.0 / (1.0 + np.exp(-np.asarray(x)))
    assert np.max(np.abs(gated - ungated)) > 0.5


def test_glu_has_three_matrices_and_three_more_with_bias():
    assert len(GLUFeedForward(2, 3, "silu", jax.random.PRNGKey(0)).params()) == 3
    assert (
        len(GLUFeedForward(2, 3, "silu", jax.random.PRNGKey(0), bias=True).params())
        == 6
    )


def test_glu_batch_equals_single_sample_and_sequence():
    f = GLUFeedForward(4, 6, "silu", jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4))
    full = np.asarray(f.forward(x))
    rows = np.stack([np.asarray(f.forward(x[0, t])) for t in range(5)])
    np.testing.assert_allclose(full[0], rows, atol=1e-6)


def test_glu_repr_names_the_activation_and_the_bias():
    f = GLUFeedForward(2, 3, "silu", jax.random.PRNGKey(0), bias=True)
    assert repr(f) == "GLUFeedForward(2, 3, 'silu', bias=True)"


def test_glu_unknown_activation_raises():
    with pytest.raises(ValueError):
        GLUFeedForward(2, 2, "relu", jax.random.PRNGKey(0))


def test_glu_backward_before_forward_raises():
    f = GLUFeedForward(2, 2, "silu", jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError):
        f.backward(jnp.ones(2))


def _dot_at(q, k, m, n, d_k):
    length = max(m, n) + 1
    cos, sin = rotary_angles(length, d_k)
    rq = apply_rotary(jnp.tile(q, (length, 1)), cos, sin)[m]
    rk = apply_rotary(jnp.tile(k, (length, 1)), cos, sin)[n]
    return float(rq @ rk)


def test_rotary_dot_product_depends_only_on_the_relative_position():
    q = jax.random.normal(jax.random.PRNGKey(0), (8,))
    k = jax.random.normal(jax.random.PRNGKey(1), (8,))
    a = _dot_at(q, k, 5, 2, 8)
    b = _dot_at(q, k, 9, 6, 8)
    c = _dot_at(q, k, 5, 3, 8)
    assert abs(a - b) < 1e-4
    assert abs(a - c) > 1e-2


def test_rotary_keeps_the_norm_and_leaves_position_zero_alone():
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 8))
    cos, sin = rotary_angles(6, 8)
    y = apply_rotary(x, cos, sin)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(y), axis=-1),
        np.linalg.norm(np.asarray(x), axis=-1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(y[0]), np.asarray(x[0]), atol=1e-6)


def test_rotary_hand_case_and_inverse():
    cos, sin = rotary_angles(2, 2)
    x = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    y = apply_rotary(x, cos, sin)
    np.testing.assert_allclose(np.asarray(y[1]), [0.5403023, 0.8414710], atol=1e-5)
    back = apply_rotary_transpose(y, cos, sin)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), atol=1e-5)


def test_rotary_angles_with_an_offset_are_the_later_rows():
    cos, sin = rotary_angles(6, 8)
    cos_o, sin_o = rotary_angles(2, 8, offset=4)
    np.testing.assert_allclose(np.asarray(cos_o), np.asarray(cos[4:6]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin_o), np.asarray(sin[4:6]), atol=1e-6)


def test_rotary_odd_dimension_raises():
    with pytest.raises(ValueError):
        rotary_angles(3, 5)
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(6, 2, jax.random.PRNGKey(0), rotary=True)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_with_rotary_gradcheck(causal):
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=causal, rotary=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    errors = _grad_errors(m, x)
    assert max(errors.values()) < 1e-4, errors


def test_attention_rotary_batch_equals_single_sample():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), rotary=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    batch = np.asarray(m.forward(x))
    single = np.stack([np.asarray(m.forward(x[i])) for i in range(2)])
    np.testing.assert_allclose(batch, single, atol=1e-5)


def test_attention_rotary_causal_step_equals_forward():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=True, rotary=True)
    seq = jax.random.normal(jax.random.PRNGKey(2), (6, 8))
    yf = m.forward(seq)
    cache = m.init_cache()
    rows = []
    for t in range(6):
        yt, cache = m.step(seq[t], cache)
        rows.append(yt)
    np.testing.assert_allclose(np.asarray(jnp.stack(rows)), np.asarray(yf), atol=1e-5)


def test_attention_rotary_changes_the_output():
    plain = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
    rotated = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), rotary=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    assert float(jnp.max(jnp.abs(plain.forward(x) - rotated.forward(x)))) > 1e-2


def test_attention_rotary_sees_the_order_of_the_tokens():
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), rotary=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    y = np.asarray(m.forward(x))
    y_rev = np.asarray(m.forward(x[::-1]))
    assert np.max(np.abs(y[0] - y_rev[-1])) > 1e-3


MASTER_VALUES = {
    (False, False): dict(
        y0=[
            -1.7638189792633057,
            -1.9617462158203125,
            -0.5266642570495605,
            0.9045798778533936,
        ],
        y_sum=-17.09138298034668,
        dx_sum=-4.9654541015625,
        dwq_sum=-1.729050636291504,
        dwo_sum=6.995484828948975,
    ),
    (False, True): dict(
        y0=[
            -1.200319528579712,
            -2.664421558380127,
            -0.42142587900161743,
            1.3955621719360352,
        ],
        y_sum=-10.545769691467285,
        dx_sum=-8.786205291748047,
        dwq_sum=-0.8111534118652344,
        dwo_sum=16.888275146484375,
    ),
    (True, False): dict(
        y0=[
            0.10256822689852667,
            -0.2194263372602911,
            -0.5140958780714502,
            0.6594305811594109,
        ],
        y_sum=9.573179532850343,
        dx_sum=17.446162950486997,
        dwq_sum=2.0515880775398134,
        dwo_sum=-15.2268447859072,
    ),
    (True, True): dict(
        y0=[
            0.256219575767904,
            -0.19267658562852993,
            0.17228992559933917,
            0.4888856620442388,
        ],
        y_sum=17.956369253757554,
        dx_sum=14.807631304469496,
        dwq_sum=2.511660563950069,
        dwo_sum=23.38741095408711,
    ),
}


@pytest.mark.parametrize("causal", [False, True])
def test_attention_default_equals_the_values_stored_from_master(causal):
    """The values were taken from master before rotary existed.

    The key of ``MASTER_VALUES`` is ``(jax_enable_x64, causal)``. The last
    digit of a loss changes with the version of Python and of the libraries,
    so the comparison is relative: 1e-9 with x64 on, 1e-5 with x64 off.
    """
    m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=causal)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (2, 5, 8))
    y = m.forward(x)
    dx = m.backward(g)
    x64 = bool(jax.config.jax_enable_x64)
    ref = MASTER_VALUES[(x64, causal)]
    tol = 1e-9 if x64 else 1e-5
    np.testing.assert_allclose(np.asarray(y[0, 0, :4]), ref["y0"], rtol=tol)
    np.testing.assert_allclose(float(y.sum()), ref["y_sum"], rtol=tol)
    np.testing.assert_allclose(float(dx.sum()), ref["dx_sum"], rtol=tol)
    np.testing.assert_allclose(float(m.W_Q.grad.sum()), ref["dwq_sum"], rtol=tol)
    np.testing.assert_allclose(float(m.W_O.grad.sum()), ref["dwo_sum"], rtol=tol)
    assert repr(m) == f"MultiHeadSelfAttention(8, 2, causal={causal})"
    assert m.rotary is False
