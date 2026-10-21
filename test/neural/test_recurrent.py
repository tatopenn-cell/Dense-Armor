"""Tests for recurrent.py: RNN, GRU, LSTM."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.recurrent import GRU, LSTM, RNN


def _x():
    return jax.random.normal(jax.random.PRNGKey(1), (5, 3))


def _gradcheck_x(layer):
    x = _x()

    def f(xx):
        layer._cache = []
        return layer.forward(xx)

    def b(xx, y, g):
        layer._cache = []
        layer.forward(xx)
        return layer.backward(g)

    return float(gradcheck(f, b, x))


def test_rnn_x_gradcheck():
    assert _gradcheck_x(RNN(3, 4, jax.random.PRNGKey(0))) < 1e-4


def test_gru_x_gradcheck():
    assert _gradcheck_x(GRU(3, 4, jax.random.PRNGKey(0))) < 1e-4


def test_lstm_peepholes_x_gradcheck():
    assert _gradcheck_x(LSTM(3, 4, jax.random.PRNGKey(0), peepholes=True)) < 1e-4


def test_lstm_no_peepholes_x_gradcheck():
    assert _gradcheck_x(LSTM(3, 4, jax.random.PRNGKey(0), peepholes=False)) < 1e-4


def _gradcheck_param(layer, pname):
    x = _x()
    p = getattr(layer, pname)
    p0 = p.value

    def f(w):
        q = getattr(layer, pname)
        q.value = w.reshape(p0.shape)
        layer._cache = []
        try:
            return layer.forward(x)
        finally:
            q.value = p0

    def b(w, y, g):
        q = getattr(layer, pname)
        q.value = w.reshape(p0.shape)
        for pp in layer.params():
            pp.grad = jnp.zeros_like(pp.value)
        layer._cache = []
        try:
            layer.forward(x)
            layer.backward(g)
            return q.grad.reshape(-1)
        finally:
            q.value = p0

    return float(gradcheck(f, b, p0.reshape(-1)))


def test_rnn_param_gradcheck():
    r = RNN(3, 4, jax.random.PRNGKey(0))
    for name in ("W", "U", "b"):
        assert _gradcheck_param(r, name) < 1e-4, name


def test_gru_param_gradcheck():
    g = GRU(3, 4, jax.random.PRNGKey(0))
    for name in ("W_r", "W_z", "W", "U_r", "U_z", "U", "b_r", "b_z", "b"):
        assert _gradcheck_param(g, name) < 1e-4, name


def test_lstm_param_gradcheck():
    ls = LSTM(3, 4, jax.random.PRNGKey(0), peepholes=True)
    for name in (
        "W_z",
        "W_i",
        "W_f",
        "W_o",
        "R_z",
        "R_i",
        "R_f",
        "R_o",
        "b_z",
        "b_i",
        "b_f",
        "b_o",
        "p_i",
        "p_f",
        "p_o",
    ):
        assert _gradcheck_param(ls, name) < 1e-4, name


def test_rnn_step_equals_forward():
    r = RNN(3, 4, jax.random.PRNGKey(0))
    seq = jax.random.normal(jax.random.PRNGKey(2), (5, 3))
    yf = r.forward(seq)
    h = r.init_state()
    ys = []
    for t in range(5):
        yt, h = r.step(seq[t], h)
        ys.append(yt)
    assert np.allclose(np.asarray(jnp.stack(ys)), np.asarray(yf), atol=1e-6)


def test_gru_step_equals_forward():
    g = GRU(3, 4, jax.random.PRNGKey(0))
    seq = jax.random.normal(jax.random.PRNGKey(2), (5, 3))
    yf = g.forward(seq)
    h = g.init_state()
    ys = []
    for t in range(5):
        yt, h = g.step(seq[t], h)
        ys.append(yt)
    assert np.allclose(np.asarray(jnp.stack(ys)), np.asarray(yf), atol=1e-6)


def test_lstm_step_equals_forward():
    ls = LSTM(3, 4, jax.random.PRNGKey(0))
    seq = jax.random.normal(jax.random.PRNGKey(2), (5, 3))
    yf = ls.forward(seq)
    y, c = ls.init_state()
    ys = []
    for t in range(5):
        yt, (y, c) = ls.step(seq[t], (y, c))
        ys.append(yt)
    assert np.allclose(np.asarray(jnp.stack(ys)), np.asarray(yf), atol=1e-6)


def test_rnn_batch_equals_single():
    r = RNN(3, 4, jax.random.PRNGKey(0))
    xb = jax.random.normal(jax.random.PRNGKey(2), (4, 5, 3))
    yb = r.forward(xb)
    y1 = r.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_gru_batch_equals_single():
    g = GRU(3, 4, jax.random.PRNGKey(0))
    xb = jax.random.normal(jax.random.PRNGKey(2), (4, 5, 3))
    yb = g.forward(xb)
    y1 = g.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_lstm_batch_equals_single():
    ls = LSTM(3, 4, jax.random.PRNGKey(0))
    xb = jax.random.normal(jax.random.PRNGKey(2), (4, 5, 3))
    yb = ls.forward(xb)
    y1 = ls.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_gru_grad_accumulates():
    g = GRU(3, 4, jax.random.PRNGKey(0))
    x = _x()
    g.forward(x)
    g.backward(jnp.ones((5, 4)))
    first = np.asarray(g.W.grad)
    g.forward(x)
    g.backward(jnp.ones((5, 4)))
    second = np.asarray(g.W.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)


def test_lstm_grad_accumulates():
    ls = LSTM(3, 4, jax.random.PRNGKey(0))
    x = _x()
    ls.forward(x)
    ls.backward(jnp.ones((5, 4)))
    first = np.asarray(ls.W_z.grad)
    ls.forward(x)
    ls.backward(jnp.ones((5, 4)))
    second = np.asarray(ls.W_z.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)


def test_rnn_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        RNN(3, 4, jax.random.PRNGKey(0)).backward(jnp.ones((5, 4)))


def test_gru_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        GRU(3, 4, jax.random.PRNGKey(0)).backward(jnp.ones((5, 4)))


def test_lstm_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        LSTM(3, 4, jax.random.PRNGKey(0)).backward(jnp.ones((5, 4)))


def test_rnn_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation must be"):
        RNN(3, 4, jax.random.PRNGKey(0), activation="sigmoid")


def test_lstm_peepholes_off_params_count():
    ls_on = LSTM(3, 4, jax.random.PRNGKey(0), peepholes=True)
    ls_off = LSTM(3, 4, jax.random.PRNGKey(0), peepholes=False)
    assert len(ls_on.params()) == 15
    assert len(ls_off.params()) == 12
    names = [getattr(p, "value", None) for p in ls_off.params()]
    assert all(v is not None for v in names)


def test_gru_one_step_by_hand():
    g = GRU(1, 1, jax.random.PRNGKey(0))
    g.W_r.value = jnp.array([[1.0]])
    g.U_r.value = jnp.array([[0.0]])
    g.b_r.value = jnp.array([0.0])
    g.W_z.value = jnp.array([[1.0]])
    g.U_z.value = jnp.array([[0.0]])
    g.b_z.value = jnp.array([0.0])
    g.W.value = jnp.array([[2.0]])
    g.U.value = jnp.array([[1.0]])
    g.b.value = jnp.array([0.0])

    x_t = jnp.array([1.0])
    h_prev = jnp.array([0.5])
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))

    r_pre = 1.0 * 1.0 + 0.0 * 0.5 + 0.0
    z_pre = 1.0 * 1.0 + 0.0 * 0.5 + 0.0
    r = sig(r_pre)
    z = sig(z_pre)
    h_tilde_pre = 2.0 * 1.0 + 1.0 * (r * 0.5) + 0.0
    h_tilde = np.tanh(h_tilde_pre)
    h = z * 0.5 + (1.0 - z) * h_tilde
    y, _ = g.step(x_t, h_prev)
    assert abs(float(y[0]) - float(h)) < 1e-6


def test_lstm_one_step_by_hand():
    ls = LSTM(1, 1, jax.random.PRNGKey(0), peepholes=True)
    for n in (
        "W_z",
        "W_i",
        "W_f",
        "W_o",
        "R_z",
        "R_i",
        "R_f",
        "R_o",
        "b_z",
        "b_i",
        "b_f",
        "b_o",
        "p_i",
        "p_f",
        "p_o",
    ):
        setattr(getattr(ls, n), "value", jnp.zeros_like(getattr(ls, n).value))
    ls.W_z.value = jnp.array([[1.0]])
    ls.W_i.value = jnp.array([[1.0]])
    ls.W_f.value = jnp.array([[1.0]])
    ls.W_o.value = jnp.array([[1.0]])

    x_t = jnp.array([0.5])
    y_prev = jnp.array([0.1])
    c_prev = jnp.array([0.2])
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))

    z = np.tanh(1.0 * 0.5)
    i = sig(1.0 * 0.5)
    f = sig(1.0 * 0.5)
    c = z * i + 0.2 * f
    o = sig(1.0 * 0.5)
    y = np.tanh(c) * o

    out, _ = ls.step(x_t, (y_prev, c_prev))
    assert abs(float(out[0]) - float(y)) < 1e-6
