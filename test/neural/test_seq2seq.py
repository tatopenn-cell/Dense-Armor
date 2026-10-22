"""Tests for seq2seq.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.recurrent import LSTM
from dense_armor.utility.neural.seq2seq import Seq2Seq
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def _seq(horizon=2, reverse=True):
    return Seq2Seq(
        LSTM(2, 3, jax.random.PRNGKey(0)),
        LSTM(3, 3, jax.random.PRNGKey(1)),
        Dense(3, 1, jax.random.PRNGKey(2)),
        horizon=horizon,
        reverse_source=reverse,
    )


def test_composition_matches_manual():
    s = _seq()
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 2))
    out = s.forward(x)
    enc_in = jnp.flip(x, axis=0)
    enc_out = s.encoder.forward(enc_in, training=False)
    v = enc_out[-1]
    v_rep = jnp.repeat(v[None], 2, axis=0)
    dec_out = s.decoder.forward(v_rep, training=False)
    expected = s.head.forward(dec_out, training=False)
    assert np.allclose(np.asarray(out), np.asarray(expected))


def test_reverse_equals_forward_on_flipped():
    s_on = _seq(reverse=True)
    s_off = _seq(reverse=False)
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 2))
    x_flip = jnp.flip(x, axis=0)
    y_on = s_on.forward(x, training=False)
    y_off = s_off.forward(x_flip, training=False)
    assert np.allclose(np.asarray(y_on), np.asarray(y_off))


def test_shapes():
    s = _seq(horizon=2)
    assert s.forward(jnp.ones((4, 2))).shape == (2, 1)
    assert s.forward(jnp.ones((5, 4, 2))).shape == (5, 2, 1)


def test_gradcheck_x():
    s = _seq(horizon=2)
    x = jax.random.normal(jax.random.PRNGKey(5), (4, 2))

    def f(xx):
        s._cache = None
        return s.forward(xx)

    def b(xx, y, g):
        s._cache = None
        s.forward(xx)
        return s.backward(g)

    assert gradcheck(f, b, x) < 1e-3


def test_gradcheck_params():
    s = _seq(horizon=2)
    x = jax.random.normal(jax.random.PRNGKey(6), (4, 2))

    for i, p in enumerate(s.params()):
        p0 = p.value

        def f_w(w, idx=i, p0=p0):
            ps = s.params()
            ps[idx].value = w.reshape(p0.shape)
            s._cache = None
            return s.forward(x)

        def b_w(w, y, g, idx=i, p0=p0):
            ps = s.params()
            ps[idx].value = w.reshape(p0.shape)
            for q in s.params():
                q.grad = jnp.zeros_like(q.value)
            s._cache = None
            s.forward(x)
            s.backward(g)
            return s.params()[idx].grad.reshape(-1)

        assert gradcheck(f_w, b_w, p0.reshape(-1)) < 1e-2, f"param {i}"


@pytest.mark.training
def test_learning_reduces_error():
    rng = np.random.default_rng(0)
    n = 40
    T = 8
    H = 4
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    t_src = np.arange(T, dtype=np.float32)
    t_tgt = np.arange(T, T + H, dtype=np.float32)
    X = np.sin(t_src[None, :, None] + phase[:, None, None]) + 0.05 * rng.normal(
        0.0, 1.0, (n, T, 1)
    ).astype(np.float32)
    Y = np.sin(t_tgt[None, :, None] + phase[:, None, None])
    Y = Y.astype(np.float32)

    s = Seq2Seq(
        LSTM(1, 4, jax.random.PRNGKey(0)),
        LSTM(4, 4, jax.random.PRNGKey(1)),
        Dense(4, 1, jax.random.PRNGKey(2)),
        horizon=H,
    )
    before = float(
        np.mean(
            np.asarray((np.asarray(s.forward(jnp.asarray(X), training=False)) - Y) ** 2)
        )
    )

    t = Trainer(s, MSE(), Adam(0.01))
    t.fit(X, Y, epochs=20, batch_size=10, key=jax.random.PRNGKey(0))
    after = float(
        np.mean(
            np.asarray((np.asarray(s.forward(jnp.asarray(X), training=False)) - Y) ** 2)
        )
    )
    print(f"before: {before:.4f}, after: {after:.4f}")
    assert after < before / 5.0


def test_value_errors_and_runtime_error():
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        Seq2Seq(
            LSTM(2, 3, jax.random.PRNGKey(0)),
            LSTM(3, 3, jax.random.PRNGKey(1)),
            Dense(3, 1, jax.random.PRNGKey(2)),
            horizon=0,
        )
    s = _seq()
    with pytest.raises(RuntimeError, match="call forward"):
        s.backward(jnp.ones((2, 1)))


def test_repr():
    s = _seq()
    r = repr(s)
    assert "Seq2Seq" in r
    assert "horizon=2" in r
    assert "reverse_source=True" in r


def test_backward_no_reverse():
    """Cover the else branch (line 133) with reverse_source=False."""
    s = _seq(reverse=False)
    x = jax.random.normal(jax.random.PRNGKey(7), (4, 2))
    y = s.forward(x)
    dx = s.backward(jnp.ones_like(y))
    assert dx.shape == x.shape


def test_backward_no_reverse_batch():
    s = _seq(reverse=False)
    x = jax.random.normal(jax.random.PRNGKey(8), (3, 4, 2))
    y = s.forward(x)
    dx = s.backward(jnp.ones_like(y))
    assert dx.shape == x.shape
