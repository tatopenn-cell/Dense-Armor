"""Tests for autoencoder.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.activations import Tanh
from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def _hand_autoencoder():
    enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    enc.layers[0].W.value = jnp.array([[1.0], [0.0]])
    enc.layers[0].b.value = jnp.array([0.0])
    dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
    dec.layers[0].W.value = jnp.array([[1.0, 0.0]])
    dec.layers[0].b.value = jnp.array([0.0, 0.0])
    return Autoencoder(enc, dec)


def test_composition_matches_encoder_then_decoder():
    enc = Sequential([Dense(3, 2, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(2, 3, jax.random.PRNGKey(1))])
    a = Autoencoder(enc, dec)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
    z = enc.forward(x, training=False)
    expected = dec.forward(z, training=False)
    assert np.allclose(np.asarray(a.forward(x, training=False)), np.asarray(expected))


def test_encode_decode_equals_forward_no_training():
    a = _hand_autoencoder()
    x = jnp.array([3.0, 4.0])
    z = a.encode(x)
    rec = a.decode(z)
    assert np.allclose(np.asarray(rec), np.asarray(a.forward(x, training=False)))


def test_hand_case():
    a = _hand_autoencoder()
    x = jnp.array([3.0, 4.0])
    assert np.allclose(np.asarray(a.forward(x, training=False)), np.array([3.0, 0.0]))
    assert abs(float(a.reconstruction_error(x)) - 8.0) < 1e-6

    xb = jnp.array([[3.0, 4.0], [1.0, 0.0]])
    err = np.asarray(a.reconstruction_error(xb))
    assert np.allclose(err, np.array([8.0, 0.0]))


def test_gradcheck_on_x_and_params():
    enc = Sequential([Dense(3, 2, jax.random.PRNGKey(0)), Tanh()])
    dec = Sequential([Dense(2, 3, jax.random.PRNGKey(1)), Tanh()])
    a = Autoencoder(enc, dec)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 3))

    def f(xx):
        a.forward(xx)
        return a.forward(xx)

    def b(xx, y, g):
        a.forward(xx)
        return a.backward(g)

    assert gradcheck(f, b, x) < 1e-3

    # every Param
    for i, p in enumerate(a.params()):
        p0 = p.value

        def f_w(w, idx=i, p0=p0):
            ps = a.params()
            ps[idx].value = w.reshape(p0.shape)
            return a.forward(x)

        def b_w(w, y, g, idx=i, p0=p0):
            ps = a.params()
            ps[idx].value = w.reshape(p0.shape)
            for q in a.params():
                q.grad = jnp.zeros_like(q.value)
            a.forward(x)
            a.backward(g)
            return a.params()[idx].grad.reshape(-1)

        assert gradcheck(f_w, b_w, p0.reshape(-1)) < 1e-3, f"param {i}"


@pytest.mark.training
def test_learning_reduces_error():
    rng = np.random.default_rng(0)
    A = rng.normal(0.0, 1.0, (2, 6)).astype(np.float32)
    Z = rng.normal(0.0, 1.0, (200, 2)).astype(np.float32)
    X = Z @ A

    enc = Sequential([Dense(6, 2, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(2, 6, jax.random.PRNGKey(1))])
    a = Autoencoder(enc, dec)
    before = float(np.mean(np.asarray(a.reconstruction_error(X))))

    t = Trainer(a, MSE(), Adam(0.01))
    t.fit(X, X, epochs=50, batch_size=20, key=jax.random.PRNGKey(0))
    after = float(np.mean(np.asarray(a.reconstruction_error(X))))
    print(f"before: {before:.4f}, after: {after:.4f}")
    assert after < before / 10.0


def test_as_transformer_wraps_the_encoder():
    enc = Sequential([Dense(3, 2, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(2, 3, jax.random.PRNGKey(1))])
    a = AutoTransformer_with_encoder(enc, dec)
    at = AsTransformer(a.encoder, ["a", "b", "c"])
    out = at.transform_one({"a": 1.0, "b": 0.0, "c": 0.0})
    assert set(out) == {"z0", "z1"}


def AutoTransformer_with_encoder(enc, dec):
    return Autoencoder(enc, dec)


def test_repr():
    a = _hand_autoencoder()
    r = repr(a)
    assert "Autoencoder" in r
