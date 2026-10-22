"""Tests for quantize.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.quantize import VectorQuantizer
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def _hand():
    vq = VectorQuantizer(2, 1, jax.random.PRNGKey(0), beta=0.25)
    vq.codebook.value = jnp.array([[0.0], [1.0]])
    vq.codebook.grad = jnp.zeros_like(vq.codebook.value)
    return vq


def test_hand_case():
    vq = _hand()
    z_e = jnp.array([[0.2], [0.9]])
    z_q = vq.forward(z_e)
    assert vq.indices_.tolist() == [0, 1]
    assert np.allclose(np.asarray(z_q), np.array([[0.0], [1.0]]))
    assert abs(float(vq.loss()) - 0.03125) < 1e-6

    g = jnp.array([[1.0], [1.0]])
    dz = np.asarray(vq.backward(g))
    assert np.allclose(dz, np.array([[1.05], [0.975]]), atol=1e-6)
    assert np.allclose(
        np.asarray(vq.codebook.grad), np.array([[-0.2], [0.1]]), atol=1e-6
    )


def test_tie_goes_to_first():
    vq = _hand()
    vq.forward(jnp.array([[0.5]]))
    assert vq.indices_.tolist() == [0]


def test_beta_zero_returns_g():
    vq = _hand()
    vq.beta = 0.0
    vq.forward(jnp.array([[0.2]]))
    g = jnp.array([[1.0]])
    assert np.allclose(np.asarray(vq.backward(g)), np.asarray(g))


def test_codebook_grad_against_jax_grad():
    vq = _hand()
    z_e = jnp.array([[0.2], [0.9]])
    vq.forward(z_e)
    vq.backward(jnp.array([[1.0], [1.0]]))
    codebook_grad = np.asarray(vq.codebook.grad)

    def obj(codebook):
        diff = z_e - codebook[jnp.array([0, 1])]
        return jnp.mean(jnp.sum(diff**2, axis=-1))

    grad = jax.grad(obj)(vq.codebook.value)
    assert np.allclose(codebook_grad, np.asarray(grad), atol=1e-6)


def test_encode_decode_indices_roundtrip():
    vq = _hand()
    z_e = jnp.array([[0.2], [0.9]])
    idx = vq.encode_indices(z_e)
    codes = vq.decode_indices(idx)
    fwd = vq.forward(z_e)
    assert np.allclose(np.asarray(codes), np.asarray(fwd))


def test_one_sample_equals_batch_row():
    vq = _hand()
    zb = jnp.array([[0.1], [0.6], [0.9]])
    qb = vq.forward(zb)
    idx_b = vq.indices_.tolist()
    q1 = vq.forward(zb[1])
    idx_1 = vq.indices_.tolist()
    assert idx_b[1] == idx_1[0]
    assert np.allclose(np.asarray(q1[0]), np.asarray(qb[1]))


@pytest.mark.training
def test_learning_reduces_reconstruction_error():
    rng = np.random.default_rng(0)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    pts = []
    for c in centers:
        pts.append(c + rng.normal(0.0, 0.2, (100, 2)).astype(np.float32))
    X = np.vstack(pts)

    enc = Sequential([Dense(2, 2, jax.random.PRNGKey(0))])
    vq = VectorQuantizer(3, 2, jax.random.PRNGKey(1), beta=0.25)

    def model_forward(enc, vq, x):
        return vq.forward(enc.forward(x, training=True))

    class AE:
        def __init__(self):
            self.enc = enc
            self.vq = vq

        def forward(self, x, training=True):
            return self.vq.forward(self.enc.forward(x, training=training))

        def backward(self, g):
            return self.enc.backward(self.vq.backward(g))

        def params(self):
            return list(self.enc.params()) + list(self.vq.params())

    a = AE()

    def err():
        return float(
            np.mean(
                np.asarray(
                    (
                        (X - np.asarray(a.forward(jnp.asarray(X), training=False))) ** 2
                    ).sum(-1)
                )
            )
        )

    before = err()
    t = Trainer(a, MSE(), Adam(0.01))
    t.fit(X, X, epochs=100, batch_size=20, key=jax.random.PRNGKey(0))
    after = err()
    print(f"before: {before:.4f}, after: {after:.4f}")
    print(
        f"codes used: {np.unique(np.asarray(vq.encode_indices(enc.forward(jnp.asarray(X), training=False)))).tolist()}"
    )
    assert after < before / 2.0


def test_value_errors_and_runtime_error():
    with pytest.raises(ValueError, match="n_codes must be"):
        VectorQuantizer(0, 1, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="dim must be"):
        VectorQuantizer(1, 0, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="beta must be non-negative"):
        VectorQuantizer(1, 1, jax.random.PRNGKey(0), beta=-0.1)

    vq = VectorQuantizer(2, 1, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        vq.backward(jnp.ones((1, 1)))
    with pytest.raises(RuntimeError, match="call forward"):
        vq.loss()
