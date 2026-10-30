"""Tests for from_keras: a Keras model inside Dense-Armor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from dense_armor.bridges.tensorflow import from_keras


def _model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )


def test_apply_matches_keras():
    tf.random.set_seed(0)
    m = _model()
    adapter = from_keras(m)
    x = tf.random.normal((3, 4))
    y_a = adapter.apply(x)
    y_d = m(x)
    np.testing.assert_allclose(y_a.numpy(), y_d.numpy(), rtol=1e-6, atol=1e-7)


def test_apply_on_jax_array():
    tf.random.set_seed(0)
    m = _model()
    adapter = from_keras(m)
    x = jnp.zeros((3, 4), dtype=jnp.float32)
    y = adapter.apply(x)
    assert tuple(y.shape) == (3, 2)


def test_grad_flows_from_jax_through_keras():
    tf.random.set_seed(0)
    m = _model()
    adapter = from_keras(m)
    x_j = jnp.ones((3, 4), dtype=jnp.float32)

    def loss(z):
        return adapter.apply(z).sum()

    g_jax = jax.grad(loss)(x_j)

    x_t = tf.Variable(x_j)
    with tf.GradientTape() as tape:
        y = m(x_t)
        loss = tf.reduce_sum(y)
    g_tf = tape.gradient(loss, x_t).numpy()

    np.testing.assert_allclose(np.asarray(g_jax), g_tf, rtol=1e-5, atol=1e-6)
