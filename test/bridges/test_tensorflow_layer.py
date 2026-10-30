"""Tests for KerasLayer: a Dense-Armor model as tf.keras layer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from dense_armor.bridges.tensorflow import KerasLayer
from dense_armor.utility.neural import Dense, Sequential
from dense_armor.utility.neural.activations import Relu


def _model(d_in=4, d_h=8, d_out=3):
    return Sequential(
        [
            Dense(d_in, d_h, key=jax.random.PRNGKey(0)),
            Relu(),
            Dense(d_h, d_out, key=jax.random.PRNGKey(1)),
        ]
    )


def test_forward_matches_jax():
    m = _model()
    layer = KerasLayer(m)
    x = tf.random.normal((2, 4))
    y_t = layer(x)
    y_j = m.forward(jnp.asarray(x.numpy()), training=False)
    np.testing.assert_allclose(y_t.numpy(), np.asarray(y_j), rtol=1e-5, atol=1e-6)


def test_vars_are_trainable():
    m = _model()
    layer = KerasLayer(m)
    assert len(layer.trainable_variables) == 4
    for v in layer.trainable_variables:
        assert hasattr(v, "trainable") and v.trainable


def test_grad_on_input_matches_jax():
    m = _model()
    layer = KerasLayer(m)
    x = tf.Variable(tf.random.normal((2, 4)))
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_sum(y)
    g_t = tape.gradient(loss, x).numpy()

    x_j = jnp.asarray(x.numpy())
    g_j = jax.grad(lambda z: m.forward(z, training=False).sum())(x_j)
    np.testing.assert_allclose(g_t, np.asarray(g_j), rtol=1e-4, atol=1e-5)


def test_optimizer_step_lowers_loss():
    m = _model()
    layer = KerasLayer(m)
    x = tf.random.normal((16, 4))
    target = tf.random.normal((16, 3))
    opt = tf.keras.optimizers.SGD(0.1)
    losses = []
    for _ in range(10):
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean((y - target) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        opt.apply_gradients(zip(grads, layer.trainable_variables))
        losses.append(float(loss.numpy()))
    assert losses[-1] < losses[0] * 0.7
