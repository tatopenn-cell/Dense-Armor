"""Tests for the TensorFlow bridge: tensors, dtype, pytree."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from dense_armor.bridges.tensorflow import to_jax, to_tf


def test_tensor_round_trip_float32():
    t = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    a = to_jax(t)
    t2 = to_tf(a)
    assert t2.dtype == t.dtype
    assert tuple(t2.shape) == tuple(t.shape)
    assert bool(tf.reduce_all(t == t2))


def test_tensor_round_trip_int32():
    t = tf.constant([1, 2, 3], dtype=tf.int32)
    a = to_jax(t)
    t2 = to_tf(a)
    assert t2.dtype == t.dtype
    assert bool(tf.reduce_all(t == t2))


def test_tensor_round_trip_bool():
    t = tf.constant([True, False, True], dtype=tf.bool)
    a = to_jax(t)
    t2 = to_tf(a)
    assert t2.dtype == t.dtype
    assert bool(tf.reduce_all(t == t2))


def test_no_hidden_float64():
    t = tf.constant([1.0, 2.0], dtype=tf.float32)
    a = to_jax(t)
    assert a.dtype == jnp.float32


def test_non_contiguous():
    t = tf.transpose(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    a = to_jax(t)
    assert a.shape == (3, 2)
    np.testing.assert_array_equal(np.asarray(a), t.numpy())


def test_variable():
    v = tf.Variable([1.0, 2.0], dtype=tf.float32)
    a = to_jax(v)
    assert a.dtype == jnp.float32
    assert tuple(a.shape) == (2,)


def test_pytree_round_trip():
    tree = {
        "a": tf.constant([1.0, 2.0], dtype=tf.float32),
        "b": [tf.constant([3.0], dtype=tf.float32)],
    }
    a = to_jax(tree)
    assert isinstance(a, dict)
    assert isinstance(a["b"], list)
    assert a["a"].dtype == jnp.float32
    back = to_tf(a)
    assert isinstance(back, dict)
    assert bool(tf.reduce_all(back["a"] == tree["a"]))
    assert bool(tf.reduce_all(back["b"][0] == tree["b"][0]))


def test_non_tensor_passthrough():
    assert to_jax(3) == 3
    assert to_tf(3) == 3
