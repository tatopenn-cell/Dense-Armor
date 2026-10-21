"""Targeted tests for the lines the main test files do not reach."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.layers import (
    Dense,
    Dropout,
    Embedding,
    Residual,
)
from dense_armor.utility.neural.norm import BatchNorm, LayerNorm


def test_dense_xavier_init_and_bad_init():
    d = Dense(3, 2, jax.random.PRNGKey(0), init="xavier")
    assert d.W.value.shape == (3, 2)
    with pytest.raises(ValueError, match="init must be"):
        Dense(3, 2, jax.random.PRNGKey(0), init="uniform")


def test_dense_eq_with_other_type_returns_not_implemented():
    d = Dense(3, 2, jax.random.PRNGKey(0))
    assert d.__eq__("not a dense") is NotImplemented
    assert not (d == "not a dense")


def test_dense_eq_equal_and_different():
    a = Dense(3, 2, jax.random.PRNGKey(0))
    b = Dense(3, 2, jax.random.PRNGKey(0))
    # stessi pesi: PRNGKey identico, quindi W e b coincidono
    assert a == b
    c = Dense(3, 4, jax.random.PRNGKey(0))
    assert a != c


def test_dense_backward_before_forward_raises():
    d = Dense(3, 2, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        d.backward(jnp.ones(2))


def test_dense_params_returns_two_param_objects():
    d = Dense(3, 2, jax.random.PRNGKey(0))
    ps = d.params()
    assert len(ps) == 2
    assert ps[0] is d.W
    assert ps[1] is d.b


def test_embedding_repr_and_eq():
    a = Embedding(10, 4, jax.random.PRNGKey(0))
    b = Embedding(10, 4, jax.random.PRNGKey(0))
    assert repr(a) == "Embedding(n_tokens=10, dim=4)"
    assert a == b
    assert a.__eq__("x") is NotImplemented
    c = Embedding(10, 5, jax.random.PRNGKey(0))
    assert a != c


def test_embedding_backward_before_forward_raises():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        e.backward(jnp.ones((1, 4)))


def test_embedding_params_returns_one_param():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    ps = e.params()
    assert len(ps) == 1
    assert ps[0] is e.E


def test_dropout_repr_and_eq():
    a = Dropout(0.5, jax.random.PRNGKey(0))
    b = Dropout(0.5, jax.random.PRNGKey(0))
    c = Dropout(0.3, jax.random.PRNGKey(0))
    assert repr(a) == "Dropout(p=0.5)"
    assert a == b
    assert a != c
    assert a.__eq__("x") is NotImplemented


def test_dropout_backward_before_forward_raises():
    d = Dropout(0.5, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        d.backward(jnp.ones(3))


def test_dropout_params_is_empty():
    d = Dropout(0.5, jax.random.PRNGKey(0))
    assert d.params() == []


def test_residual_repr_and_eq():
    a = Residual(Dense(4, 4, jax.random.PRNGKey(0)))
    b = Residual(Dense(4, 4, jax.random.PRNGKey(0)))
    assert repr(a).startswith("Residual(inner=Dense(")
    assert a == b
    assert a.__eq__("x") is NotImplemented
    c = Residual(Dense(4, 5, jax.random.PRNGKey(0)))
    assert a != c


def test_residual_backward_before_forward_raises():
    r = Residual(Dense(4, 4, jax.random.PRNGKey(0)))
    with pytest.raises(RuntimeError, match="call forward"):
        r.backward(jnp.ones(4))


def test_residual_params_returns_inner_params():
    inner = Dense(4, 4, jax.random.PRNGKey(0))
    r = Residual(inner)
    ps = r.params()
    assert len(ps) == 2
    assert ps[0] is inner.W
    assert ps[1] is inner.b


def test_batchnorm_backward_before_forward_raises():
    bn = BatchNorm(3)
    with pytest.raises(RuntimeError, match="call forward"):
        bn.backward(jnp.ones((2, 3)))


def test_batchnorm_params_returns_gamma_and_beta():
    bn = BatchNorm(3)
    ps = bn.params()
    assert ps[0] is bn.gamma
    assert ps[1] is bn.beta


def test_layernorm_backward_before_forward_raises():
    ln = LayerNorm(3)
    with pytest.raises(RuntimeError, match="call forward"):
        ln.backward(jnp.ones((2, 3)))


def test_layernorm_params_returns_gain_and_bias():
    ln = LayerNorm(3)
    ps = ln.params()
    assert ps[0] is ln.gain
    assert ps[1] is ln.bias


def test_adapter_non_numeric_value_counts_as_missing():
    a = AsTransformer(Dense(3, 2, jax.random.PRNGKey(0)), ["a", "b", "c"])
    out = a.transform_one({"a": "not-a-number", "b": 0.0, "c": 0.0})
    assert out == {}
    assert a.n_missing == 1


def test_layernorm_backward_1d_matches_batch_row():
    from dense_armor.utility.neural.norm import LayerNorm

    x_row = jnp.array([0.112, -0.406, 2.108, -0.268])
    x_batch = x_row[None, :]

    ln1 = LayerNorm(4)
    ln1.forward(x_row)
    g1 = jnp.ones(4)
    dx1 = np.asarray(ln1.backward(g1))
    gain1 = np.asarray(ln1.gain.grad)
    bias1 = np.asarray(ln1.bias.grad)

    ln2 = LayerNorm(4)
    ln2.forward(x_batch)
    g2 = jnp.ones((1, 4))
    dx2 = np.asarray(ln2.backward(g2))
    gain2 = np.asarray(ln2.gain.grad)
    bias2 = np.asarray(ln2.bias.grad)

    assert np.allclose(dx1, dx2[0], atol=1e-6)
    assert np.allclose(gain1, gain2, atol=1e-6)
    assert np.allclose(bias1, bias2, atol=1e-6)
    assert not np.allclose(gain1, gain1[0])


def test_batchnorm_backward_after_eval_raises():
    from dense_armor.utility.neural.norm import BatchNorm

    bn = BatchNorm(3)
    bn.forward(jnp.ones((4, 3)), training=True)
    bn.forward(jnp.ones((1, 3)), training=False)
    with pytest.raises(RuntimeError, match="backward after an evaluation"):
        bn.backward(jnp.ones((1, 3)))


def test_dropout_backward_after_eval_raises():
    d = Dropout(0.5, jax.random.PRNGKey(0))
    d.forward(jnp.ones(4), training=True)
    d.forward(jnp.ones(4), training=False)
    with pytest.raises(RuntimeError, match="backward after an evaluation"):
        d.backward(jnp.ones(4))


def test_batchnorm_training_rejects_1d_input():
    from dense_armor.utility.neural.norm import BatchNorm

    bn = BatchNorm(3)
    with pytest.raises(ValueError, match="2-D batch"):
        bn.forward(jnp.ones(3), training=True)


def test_embedding_rejects_float_ids():
    e = Embedding(10, 4, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="integer array"):
        e.forward(jnp.array([1.0, 2.0]))
