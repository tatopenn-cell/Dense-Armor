"""Tests for mlp and resnet_mlp."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.activations import Relu, Sigmoid, Tanh
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import (
    bce_with_logits_backward,
    bce_with_logits_forward,
)
from dense_armor.utility.neural.models import mlp, resnet_mlp
from dense_armor.utility.optim.adam import Adam


def test_mlp_shapes():
    m = mlp([2, 8, 1], jax.random.PRNGKey(0))
    shapes = [p.value.shape for p in m.params()]
    assert shapes == [(2, 8), (8,), (8, 1), (1,)]


def test_mlp_full_gradcheck():
    m = mlp([3, 5, 2], jax.random.PRNGKey(0))
    x = jnp.array([0.5, -1.0, 2.0])

    def f(xx):
        return m.forward(xx)

    def b(xx, y, g):
        m.forward(xx)
        return m.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_mlp_activations_are_the_right_classes():
    for name, cls in [("relu", Relu), ("tanh", Tanh), ("sigmoid", Sigmoid)]:
        m = mlp([3, 4, 2], jax.random.PRNGKey(0), activation=name)
        acts = [x for x in m.layers if isinstance(x, cls)]
        assert len(acts) == 1


def test_mlp_bad_args_raise():
    with pytest.raises(ValueError, match="at least two"):
        mlp([3], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="activation must be"):
        mlp([3, 4], jax.random.PRNGKey(0), activation="relu6")


def test_mlp_init_choice():
    m_relu = mlp([3, 4], jax.random.PRNGKey(0), activation="relu")
    d_relu = [x for x in m_relu.layers if isinstance(x, Dense)][0]
    assert d_relu.init == "he"
    m_tanh = mlp([3, 4], jax.random.PRNGKey(0), activation="tanh")
    d_tanh = [x for x in m_tanh.layers if isinstance(x, Dense)][0]
    assert d_tanh.init == "xavier"


def test_resnet_mlp_param_count():
    r = resnet_mlp(8, 2, 3, 2, jax.random.PRNGKey(0))
    assert len(r.params()) == 2 + 4 * 2 + 2


def test_resnet_mlp_gradcheck():
    r = resnet_mlp(4, 2, 3, 2, jax.random.PRNGKey(0))
    x = jnp.array([0.5, -1.0, 2.0])

    def f(xx):
        return r.forward(xx)

    def b(xx, y, g):
        r.forward(xx)
        return r.backward(g)

    assert gradcheck(f, b, x) < 1e-3


def test_resnet_block_zero_second_dense_is_the_rectifier_of_input():
    from dense_armor.utility.neural.chain import Sequential
    from dense_armor.utility.neural.layers import Residual

    r = resnet_mlp(4, 1, 4, 1, jax.random.PRNGKey(0))
    block = None
    for layer in r.layers:
        if isinstance(layer, Sequential) and any(
            isinstance(x, Residual) for x in layer.layers
        ):
            block = layer
            break
    assert block is not None
    residual = block.layers[0]
    inner = residual.inner
    second = inner.layers[2]
    second.W.value = jnp.zeros_like(second.W.value)
    second.b.value = jnp.zeros_like(second.b.value)
    x = jnp.array([1.0, -2.0, 3.0, -4.0])
    y = block.forward(x)
    expected = jnp.maximum(x, 0.0)
    assert np.allclose(np.asarray(y), np.asarray(expected), atol=1e-5)


def test_resnet_bad_depth_raises():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        resnet_mlp(8, 0, 2, 1, jax.random.PRNGKey(0))


def _train_xor(make_model, steps, lr, seed=0):
    k = jax.random.PRNGKey(seed)
    model = make_model(k)
    opt = {id(p): Adam(lr=lr) for p in model.params()}
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        model.zero_grad()
        perm = rng.permutation(4)
        batch_loss = 0.0
        for i in perm:
            logit = model.forward(jnp.asarray(X[i]))[0]
            batch_loss += float(
                bce_with_logits_forward(jnp.array([logit]), jnp.array([y[i]]))
            )
            g = bce_with_logits_backward(jnp.array([logit]), jnp.array([y[i]]))[0]
            model.backward(jnp.array([g]))
        for p in model.params():
            p.update(opt[id(p)])
        losses.append(batch_loss / 4.0)
    ok = 0
    for xi, yi in zip(X, y):
        logit = float(model.forward(jnp.asarray(xi))[0])
        pred = 1.0 if logit > 0.0 else 0.0
        if pred == float(yi):
            ok += 1
    return ok / 4.0, losses[-1]


def test_mlp_learns_xor():
    acc, final_loss = _train_xor(
        lambda k: mlp([2, 8, 1], k),
        steps=200,
        lr=0.05,
    )
    assert acc == 1.0, f"accuracy {acc}, final loss {final_loss}"


def test_resnet_mlp_learns_xor():
    acc, final_loss = _train_xor(
        lambda k: resnet_mlp(8, 2, 2, 1, k),
        steps=200,
        lr=0.05,
    )
    assert acc == 1.0, f"accuracy {acc}, final loss {final_loss}"


def test_same_key_twice_gives_the_same_final_loss():
    acc1, loss1 = _train_xor(lambda k: mlp([2, 8, 1], k), 100, 0.05)
    acc2, loss2 = _train_xor(lambda k: mlp([2, 8, 1], k), 100, 0.05)
    assert abs(loss1 - loss2) < 1e-6
