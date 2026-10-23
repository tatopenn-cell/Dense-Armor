"""Tests for positions.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.positions import LearnedPositions


def _fixed():
    m = LearnedPositions(3, 2, jax.random.PRNGKey(0))
    m.P.value = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return m


def test_forward_hand_case():
    m = _fixed()
    y = np.asarray(m.forward(jnp.zeros((2, 2))))
    print(f"y: {y.tolist()}")
    assert np.allclose(y, [[1.0, 2.0], [3.0, 4.0]])


def test_forward_batch_hand_case():
    m = _fixed()
    x = jnp.zeros((3, 2, 2))
    y = np.asarray(m.forward(x))
    print(f"batch y shape {y.shape}")
    assert y.shape == (3, 2, 2)
    for b in range(3):
        assert np.allclose(y[b], [[1.0, 2.0], [3.0, 4.0]])


def test_backward_accumulates_P_grad():
    m = _fixed()
    x = jnp.zeros((3, 2, 2))
    y = m.forward(x)
    m.backward(jnp.ones_like(y))
    print(f"P.grad: {np.asarray(m.P.grad).tolist()}")
    expected = np.array([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0]])
    assert np.allclose(m.P.grad, expected)


def test_backward_squeeze():
    m = _fixed()
    x = jnp.zeros((2, 2))
    y = m.forward(x)
    m.backward(jnp.ones_like(y))
    print(f"squeeze P.grad: {np.asarray(m.P.grad).tolist()}")
    expected = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert np.allclose(m.P.grad, expected)


def test_gradient_of_x_equals_g():
    m = _fixed()
    x = jnp.zeros((2, 2))
    _ = m.forward(x)
    g = jnp.array([[0.5, -0.3], [1.2, 0.7]])
    dx = m.backward(g)
    assert np.allclose(dx, g)


def test_too_long_raises():
    m = _fixed()
    with pytest.raises(ValueError):
        m.forward(jnp.zeros((4, 2)))


def test_value_errors():
    with pytest.raises(ValueError):
        LearnedPositions(0, 2, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        LearnedPositions(-1, 2, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        LearnedPositions(2, 0, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        LearnedPositions(2, 2, jax.random.PRNGKey(0), std=0.0)
    with pytest.raises(ValueError):
        LearnedPositions(2, 2, jax.random.PRNGKey(0), std=-0.01)


def test_backward_before_forward_raises():
    m = _fixed()
    with pytest.raises(RuntimeError):
        m.backward(jnp.ones((2, 2)))


def test_params_repr():
    m = _fixed()
    assert len(m.params()) == 1
    assert m.params()[0] is m.P
    s = repr(m)
    assert "LearnedPositions(3, 2" in s
    assert "std=0.02" in s


def test_grad_matches_jax_grad():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 3)).astype(np.float32))
    m = LearnedPositions(4, 3, jax.random.PRNGKey(0))
    y = m.forward(x)
    g = jnp.asarray(rng.normal(0.0, 1.0, y.shape).astype(np.float32)).astype(y.dtype)
    dx = np.asarray(m.backward(g))

    m2 = LearnedPositions(4, 3, jax.random.PRNGKey(0))
    m2.P.value = m.P.value

    def loss(xv):
        return jnp.sum(m2.forward(xv) * g)

    dx_ref = np.asarray(jax.grad(loss)(x))
    print(f"dx diff vs jax.grad: {np.max(np.abs(dx - dx_ref)):.2e}")
    assert np.allclose(dx, dx_ref, atol=1e-6)


@pytest.mark.training
def test_trainer_learns_order():
    from dense_armor.utility.neural.chain import Sequential
    from dense_armor.utility.neural.conv import Flatten
    from dense_armor.utility.neural.layers import Dense
    from dense_armor.utility.neural.losses import MSE
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import Adam

    rng = np.random.default_rng(0)
    n = 40
    T = 3
    d = 4
    X = np.zeros((n, T, d), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        first_high = rng.integers(0, 2)
        if first_high:
            X[i, 0, 0] = 1.0
            X[i, 1, 1] = 1.0
            y[i, 0] = 1.0
        else:
            X[i, 0, 1] = 1.0
            X[i, 1, 0] = 1.0
            y[i, 0] = -1.0

    model = Sequential(
        [
            LearnedPositions(T, d, jax.random.PRNGKey(0)),
            Flatten(),
            Dense(T * d, 1, jax.random.PRNGKey(1)),
        ]
    )
    before = float(
        np.mean((np.asarray(model.forward(jnp.asarray(X), training=False)) - y) ** 2)
    )
    t = Trainer(model, MSE(), Adam(0.02))
    t.fit(X, y, epochs=60, batch_size=10, key=jax.random.PRNGKey(2))
    after = float(
        np.mean((np.asarray(model.forward(jnp.asarray(X), training=False)) - y) ** 2)
    )
    print(f"order task: before={before:.4f} after={after:.4f}")
    assert after < before / 2.0, f"{after} not < {before / 2.0}"
