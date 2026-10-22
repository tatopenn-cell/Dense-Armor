"""Tests for bidirectional.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.bidirectional import Bidirectional
from dense_armor.utility.neural.recurrent import GRU, LSTM, RNN


def _one_one_rnn(key):
    r = RNN(1, 1, key)
    r.W.value = jnp.array([[1.0]])
    r.U.value = jnp.array([[0.5]])
    r.b.value = jnp.array([0.0])
    return r


def test_hand_case_1x1_rnn():
    m = Bidirectional(
        _one_one_rnn(jax.random.PRNGKey(0)), _one_one_rnn(jax.random.PRNGKey(1))
    )
    out = np.asarray(m.forward(jnp.array([[1.0], [0.0], [-1.0]])))
    assert out.shape == (3, 2)
    print(f"fwd: {out[:, 0].tolist()}")
    print(f"bwd: {out[:, 1].tolist()}")
    assert np.allclose(out[:, 0], [0.761594, 0.363399, -0.674144], atol=1e-6)
    assert np.allclose(out[:, 1], [0.674144, -0.363399, -0.761594], atol=1e-6)


def test_backward_half_does_not_depend_on_past():
    m = Bidirectional(
        _one_one_rnn(jax.random.PRNGKey(0)), _one_one_rnn(jax.random.PRNGKey(1))
    )
    x = jnp.array([[1.0], [0.0], [-1.0]])
    x2 = x.at[0, 0].set(2.0)
    y1 = np.asarray(m.forward(x))
    y2 = np.asarray(m.forward(x2))
    print(f"bwd half diff at t=2 when x[0] changes: {abs(y1[2, 1] - y2[2, 1]):.3e}")
    assert abs(y1[2, 1] - y2[2, 1]) < 1e-6


def test_forward_half_does_not_depend_on_future():
    m = Bidirectional(
        _one_one_rnn(jax.random.PRNGKey(0)), _one_one_rnn(jax.random.PRNGKey(1))
    )
    x = jnp.array([[1.0], [0.0], [-1.0]])
    x2 = x.at[2, 0].set(2.0)
    y1 = np.asarray(m.forward(x))
    y2 = np.asarray(m.forward(x2))
    print(f"fwd half diff at t=0 when x[2] changes: {abs(y1[0, 0] - y2[0, 0]):.3e}")
    assert abs(y1[0, 0] - y2[0, 0]) < 1e-6


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RNN, {}),
        (GRU, {}),
        (LSTM, {}),
    ],
)
def _copy_params(src, dst):
    for ps, pd in zip(src.params(), dst.params()):
        pd.value = ps.value


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RNN, {}),
        (GRU, {}),
        (LSTM, {}),
    ],
)
def test_matches_manual_composition(cls, kwargs):
    x = jnp.array([[1.0, 2.0], [0.5, -0.5], [-1.0, 0.0]])
    key = jax.random.PRNGKey(0)
    f = cls(2, 3, key, **kwargs)
    b = cls(2, 3, jax.random.PRNGKey(1), **kwargs)
    m = Bidirectional(f, b)
    out = np.asarray(m.forward(x))

    f2 = cls(2, 3, key, **kwargs)
    _copy_params(f, f2)
    y_fwd = np.asarray(f2.forward(x))

    b2 = cls(2, 3, jax.random.PRNGKey(1), **kwargs)
    _copy_params(b, b2)
    y_bwd = np.asarray(b2.forward(jnp.flip(x, axis=0)))
    y_bwd_back = np.flip(y_bwd, axis=0)

    expected = np.concatenate([y_fwd, y_bwd_back], axis=-1)
    assert np.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("cls", [LSTM, GRU])
def test_gradcheck(cls):
    f = cls(2, 3, jax.random.PRNGKey(0))
    b = cls(2, 3, jax.random.PRNGKey(1))
    m = Bidirectional(f, b)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
    g = jax.random.normal(jax.random.PRNGKey(3), (4, 6))
    _ = m.forward(x)
    params = m.params()
    for p in params:
        p.zero_grad()
    dx = m.backward(g)
    print(f"{cls.__name__}: dx shape {np.asarray(dx).shape}")

    def fwd(xv):
        return jnp.sum(m.forward(xv) * g)

    dx_ref = jax.grad(fwd)(x)
    diff = float(np.max(np.abs(np.asarray(dx) - np.asarray(dx_ref))))
    print(f"{cls.__name__}: dx diff {diff:.3e}")
    assert diff < 1e-4


def test_different_hidden_sizes():
    f = RNN(2, 3, jax.random.PRNGKey(0))
    b = RNN(2, 5, jax.random.PRNGKey(1))
    m = Bidirectional(f, b)
    y = m.forward(jnp.ones((4, 2)))
    print(f"shape: {y.shape}")
    assert y.shape == (4, 8)


def test_batch_and_squeeze_match():
    f = RNN(2, 3, jax.random.PRNGKey(0))
    b = RNN(2, 3, jax.random.PRNGKey(1))
    m = Bidirectional(f, b)
    x = jnp.array([[1.0, 2.0], [0.5, -0.5], [-1.0, 0.0]])
    y_squeeze = np.asarray(m.forward(x))

    f2 = RNN(2, 3, jax.random.PRNGKey(0))
    f2.W.value = f.W.value
    f2.U.value = f.U.value
    f2.b.value = f.b.value
    b2 = RNN(2, 3, jax.random.PRNGKey(1))
    b2.W.value = b.W.value
    b2.U.value = b.U.value
    b2.b.value = b.b.value
    m2 = Bidirectional(f2, b2)
    y_batch = np.asarray(m2.forward(x[None]))

    assert y_batch.shape == (1, 3, 6)
    assert np.allclose(y_squeeze, y_batch[0], atol=1e-6)


def test_value_error_on_non_layer():
    class Fake:
        def forward(self, x, training=True):
            return x

    with pytest.raises(ValueError):
        Bidirectional(Fake(), _one_one_rnn(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError):
        Bidirectional(_one_one_rnn(jax.random.PRNGKey(0)), Fake())


def test_backward_before_forward_raises():
    m = Bidirectional(
        _one_one_rnn(jax.random.PRNGKey(0)), _one_one_rnn(jax.random.PRNGKey(1))
    )
    with pytest.raises(RuntimeError):
        m.backward(jnp.ones((3, 2)))
