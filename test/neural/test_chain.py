"""Tests for Sequential and Parallel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.chain import Parallel, Sequential
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.norm import LayerNorm


def _seq3():
    return Sequential(
        [
            Dense(3, 4, jax.random.PRNGKey(0)),
            Relu(),
            Dense(4, 2, jax.random.PRNGKey(1)),
        ]
    )


def test_sequential_backward_matches_finite_differences_input():
    s = _seq3()

    def f(x):
        s.forward(x)
        return s.forward(x)

    def b(x, y, g):
        s.forward(x)
        return s.backward(g)

    x = jnp.array([0.5, -1.0, 2.0])
    assert gradcheck(f, b, x) < 1e-4


def test_sequential_gradient_of_each_param_matches_finite_differences():
    s = _seq3()
    x = jnp.array([0.5, -1.0, 2.0])
    for i, p in enumerate(s.params()):

        def f_w(w, i=i):
            old = s.params()[i].value
            s.params()[i].value = w
            try:
                return s.forward(x)
            finally:
                s.params()[i].value = old

        def b_w(w, y, gg, i=i):
            old = s.params()[i].value
            s.params()[i].value = w
            s.zero_grad()
            s.forward(x)
            s.backward(gg)
            grad = s.params()[i].grad
            s.params()[i].value = old
            return grad

        assert gradcheck(f_w, b_w, p.value) < 1e-4


def test_sequential_single_sample_equals_batch_row():
    s = _seq3()
    xb = jnp.array(
        [
            [0.1, 0.2, 0.3],
            [1.0, -1.0, 0.0],
        ]
    )
    yb = s.forward(xb)
    y1 = s.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]), atol=1e-6)


def test_sequential_one_sample_gradients_sum_to_batch_gradient():
    s = _seq3()
    rng = np.random.default_rng(0)
    xb = jnp.asarray(rng.normal(0.0, 1.0, (8, 3)).astype(np.float32))

    s.zero_grad()
    yb = s.forward(xb)
    gb = jnp.ones_like(yb) / 8.0
    s.backward(gb)
    grad_batch = [np.array(p.grad) for p in s.params()]

    s.zero_grad()
    for row in xb:
        y = s.forward(row)
        g = jnp.ones_like(y) / 8.0
        s.backward(g)
    grad_online = [np.array(p.grad) for p in s.params()]

    for gb_, go_ in zip(grad_batch, grad_online):
        assert np.allclose(gb_, go_, atol=1e-5)


def test_sequential_grad_accumulates_and_zero_grad_resets():
    s = _seq3()
    x = jnp.array([0.5, -1.0, 2.0])
    g = jnp.ones(2)
    s.forward(x)
    s.backward(g)
    first = [np.array(p.grad) for p in s.params()]
    s.forward(x)
    s.backward(g)
    second = [np.array(p.grad) for p in s.params()]
    for f, sec in zip(first, second):
        assert np.allclose(sec, 2.0 * f, atol=1e-6)
    s.zero_grad()
    for p in s.params():
        assert bool(jnp.all(p.grad == 0.0))


def test_sequential_state_dict_roundtrip():
    s = _seq3()
    x = jnp.array([0.5, -1.0, 2.0])
    y0 = np.asarray(s.forward(x))
    state = s.state_dict()
    s.load_state_dict(state)
    y1 = np.asarray(s.forward(x))
    assert np.allclose(y0, y1)
    with pytest.raises(ValueError, match="state has"):
        s.load_state_dict(state[:-1])


def test_sequential_empty_raises():
    with pytest.raises(ValueError, match="at least one layer"):
        Sequential([])


def test_sequential_with_dense_layernorm_relu():
    s = Sequential(
        [
            Dense(3, 4, jax.random.PRNGKey(0)),
            LayerNorm(4),
            Relu(),
        ]
    )
    y = s.forward(jnp.array([0.1, 0.2, 0.3]))
    assert y.shape == (4,)


def test_sequential_as_transformer_passes_check_estimator():
    s = Sequential(
        [
            Dense(3, 2, jax.random.PRNGKey(0)),
        ]
    )
    a = AsTransformer(s, ["a", "b", "c"])
    out = check_estimator(a)
    assert len(out["run"]) > 0
    assert out["not_applicable"] == []
    assert a._unit_test_skips() == set()


def test_sequential_add_len_getitem():
    s = Sequential([Dense(3, 2, jax.random.PRNGKey(0))])
    s.add(Relu())
    assert len(s) == 2
    assert isinstance(s[1], Relu)


def test_sequential_backward_before_forward_raises():
    s = _seq3()
    with pytest.raises(RuntimeError, match="call forward"):
        s.backward(jnp.ones(2))


def test_parallel_add_gradcheck_input():
    p = Parallel(
        [
            Dense(3, 4, jax.random.PRNGKey(0)),
            Dense(3, 4, jax.random.PRNGKey(1)),
        ],
        "add",
    )

    def f(x):
        return p.forward(x)

    def b(x, y, g):
        p.forward(x)
        return p.backward(g)

    x = jnp.array([0.5, -1.0, 2.0])
    assert gradcheck(f, b, x) < 1e-4


def test_parallel_concat_gradcheck_input():
    p = Parallel(
        [
            Dense(3, 4, jax.random.PRNGKey(0)),
            Dense(3, 2, jax.random.PRNGKey(1)),
        ],
        "concat",
    )

    def f(x):
        return p.forward(x)

    def b(x, y, g):
        p.forward(x)
        return p.backward(g)

    x = jnp.array([0.5, -1.0, 2.0])
    assert gradcheck(f, b, x) < 1e-4
    assert p.forward(x).shape == (6,)


def test_parallel_add_is_the_sum_of_branches():
    d1 = Dense(3, 4, jax.random.PRNGKey(0))
    d2 = Dense(3, 4, jax.random.PRNGKey(1))
    p = Parallel([d1, d2], "add")
    x = jnp.array([0.5, -1.0, 2.0])
    y = p.forward(x)
    expected = d1.forward(x) + d2.forward(x)
    assert np.allclose(np.asarray(y), np.asarray(expected))


def test_parallel_nested_in_sequential_and_residual():
    from dense_armor.utility.neural.layers import Residual

    inner = Parallel(
        [
            Dense(3, 3, jax.random.PRNGKey(0)),
            Dense(3, 3, jax.random.PRNGKey(1)),
        ],
        "add",
    )
    s = Sequential([Residual(inner)])
    x = jnp.array([0.5, -1.0, 2.0])

    def f(xx):
        s.forward(xx)
        return s.forward(xx)

    def b(xx, y, g):
        s.forward(xx)
        return s.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_parallel_bad_args_raise():
    with pytest.raises(ValueError, match="at least two branches"):
        Parallel([Dense(3, 4, jax.random.PRNGKey(0))])
    with pytest.raises(ValueError, match="merge must be"):
        Parallel(
            [Dense(3, 4, jax.random.PRNGKey(0)), Dense(3, 4, jax.random.PRNGKey(1))],
            "mean",
        )
    with pytest.raises(ValueError, match="expected the shape of branch 0"):
        p = Parallel(
            [Dense(3, 4, jax.random.PRNGKey(0)), Dense(3, 2, jax.random.PRNGKey(1))],
            "add",
        )
        p.forward(jnp.ones(3))


def test_parallel_params_and_zero_grad():
    p = Parallel(
        [
            Dense(3, 4, jax.random.PRNGKey(0)),
            Dense(3, 4, jax.random.PRNGKey(1)),
        ]
    )
    assert len(p.params()) == 4
    p.forward(jnp.ones(3))
    p.backward(jnp.ones(4))
    p.zero_grad()
    for par in p.params():
        assert bool(jnp.all(par.grad == 0.0))


def test_chain_two_layers_gradient_by_hand():
    """Backward of ``Dense(2,2) -> Relu -> Dense(2,1)`` with MSE.

    Two samples, weights written in the test, numbers computed by
    hand and checked against ``backward`` within ``1e-6``.
    """
    d1 = Dense(2, 2, jax.random.PRNGKey(0))
    d1.W.value = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    d1.b.value = jnp.array([0.0, 0.0])
    d2 = Dense(2, 1, jax.random.PRNGKey(1))
    d2.W.value = jnp.array([[5.0], [6.0]])
    d2.b.value = jnp.array([0.0])
    s = Sequential([d1, Relu(), d2])

    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    T = jnp.array([[90.0], [200.0]])
    Y = s.forward(X)
    assert np.allclose(np.asarray(Y), np.array([[95.0], [207.0]]))

    g = (Y - T) / X.shape[0]
    s.zero_grad()
    dx = s.backward(g)

    assert np.allclose(
        np.asarray(d1.W.grad), np.array([[65.0, 78.0], [95.0, 114.0]]), atol=1e-6
    )
    assert np.allclose(np.asarray(d1.b.grad), np.array([30.0, 36.0]), atol=1e-6)
    assert np.allclose(np.asarray(d2.W.grad), np.array([[70.0], [102.0]]), atol=1e-6)
    assert np.allclose(np.asarray(d2.b.grad), np.array([6.0]), atol=1e-6)
    assert np.allclose(
        np.asarray(dx), np.array([[42.5, 97.5], [59.5, 136.5]]), atol=1e-6
    )
