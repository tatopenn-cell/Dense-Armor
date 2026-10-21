"""Tests for conv.py: Conv1D, Conv2D, pooling, Flatten."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.conv import (
    AvgPool1D,
    AvgPool2D,
    Conv1D,
    Conv2D,
    Flatten,
    MaxPool1D,
    MaxPool2D,
)
from dense_armor.utility.neural.gradcheck import gradcheck


def test_conv1d_x_gradcheck():
    c = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 2))

    def f(xx):
        c.x_ = None
        return c.forward(xx)

    def b(xx, y, g):
        c.x_ = None
        c.forward(xx)
        return c.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_conv1d_w_gradcheck():
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    c0 = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)

    def f(w):
        c = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)
        c.W.value = w.reshape(3, 2, 3)
        return c.forward(x)

    def b(w, y, g):
        c = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)
        c.W.value = w.reshape(3, 2, 3)
        c.forward(x)
        c.backward(g)
        return c.W.grad.reshape(-1)

    assert gradcheck(f, b, c0.W.value.reshape(-1)) < 1e-4


def test_conv2d_x_gradcheck():
    c = Conv2D(1, 2, 3, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 5, 1))

    def f(xx):
        c.x_ = None
        return c.forward(xx)

    def b(xx, y, g):
        c.x_ = None
        c.forward(xx)
        return c.backward(g)

    assert gradcheck(f, b, x) < 1e-4


def test_conv2d_sums_by_hand():
    c = Conv2D(1, 1, 3, jax.random.PRNGKey(0))
    c.W.value = jnp.ones_like(c.W.value)
    c.b.value = jnp.zeros_like(c.b.value)
    x = jnp.arange(1.0, 26.0).reshape(5, 5, 1)
    y = np.asarray(c.forward(x)).reshape(3, 3)
    hand = np.array(
        [
            [
                1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13,
                2 + 3 + 4 + 7 + 8 + 9 + 12 + 13 + 14,
                3 + 4 + 5 + 8 + 9 + 10 + 13 + 14 + 15,
            ],
            [
                6 + 7 + 8 + 11 + 12 + 13 + 16 + 17 + 18,
                7 + 8 + 9 + 12 + 13 + 14 + 17 + 18 + 19,
                8 + 9 + 10 + 13 + 14 + 15 + 18 + 19 + 20,
            ],
            [
                11 + 12 + 13 + 16 + 17 + 18 + 21 + 22 + 23,
                12 + 13 + 14 + 17 + 18 + 19 + 22 + 23 + 24,
                13 + 14 + 15 + 18 + 19 + 20 + 23 + 24 + 25,
            ],
        ],
        dtype=float,
    )
    assert np.allclose(y, hand)


def test_output_size_relationship_6():
    for i, k, p, s, expected in [
        (5, 3, 0, 1, 3),
        (5, 3, 1, 1, 5),
        (5, 3, 0, 2, 2),
        (6, 3, 1, 2, 3),
    ]:
        c = Conv1D(1, 1, k, jax.random.PRNGKey(0), stride=s, padding=p)
        out = c.forward(jnp.ones((i, 1)))
        assert out.shape[0] == expected, (
            f"i={i}, k={k}, p={p}, s={s}: {out.shape[0]} != {expected}"
        )


def test_conv1d_matches_lax():
    c = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    y = c.forward(x)
    y_ref = (
        jax.lax.conv_general_dilated(
            x[None],
            c.W.value,
            (1,),
            ((1, 1),),
            dimension_numbers=("NWC", "WIO", "NWC"),
        )[0]
        + c.b.value
    )
    assert np.allclose(np.asarray(y), np.asarray(y_ref))


def test_conv2d_batch_equals_single():
    c = Conv2D(1, 2, 3, jax.random.PRNGKey(0))
    xb = jax.random.normal(jax.random.PRNGKey(1), (4, 5, 5, 1))
    yb = c.forward(xb)
    y1 = c.forward(xb[0])
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_conv1d_grad_accumulates():
    c = Conv1D(1, 1, 3, jax.random.PRNGKey(0), padding=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 1))
    g = jnp.ones((5, 1))
    c.forward(x)
    c.backward(g)
    first = np.asarray(c.W.grad)
    c.forward(x)
    c.backward(g)
    second = np.asarray(c.W.grad)
    assert np.allclose(second, 2.0 * first, atol=1e-6)


def test_conv1d_causal_step_equals_forward():
    c = Conv1D(1, 1, 3, jax.random.PRNGKey(0), causal=True)
    seq = jax.random.normal(jax.random.PRNGKey(1), (7, 1))
    yf = c.forward(seq)
    buf = c.init_buffer()
    ys = []
    for t in range(7):
        yt, buf = c.step(seq[t], buf)
        ys.append(yt)
    ys = jnp.stack(ys)
    assert np.allclose(np.asarray(ys), np.asarray(yf), atol=1e-6)


def test_conv1d_causal_future_does_not_change_past():
    c = Conv1D(1, 1, 3, jax.random.PRNGKey(0), causal=True)
    s1 = jax.random.normal(jax.random.PRNGKey(1), (7, 1))
    s2 = s1.at[5:].set(s1[5:] + 1.0)
    y1 = c.forward(s1)
    y2 = c.forward(s2)
    assert np.allclose(np.asarray(y1[:5]), np.asarray(y2[:5]))


def test_conv1d_causal_padding_and_stride_raise():
    with pytest.raises(ValueError, match="causal=True requires stride=1"):
        Conv1D(1, 1, 3, jax.random.PRNGKey(0), stride=2, causal=True)
    with pytest.raises(ValueError, match="causal=True requires padding=0"):
        Conv1D(1, 1, 3, jax.random.PRNGKey(0), padding=1, causal=True)


def test_conv1d_kernel_larger_than_input_raises():
    c = Conv1D(1, 1, 5, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="is larger than input"):
        c.forward(jnp.ones((3, 1)))


def test_conv2d_kernel_larger_than_input_raises():
    c = Conv2D(1, 1, 5, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="is larger than input"):
        c.forward(jnp.ones((3, 3, 1)))


def test_conv_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        Conv1D(1, 1, 3, jax.random.PRNGKey(0)).backward(jnp.ones((3, 1)))
    with pytest.raises(RuntimeError, match="call forward"):
        Conv2D(1, 1, 3, jax.random.PRNGKey(0)).backward(jnp.ones((3, 3, 1)))


def test_maxpool2d_hand_and_grad():
    p = MaxPool2D(2)
    x = jnp.arange(16.0).reshape(4, 4, 1)
    y = np.asarray(p.forward(x)).reshape(2, 2)
    assert np.allclose(y, np.array([[5.0, 7.0], [13.0, 15.0]]))
    g = jnp.ones((2, 2, 1))
    dx = np.asarray(p.backward(g)).reshape(4, 4)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(dx, expected)


def test_maxpool2d_tie_to_first_position():
    p = MaxPool2D(2)
    x = jnp.ones((2, 2, 1))
    p.forward(x)
    dx = np.asarray(p.backward(jnp.ones((1, 1, 1)))).reshape(2, 2)
    assert dx[0, 0] == 1.0 and dx[0, 1] == 0.0
    assert dx[1, 0] == 0.0 and dx[1, 1] == 0.0


def test_maxpool1d_hand_and_grad():
    p = MaxPool1D(2)
    x = jnp.array([[1.0], [3.0], [2.0], [5.0]])
    y = np.asarray(p.forward(x)).reshape(2)
    assert np.allclose(y, np.array([3.0, 5.0]))
    dx = np.asarray(p.backward(jnp.ones((2, 1)))).reshape(4)
    assert np.allclose(dx, np.array([0.0, 1.0, 0.0, 1.0]))


def test_avgpool2d_hand_and_grad():
    p = AvgPool2D(2)
    x = jnp.arange(16.0).reshape(4, 4, 1)
    y = np.asarray(p.forward(x)).reshape(2, 2)
    assert np.allclose(y, np.array([[2.5, 4.5], [10.5, 12.5]]))
    dx = np.asarray(p.backward(jnp.ones((2, 2, 1)))).reshape(4, 4)
    assert np.allclose(dx, np.full((4, 4), 1.0 / 4.0))


def test_avgpool1d_hand_and_grad():
    p = AvgPool1D(2)
    x = jnp.array([[1.0], [3.0], [2.0], [4.0]])
    y = np.asarray(p.forward(x)).reshape(2)
    assert np.allclose(y, np.array([2.0, 3.0]))
    dx = np.asarray(p.backward(jnp.ones((2, 1)))).reshape(4)
    assert np.allclose(dx, np.array([0.5, 0.5, 0.5, 0.5]))


def test_pooling_gradchecks():
    x1 = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
    x2 = jax.random.normal(jax.random.PRNGKey(0), (4, 4, 2))
    for pool, xx in [
        (MaxPool1D(2), x1),
        (AvgPool1D(2), x1),
        (MaxPool2D(2), x2),
        (AvgPool2D(2), x2),
    ]:

        def f(x, p=pool):
            p.x_ = None
            return p.forward(x)

        def b(x, y, g, p=pool):
            p.x_ = None
            p.forward(x)
            return p.backward(g)

        assert gradcheck(f, b, xx) < 1e-4


def test_flatten_roundtrip():
    f = Flatten()
    x = jnp.ones((3, 4, 5))
    y = f.forward(x)
    assert y.shape == (3, 20)
    dx = f.backward(jnp.ones((3, 20)))
    assert dx.shape == (3, 4, 5)


def test_flatten_gradcheck():
    f = Flatten()
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))

    def fn(xx):
        f.x_ = None
        return f.forward(xx)

    def bd(xx, y, g):
        f.x_ = None
        f.forward(xx)
        return f.backward(g)

    assert gradcheck(fn, bd, x) < 1e-4


def test_pooling_has_no_params():
    for p in [MaxPool1D(2), MaxPool2D(2), AvgPool1D(2), AvgPool2D(2), Flatten()]:
        assert p.params() == []


def test_pooling_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        MaxPool1D(2).backward(jnp.ones((2, 1)))
    with pytest.raises(RuntimeError, match="call forward"):
        AvgPool2D(2).backward(jnp.ones((1, 1, 1)))


def test_kernel_below_one_raises_everywhere():
    for cls in (Conv1D, Conv2D, MaxPool1D, MaxPool2D, AvgPool1D, AvgPool2D):
        with pytest.raises(ValueError, match="kernel must be at least 1"):
            if cls in (Conv1D, Conv2D):
                cls(1, 1, 0, jax.random.PRNGKey(0))
            else:
                cls(0)


def test_repr_all_layers():
    assert "Conv1D" in repr(Conv1D(1, 2, 3, jax.random.PRNGKey(0)))
    assert "Conv2D" in repr(Conv2D(1, 2, 3, jax.random.PRNGKey(0)))
    assert "MaxPool1D" in repr(MaxPool1D(2))
    assert "MaxPool2D" in repr(MaxPool2D(2))
    assert "AvgPool1D" in repr(AvgPool1D(2))
    assert "AvgPool2D" in repr(AvgPool2D(2))
    assert repr(Flatten()) == "Flatten()"


def test_conv_params_returns_two():
    for c in [
        Conv1D(1, 2, 3, jax.random.PRNGKey(0)),
        Conv2D(1, 2, 3, jax.random.PRNGKey(0)),
    ]:
        ps = c.params()
        assert len(ps) == 2
        assert ps[0] is c.W
        assert ps[1] is c.b


def test_init_buffer_and_step_non_causal_raise():
    c = Conv1D(1, 1, 3, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="init_buffer\(\) requires causal=True"):
        c.init_buffer()
    with pytest.raises(ValueError, match="step\(\) requires causal=True"):
        c.step(jnp.zeros(1), jnp.zeros((2, 1)))


def test_avg_pool_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        AvgPool1D(2).backward(jnp.ones((2, 1)))
    with pytest.raises(RuntimeError, match="call forward"):
        AvgPool2D(2).backward(jnp.ones((1, 1, 1)))


def test_flatten_1d_input():
    f = Flatten()
    x = jnp.array([1.0, 2.0, 3.0])
    y = f.forward(x)
    assert y.shape == (3,)
    dx = f.backward(jnp.ones(3))
    assert dx.shape == (3,)


def test_flatten_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        Flatten().backward(jnp.ones(3))
