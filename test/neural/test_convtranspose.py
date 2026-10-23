"""Tests for convtranspose.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.conv import Conv2D
from dense_armor.utility.neural.convtranspose import (
    ConvTranspose1D,
    ConvTranspose2D,
    Upsample1D,
    Upsample2D,
)
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def test_convtranspose1d_hand_case_stride2_no_padding():
    c = ConvTranspose1D(1, 1, 3, jax.random.PRNGKey(0), stride=2)
    c.W.value = jnp.array([[[1.0]], [[10.0]], [[100.0]]])
    c.b.value = jnp.array([0.0])
    y = np.asarray(c.forward(jnp.array([[1.0], [2.0]]))).reshape(-1)
    print(f"stride=2 p=0 a=0: {y.tolist()}")
    assert np.allclose(y, [1.0, 10.0, 102.0, 20.0, 200.0], atol=1e-6)


def test_convtranspose1d_hand_case_stride2_padding1_output_padding1():
    c = ConvTranspose1D(
        1, 1, 3, jax.random.PRNGKey(0), stride=2, padding=1, output_padding=1
    )
    c.W.value = jnp.array([[[1.0]], [[10.0]], [[100.0]]])
    c.b.value = jnp.array([0.0])
    y = np.asarray(c.forward(jnp.array([[1.0], [2.0]]))).reshape(-1)
    print(f"stride=2 p=1 a=1: {y.tolist()}")
    assert np.allclose(y, [10.0, 102.0, 20.0, 200.0], atol=1e-6)


def test_convtranspose2d_adjoint_property():
    key = jax.random.PRNGKey(0)
    c_in, c_out, k = 2, 3, 3
    for s in (1, 2):
        conv = Conv2D(c_in, c_out, k, key, stride=s)
        conv.b.value = jnp.zeros_like(conv.b.value)
        tconv = ConvTranspose2D(c_out, c_in, k, key, stride=s)
        tconv.W.value = jnp.transpose(conv.W.value, (0, 1, 3, 2))
        tconv.b.value = jnp.zeros_like(tconv.b.value)
        H = 7
        x = jax.random.normal(key, (H, H, c_in))
        y = jax.random.normal(jax.random.PRNGKey(1), (conv.forward(x).shape))
        lhs = float(jnp.sum(conv.forward(x) * y))
        rhs = float(jnp.sum(x * tconv.forward(y)))
        print(f"stride={s}: lhs={lhs:.4f} rhs={rhs:.4f} diff={abs(lhs - rhs):.3e}")
        assert abs(lhs - rhs) < 1e-4, f"adjoint fails at stride={s}"


def test_output_sizes_match_relationship14():
    # Relationship 14, page 26: o' = s*(L-1) + a + k - 2p
    # with a = output_padding (our parameter).
    for L, k, s, p, a, expected in [
        (5, 3, 2, 1, 0, 9),
        (5, 3, 2, 0, 1, 12),
        (6, 4, 3, 1, 2, 19),
    ]:
        c = ConvTranspose1D(
            1,
            1,
            k,
            jax.random.PRNGKey(0),
            stride=s,
            padding=p,
            output_padding=a,
        )
        got = c.forward(jnp.zeros((L, 1))).shape[0]
        print(
            f"Rel14 (L={L}, k={k}, s={s}, p={p}, a={a}) -> {got} (expected {expected})"
        )
        assert got == expected, f"{got} != {expected}"


def test_output_sizes_match_relationship13_with_a0():
    # Relationship 13, page 26: o' = s*(L-1) + k - 2p, same as
    # Relationship 14 with a = 0. The three triples of ISTRUZIONI with
    # a = 0 give 9, 11, 17, which is what ISTRUZIONI states.
    for L, k, s, p, expected in [
        (5, 3, 2, 1, 9),
        (5, 3, 2, 0, 11),
        (6, 4, 3, 1, 17),
    ]:
        c = ConvTranspose1D(
            1,
            1,
            k,
            jax.random.PRNGKey(0),
            stride=s,
            padding=p,
            output_padding=0,
        )
        got = c.forward(jnp.zeros((L, 1))).shape[0]
        print(f"Rel13 (L={L}, k={k}, s={s}, p={p}, a=0) -> {got} (expected {expected})")
        assert got == expected, f"{got} != {expected}"


def test_upsample1d_hand_case():
    u = Upsample1D(2)
    y = np.asarray(u.forward(jnp.array([[1.0], [2.0], [3.0]]))).reshape(-1)
    print(f"Upsample1D(2): {y.tolist()}")
    assert np.allclose(y, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    g = np.asarray(u.backward(jnp.ones((6, 1)))).reshape(-1)
    print(f"Upsample1D(2).backward(ones): {g.tolist()}")
    assert np.allclose(g, [2.0, 2.0, 2.0])


def test_upsample2d_shapes():
    u = Upsample2D(2)
    x = jnp.ones((2, 2, 1))
    y = u.forward(x)
    print(f"Upsample2D(2): input {x.shape} -> output {y.shape}")
    assert y.shape == (4, 4, 1)
    g = u.backward(jnp.ones_like(y))
    print(
        f"Upsample2D.backward(ones): {g.shape} values {np.asarray(g).reshape(-1).tolist()}"
    )
    assert g.shape == (2, 2, 1)
    assert np.allclose(np.asarray(g), 4.0)


@pytest.mark.training
def test_conv_autoencoder_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (64, 8, 8, 1)).astype(np.float32)
    enc = Conv2D(1, 4, 3, jax.random.PRNGKey(0), stride=2, padding=1)
    dec = ConvTranspose2D(
        4, 1, 3, jax.random.PRNGKey(1), stride=2, padding=1, output_padding=1
    )
    m = Sequential([enc, dec])
    before = float(
        np.mean((np.asarray(m.forward(jnp.asarray(X), training=False)) - X) ** 2)
    )
    t = Trainer(m, MSE(), Adam(0.01))
    t.fit(X, X, epochs=20, batch_size=16, key=jax.random.PRNGKey(0))
    after = float(
        np.mean((np.asarray(m.forward(jnp.asarray(X), training=False)) - X) ** 2)
    )
    print(f"conv autoencoder: before={before:.4f} after={after:.4f}")
    assert after < before / 2.0, f"{after} not < {before / 2.0}"


def test_value_errors_convtranspose():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        ConvTranspose1D(1, 1, 0, key)
    with pytest.raises(ValueError):
        ConvTranspose1D(1, 1, 3, key, padding=-1)
    with pytest.raises(ValueError):
        ConvTranspose1D(1, 1, 3, key, stride=2, output_padding=2)
    with pytest.raises(ValueError):
        ConvTranspose1D(1, 1, 3, key, stride=2, output_padding=-1)
    with pytest.raises(ValueError):
        ConvTranspose2D(1, 1, 0, key)
    with pytest.raises(ValueError):
        ConvTranspose2D(1, 1, 3, key, padding=-1)
    with pytest.raises(ValueError):
        ConvTranspose2D(1, 1, 3, key, stride=2, output_padding=2)
    with pytest.raises(ValueError):
        ConvTranspose1D(1, 1, 5, key, stride=1, padding=10).forward(jnp.zeros((1, 1)))


def test_value_errors_upsample():
    with pytest.raises(ValueError):
        Upsample1D(0)
    with pytest.raises(ValueError):
        Upsample1D(-1)
    with pytest.raises(ValueError):
        Upsample2D(0)
    with pytest.raises(ValueError):
        Upsample2D(-1)


def test_backward_before_forward_raises():
    c = ConvTranspose1D(1, 1, 3, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError):
        c.backward(jnp.ones((3, 1)))
    c2 = ConvTranspose2D(1, 1, 3, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError):
        c2.backward(jnp.ones((3, 3, 1)))
    u = Upsample1D(2)
    with pytest.raises(RuntimeError):
        u.backward(jnp.ones((4, 1)))
    u2 = Upsample2D(2)
    with pytest.raises(RuntimeError):
        u2.backward(jnp.ones((4, 4, 1)))


def test_convtranspose1d_backward_matches_jax_grad():
    rng = np.random.default_rng(0)
    x_arr = rng.normal(0.0, 1.0, (4, 2)).astype(np.float32)
    g_arr = rng.normal(0.0, 1.0, (9, 3)).astype(np.float32)

    c = ConvTranspose1D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    _ = c.forward(jnp.asarray(x_arr))
    dx = np.asarray(c.backward(jnp.asarray(g_arr)))
    assert dx.shape == x_arr.shape
    assert c.W.grad.shape == c.W.value.shape
    assert c.b.grad.shape == c.b.value.shape

    def loss(xv):
        cc = ConvTranspose1D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
        cc.W.value = c.W.value
        cc.b.value = c.b.value
        return jnp.sum(cc.forward(xv) * jnp.asarray(g_arr))

    ref = np.asarray(jax.grad(loss)(jnp.asarray(x_arr)))
    print(f"ConvTranspose1D dx max diff: {np.max(np.abs(dx - ref)):.3e}")
    assert np.allclose(dx, ref, atol=1e-4)


def test_convtranspose2d_backward_matches_jax_grad():
    rng = np.random.default_rng(1)
    x_arr = rng.normal(0.0, 1.0, (3, 3, 2)).astype(np.float32)
    g_arr = rng.normal(0.0, 1.0, (7, 7, 3)).astype(np.float32)

    c = ConvTranspose2D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    _ = c.forward(jnp.asarray(x_arr))
    dx = np.asarray(c.backward(jnp.asarray(g_arr)))
    assert dx.shape == x_arr.shape
    assert c.W.grad.shape == c.W.value.shape
    assert c.b.grad.shape == c.b.value.shape

    def loss(xv):
        cc = ConvTranspose2D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
        cc.W.value = c.W.value
        cc.b.value = c.b.value
        return jnp.sum(cc.forward(xv) * jnp.asarray(g_arr))

    ref = np.asarray(jax.grad(loss)(jnp.asarray(x_arr)))
    print(f"ConvTranspose2D dx max diff: {np.max(np.abs(dx - ref)):.3e}")
    assert np.allclose(dx, ref, atol=1e-4)


def test_squeeze_and_batch_match_convtranspose():
    rng = np.random.default_rng(2)
    x1 = rng.normal(0.0, 1.0, (4, 2)).astype(np.float32)
    xb = x1[None]
    g1 = rng.normal(0.0, 1.0, (7, 3)).astype(np.float32)
    gb = g1[None]

    c1 = ConvTranspose1D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    y1 = np.asarray(c1.forward(jnp.asarray(x1)))
    c2 = ConvTranspose1D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    c2.W.value = c1.W.value
    c2.b.value = c1.b.value
    yb = np.asarray(c2.forward(jnp.asarray(xb)))
    assert np.allclose(y1, yb[0], atol=1e-6)
    dx1 = np.asarray(c1.backward(jnp.asarray(g1)))
    dxb = np.asarray(c2.backward(jnp.asarray(gb)))
    assert np.allclose(dx1, dxb[0], atol=1e-6)

    x2d = rng.normal(0.0, 1.0, (3, 3, 2)).astype(np.float32)
    g2d = rng.normal(0.0, 1.0, (6, 6, 3)).astype(np.float32)
    d1 = ConvTranspose2D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    y_s = np.asarray(d1.forward(jnp.asarray(x2d)))
    d2 = ConvTranspose2D(2, 3, 3, jax.random.PRNGKey(0), stride=2)
    d2.W.value = d1.W.value
    d2.b.value = d1.b.value
    y_b = np.asarray(d2.forward(jnp.asarray(x2d[None])))
    assert np.allclose(y_s, y_b[0], atol=1e-6)
    dx_s = np.asarray(d1.backward(jnp.asarray(g2d)))
    dx_b = np.asarray(d2.backward(jnp.asarray(g2d[None])))
    assert np.allclose(dx_s, dx_b[0], atol=1e-6)


def test_repr_and_params_all_convtranspose():
    key = jax.random.PRNGKey(0)
    for cls, name, n_params in [
        (ConvTranspose1D, "ConvTranspose1D", 2),
        (ConvTranspose2D, "ConvTranspose2D", 2),
    ]:
        c = cls(2, 3, 3, key, stride=2, padding=1, output_padding=1)
        s = repr(c)
        assert name in s
        assert "c_in=2" in s or "2," in s
        assert "stride=2" in s
        ps = c.params()
        assert len(ps) == n_params
        assert ps[0] is c.W
        assert ps[1] is c.b

    u = Upsample1D(2)
    assert repr(u) == "Upsample1D(factor=2)"
    assert u.params() == []
    u2 = Upsample2D(3)
    assert repr(u2) == "Upsample2D(factor=3)"
    assert u2.params() == []


def test_convtranspose_adjoint_full_batch():
    rng = np.random.default_rng(3)
    c_in, c_out, k = 2, 3, 3
    x = rng.normal(0.0, 1.0, (5, 5, c_in)).astype(np.float32)
    conv = Conv2D(c_in, c_out, k, jax.random.PRNGKey(0))
    conv.b.value = jnp.zeros_like(conv.b.value)
    y_shape = conv.forward(jnp.asarray(x[None])).shape
    y = rng.normal(0.0, 1.0, y_shape).astype(np.float32)
    tconv = ConvTranspose2D(c_out, c_in, k, jax.random.PRNGKey(0))
    tconv.W.value = jnp.transpose(conv.W.value, (0, 1, 3, 2))
    tconv.b.value = jnp.zeros_like(tconv.b.value)
    lhs = float(jnp.sum(conv.forward(jnp.asarray(x[None])) * jnp.asarray(y)))
    rhs = float(jnp.sum(jnp.asarray(x[None]) * tconv.forward(jnp.asarray(y))))
    print(f"batch adjoint: lhs={lhs:.4f} rhs={rhs:.4f} diff={abs(lhs - rhs):.3e}")
    assert abs(lhs - rhs) < 1e-4
