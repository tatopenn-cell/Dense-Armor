"""Tests for norm_more.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.norm_more import GroupNorm, InstanceNorm


def _hand_x():
    return jnp.array([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])


def test_groupnorm_hand_case():
    y = np.asarray(GroupNorm(2, 4).forward(_hand_x())).reshape(-1)
    expected = [
        -1.212677,
        -0.727606,
        -1.212677,
        -0.727606,
        0.727606,
        1.212677,
        0.727606,
        1.212677,
    ]
    print(f"GroupNorm(2,4): {y.tolist()}")
    assert np.allclose(y, expected, atol=1e-5)


def test_instancenorm_hand_case():
    y = np.asarray(InstanceNorm(4).forward(_hand_x())).reshape(-1)
    expected = [-0.999999] * 4 + [0.999999] * 4
    print(f"InstanceNorm(4): {y.tolist()}")
    assert np.allclose(y, expected, atol=1e-5)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
def test_group_mean_zero_var_one(num_groups):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(0.0, 1.0, (3, 5, 4, 8)).astype(np.float32))
    gn = GroupNorm(num_groups, 8)
    y = np.asarray(gn.forward(x))
    G = num_groups
    Cg = 8 // G
    yr = y.reshape(3, 5 * 4, G, Cg)
    mu = yr.mean(axis=(1, 3))
    var = yr.var(axis=(1, 3))
    print(
        f"G={G}: max|mu|={np.max(np.abs(mu)):.2e}, max|var-1/(1+eps)|={np.max(np.abs(var - 1.0 / (1 + 1e-5))):.2e}"
    )
    assert np.allclose(mu, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0 / (1.0 + 1e-5), atol=1e-4)


def test_groupnorm_1_group_equals_global():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 3)).astype(np.float32))
    gn = GroupNorm(1, 3)
    y = np.asarray(gn.forward(x))
    x_np = np.asarray(x)
    mu = x_np.mean(axis=(0, 1, 2), keepdims=True)
    var = x_np.var(axis=(0, 1, 2), keepdims=True)
    ref = (x_np - mu) / np.sqrt(var + 1e-5)
    print(f"GroupNorm(1,C) vs manual: max diff {np.max(np.abs(y - ref)):.2e}")
    assert np.allclose(y, ref, atol=1e-5)


def test_groupnorm_C_equals_instancenorm():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 5, 3)).astype(np.float32))
    y_gn = np.asarray(GroupNorm(3, 3).forward(x))
    y_in = np.asarray(InstanceNorm(3).forward(x))
    print(
        f"GroupNorm(C,C) vs InstanceNorm(C): max diff {np.max(np.abs(y_gn - y_in)):.2e}"
    )
    assert np.allclose(y_gn, y_in, atol=1e-6)


def test_batch_independence():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 1, 3)).astype(np.float32))
    gn = GroupNorm(1, 3)
    y = np.asarray(gn.forward(x))
    x2 = x.at[0].set(x[0] + 10.0)
    y2 = np.asarray(gn.forward(x2))
    print(f"sample 1 diff: {np.max(np.abs(y[1] - y2[1])):.2e}")
    assert np.allclose(y[1], y2[1], atol=1e-6)


def test_gradients_against_jax_grad():
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 3)).astype(np.float32))
    g = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 3)).astype(np.float32))

    gn = GroupNorm(1, 3)
    y = gn.forward(x)
    dx = np.asarray(gn.backward(g))

    def loss(xv):
        return jnp.sum(GroupNorm(1, 3).forward(xv) * g)

    dx_ref = np.asarray(jax.grad(loss)(x))
    print(f"GroupNorm dx max diff: {np.max(np.abs(dx - dx_ref)):.2e}")
    assert np.allclose(dx, dx_ref, atol=1e-4)

    inn = InstanceNorm(3)
    _ = inn.forward(x)
    dx2 = np.asarray(inn.backward(g))

    def loss2(xv):
        return jnp.sum(InstanceNorm(3).forward(xv) * g)

    dx2_ref = np.asarray(jax.grad(loss2)(x))
    print(f"InstanceNorm dx max diff: {np.max(np.abs(dx2 - dx2_ref)):.2e}")
    assert np.allclose(dx2, dx2_ref, atol=1e-4)


def test_gamma_beta_receive_grad():
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 3)).astype(np.float32))
    g = jnp.asarray(rng.normal(0.0, 1.0, (2, 4, 3)).astype(np.float32))
    gn = GroupNorm(1, 3)
    _ = gn.forward(x)
    _ = gn.backward(g)
    print(
        f"gamma.grad shape: {gn.gamma.grad.shape}, beta.grad shape: {gn.beta.grad.shape}"
    )
    assert gn.gamma.grad.shape == (3,)
    assert gn.beta.grad.shape == (3,)


def test_squeeze_and_batch_match():
    rng = np.random.default_rng(6)
    x = jnp.asarray(rng.normal(0.0, 1.0, (4, 1, 3)).astype(np.float32))
    gn = GroupNorm(1, 3)
    y_single = np.asarray(gn.forward(x))
    gn2 = GroupNorm(1, 3)
    y_batch = np.asarray(gn2.forward(x[None]))
    assert y_single.shape == (4, 1, 3)
    assert y_batch.shape == (1, 4, 1, 3)
    assert np.allclose(y_single, y_batch[0], atol=1e-6)


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        GroupNorm(2, 4).backward(jnp.ones((2, 1, 4)))
    with pytest.raises(RuntimeError):
        InstanceNorm(4).backward(jnp.ones((2, 1, 4)))


def test_value_errors_groupnorm():
    with pytest.raises(ValueError):
        GroupNorm(0, 4)
    with pytest.raises(ValueError):
        GroupNorm(2, 0)
    with pytest.raises(ValueError):
        GroupNorm(3, 4)
    with pytest.raises(ValueError):
        GroupNorm(2, 4, eps=0.0)
    with pytest.raises(ValueError):
        GroupNorm(2, 4, spatial_ndim=3)
    with pytest.raises(ValueError):
        GroupNorm(2, 4).forward(jnp.ones((2, 4)))


def test_value_errors_instancenorm():
    with pytest.raises(ValueError):
        InstanceNorm(0)
    with pytest.raises(ValueError):
        InstanceNorm(4, eps=-1.0)
    with pytest.raises(ValueError):
        InstanceNorm(4, spatial_ndim=3)


def test_groupnorm_params_and_repr():
    gn = GroupNorm(2, 4)
    ps = gn.params()
    assert len(ps) == 2
    assert ps[0] is gn.gamma
    assert ps[1] is gn.beta
    s = repr(gn)
    assert "GroupNorm(2, 4" in s
    assert "eps=1e-05" in s
    assert "spatial_ndim=2" in s


def test_groupnorm_wrong_channel_count():
    gn = GroupNorm(2, 4)
    with pytest.raises(ValueError):
        gn.forward(jnp.ones((2, 2, 3)))


def test_instancenorm_params_repr_properties():
    inn = InstanceNorm(4)
    ps = inn.params()
    assert len(ps) == 2
    assert ps[0] is inn.gamma
    assert ps[1] is inn.beta
    assert inn.gamma is inn._gn.gamma
    assert inn.beta is inn._gn.beta
    s = repr(inn)
    assert "InstanceNorm(4" in s
    assert "eps=1e-05" in s
