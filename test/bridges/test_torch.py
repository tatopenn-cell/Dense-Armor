"""Tests for the PyTorch bridge: tensors, dtype, pytree, no-copy."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dense_armor.bridges.torch import to_jax, to_torch


def test_tensor_round_trip_float32():
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    a = to_jax(t)
    t2 = to_torch(a)
    assert t2.dtype == t.dtype
    assert tuple(t2.shape) == tuple(t.shape)
    assert torch.equal(t, t2)


def test_tensor_round_trip_int32():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    a = to_jax(t)
    t2 = to_torch(a)
    assert t2.dtype == t.dtype
    assert torch.equal(t, t2)


def test_tensor_round_trip_bool():
    t = torch.tensor([True, False, True], dtype=torch.bool)
    a = to_jax(t)
    t2 = to_torch(a)
    assert t2.dtype == t.dtype
    assert torch.equal(t, t2)


def test_no_hidden_float64():
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    a = to_jax(t)
    assert a.dtype == jnp.float32


def test_contiguous_same_pointer():
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    a = to_jax(t)
    t2 = to_torch(a)
    assert t.data_ptr() == t2.data_ptr()


def test_non_contiguous_is_copied():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).t()
    assert not t.is_contiguous()
    a = to_jax(t)
    assert a.shape == (3, 2)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(t.contiguous()))


def test_requires_grad_is_detached():
    t = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    a = to_jax(t)
    assert a.dtype == jnp.float32


def test_pytree_round_trip():
    tree = {
        "a": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "b": [torch.tensor([3.0], dtype=torch.float32)],
    }
    a = to_jax(tree)
    assert isinstance(a, dict)
    assert isinstance(a["b"], list)
    assert a["a"].dtype == jnp.float32
    back = to_torch(a)
    assert isinstance(back, dict)
    assert back["a"].dtype == torch.float32
    assert torch.equal(back["a"], tree["a"])
    assert torch.equal(back["b"][0], tree["b"][0])


def test_non_tensor_passthrough():
    assert to_jax(3) == 3
    assert to_torch(3) == 3
    assert to_jax("x") == "x"
