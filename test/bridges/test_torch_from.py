"""Tests for from_torch: a torch module inside Dense-Armor."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dense_armor.bridges.torch import from_torch


def test_apply_matches_torch():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 2)
    adapter = from_torch(module)
    x = torch.randn(3, 4)
    y_t = adapter.apply(x)
    y_direct = module(x)
    np.testing.assert_allclose(
        y_t.detach().numpy(), y_direct.detach().numpy(), rtol=1e-6, atol=1e-7
    )


def test_apply_on_jax_array():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 2)
    adapter = from_torch(module)
    x = jnp.zeros((3, 4), dtype=jnp.float32)
    y = adapter.apply(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (3, 2)


def test_grad_flows_through_module():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 2)
    adapter = from_torch(module)
    x = torch.randn(3, 4, requires_grad=True)
    y = adapter.apply(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_sequential_module():
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    )
    adapter = from_torch(module)
    x = torch.randn(5, 4)
    y = adapter.apply(x)
    assert tuple(y.shape) == (5, 2)
    np.testing.assert_allclose(
        y.detach().numpy(), module(x).detach().numpy(), rtol=1e-6, atol=1e-7
    )
