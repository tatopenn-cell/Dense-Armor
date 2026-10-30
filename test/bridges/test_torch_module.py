"""Tests for TorchModule: a Dense-Armor model as torch.nn.Module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from dense_armor.bridges.torch import TorchModule
from dense_armor.utility.neural import Dense, Sequential
from dense_armor.utility.neural.activations import Relu


def _model(d_in=4, d_hidden=8, d_out=3):
    return Sequential(
        [
            Dense(d_in, d_hidden, key=jax.random.PRNGKey(0)),
            Relu(),
            Dense(d_hidden, d_out, key=jax.random.PRNGKey(1)),
        ]
    )


def test_forward_matches_jax():
    m = _model()
    tm = TorchModule(m)
    x = torch.randn(2, 4, dtype=torch.float32)
    y_t = tm(x)
    y_j = m.forward(jnp.asarray(x.numpy()), training=False)
    np.testing.assert_allclose(
        y_t.detach().numpy(), np.asarray(y_j), rtol=1e-5, atol=1e-6
    )


def test_params_are_torch_parameters():
    m = _model()
    tm = TorchModule(m)
    assert len(list(tm.parameters())) == 4
    for p in tm.parameters():
        assert isinstance(p, torch.nn.Parameter)
        assert p.requires_grad


def test_gradcheck_on_input():
    m = _model()
    tm = TorchModule(m)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

    # gradcheck vuole float64
    def tm64(z):
        return tm(z.float()).double()

    assert torch.autograd.gradcheck(tm64, (x,), eps=1e-4, atol=1e-3)


def test_grad_on_input_matches_jax():
    m = _model()
    tm = TorchModule(m)
    x_t = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    y = tm(x_t)
    y.sum().backward()
    g_torch = x_t.grad.numpy()

    x_j = jnp.asarray(x_t.detach().numpy())
    g_jax = jax.grad(lambda z: m.forward(z, training=False).sum())(x_j)
    np.testing.assert_allclose(g_torch, np.asarray(g_jax), rtol=1e-4, atol=1e-5)


def test_sgd_step_lowers_loss():
    m = _model()
    tm = TorchModule(m)
    x = torch.randn(16, 4, dtype=torch.float32)
    target = torch.randn(16, 3, dtype=torch.float32)
    opt = torch.optim.SGD(tm.parameters(), lr=0.1)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        y = tm(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop: {losses[0]} -> {losses[-1]}"
    )


def test_train_mode_reaches_dropout():
    """In eval mode the Dropout is identity; in train mode a mask is applied.

    The deep copy of the model at every forward resets the Dropout key,
    so two calls in train mode give the same mask. The test checks what
    is true: eval is identity, train applies a mask.
    """
    import jax

    from dense_armor.utility.neural import Dropout, Sequential

    m = Sequential([Dropout(0.5, key=jax.random.PRNGKey(0))])
    tm = TorchModule(m)
    x = torch.ones(100, 4, dtype=torch.float32)

    tm.eval()
    y_eval = tm(x)
    assert torch.equal(y_eval, x)

    tm.train()
    y_train = tm(x)
    assert not torch.equal(y_train, x)
