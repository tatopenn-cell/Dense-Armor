"""The gating noise of SparseMoE: ``noise_std`` is used, and only in training."""

import jax.numpy as jnp
import numpy as np

from dense_armor.core.deep.neural.attention.moe import SparseMoE


def _moe(**kw):
    return SparseMoE(4, 3, n_experts=2, top_k=1, hidden=4, capacity_factor=None, **kw)


def test_no_noise_at_inference():
    x = jnp.ones((6, 4))
    plain = _moe(noisy_gating=False).forward(x)
    noisy = _moe(noisy_gating=True, noise_std=5.0)
    np.testing.assert_allclose(noisy.forward(x, training=False), plain)


def test_noise_std_changes_the_output_in_training():
    x = jnp.ones((6, 4))
    plain = _moe(noisy_gating=False).forward(x)
    noisy = _moe(noisy_gating=True, noise_std=5.0).forward(x, training=True)
    assert not np.allclose(noisy, plain)


def test_zero_noise_equals_the_plain_gating():
    x = jnp.ones((6, 4))
    plain = _moe(noisy_gating=False).forward(x)
    zero = _moe(noisy_gating=True, noise_std=0.0).forward(x, training=True)
    np.testing.assert_allclose(zero, plain)
