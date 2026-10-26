"""Structural tests for the ResidualTree speed fix."""

import numpy as np
import jax.numpy as jnp
import pytest

from dense_armor.utility.neural.consolidation import (
    ArchivalMemory,
    ResidualTree,
    VQMemory,
)


def test_vq_fit_reuses_optimizers():
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    xs = jnp.asarray(np.random.default_rng(0).normal(size=(8, 2)), dtype=jnp.float32)
    vq.fit(xs, n_steps=1)
    assert hasattr(vq, "_opt_enc"), "fit should create _opt_enc on the first call"
    enc_id = id(vq._opt_enc)
    dec_id = id(vq._opt_dec)
    vq.fit(xs, n_steps=1)
    assert id(vq._opt_enc) == enc_id, "fit re-created the encoder optimizers"
    assert id(vq._opt_dec) == dec_id, "fit re-created the decoder optimizers"


def test_residualtree_writes_fit_every_8_samples():
    # kappa very large: every insert has r <= kappa and lands on the root,
    # so the root's support reaches 8 after exactly 8 inserts.
    tree = ResidualTree(
        d_model=2, n_codes=8, max_depth=2, branching=4, kappa=1e9, seed=0
    )
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(8, 2)).astype(np.float32)
    for i in range(7):
        tree.insert(jnp.asarray(xs[i]))
    trained_after_7 = hasattr(tree.root["vq"], "_opt_enc")
    tree.insert(jnp.asarray(xs[7]))
    trained_after_8 = hasattr(tree.root["vq"], "_opt_enc")
    assert not trained_after_7, "fit was called before the 8th sample"
    assert trained_after_8, "fit was not called on the 8th sample"


def test_archival_memory_router_default_is_linear():
    am = ArchivalMemory(d_model=2, n_codes=8, max_depth=2, branching=4, seed=0)
    assert am.router == "linear"
    assert am.tree.router == "linear"


def test_archival_memory_router_mondrian_is_passed():
    am = ArchivalMemory(
        d_model=2, n_codes=8, max_depth=2, branching=4, seed=0, router="mondrian"
    )
    assert am.router == "mondrian"
    assert am.tree.router == "mondrian"


def test_archival_memory_router_invalid_raises():
    with pytest.raises(ValueError, match="router must be"):
        ArchivalMemory(d_model=2, router="invalid")
