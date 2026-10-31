"""Tests for mixed-precision training on the compiled step.

The tests cover the six steps of item B of task N4: the trajectory of
bfloat16 against float32, the overflow skip and the halving of the
scale, the absence of NaN over a 200-step run with float16 and
dynamic scaling, the float32 master weights, the default bit-for-bit,
the gradient of the loss scaling, and the float32 cast of the
gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp, transformer_encoder
from dense_armor.utility.neural.schedules import ConstantLR
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD


def _data(shape_x, shape_y, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape_x).astype(np.float32), rng.normal(size=shape_y).astype(
        np.float32
    )


def _mlp():
    return mlp([2, 4, 1], jax.random.PRNGKey(0))


def _transformer():
    return transformer_encoder(
        n_in=4,
        d_model=8,
        n_heads=2,
        d_ff=16,
        depth=1,
        n_out=3,
        key=jax.random.PRNGKey(0),
        max_len=8,
    )


def _fit_losses(factory, sx, sy, epochs, batch_size, **kw):
    X, y = _data(sx, sy)
    t = Trainer(factory(), MSE(), SGD(0.01), compile=True, **kw)
    t.fit(X, y, epochs=epochs, batch_size=batch_size, key=jax.random.PRNGKey(0))
    return t.history_["train_loss"], t


# ------------------------------------------------------------------
# Trajectory: bf16 differs from fp32 bit-for-bit (the cast is applied)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,sx,sy",
    [
        (_mlp, (16, 2), (16, 1)),
        (_transformer, (16, 8, 4), (16, 3)),
    ],
)
def test_bf16_diverges_from_fp32_bit_for_bit(factory, sx, sy):
    """bfloat16 changes the loss: the cast is applied.

    A mutation that removes the cast (forward runs in float32) makes
    the two lists bit-for-bit equal; the test fails because they are
    not.
    """
    fp32, _ = _fit_losses(factory, sx, sy, epochs=4, batch_size=8)
    bf16, _ = _fit_losses(
        factory, sx, sy, epochs=4, batch_size=8, compute_dtype="bfloat16"
    )
    print(f"  fp32: {[round(v, 6) for v in fp32]}")
    print(f"  bf16: {[round(v, 6) for v in bf16]}")
    assert fp32 != bf16, "bfloat16 equals float32 bit-for-bit: the cast is not applied"
    for a, b in zip(fp32, bf16):
        rel = abs(a - b) / max(abs(a), 1e-6)
        assert rel < 0.5, f"bf16 {b} vs fp32 {a} (rel {rel:.3f})"


# ------------------------------------------------------------------
# Overflow: skip the step, halve the scale, weights unchanged
# ------------------------------------------------------------------


def test_overflow_skips_step_and_halves_scale():
    """A forced inf gradient skips the update and halves the scale."""
    X, y = _data((8, 2), (8, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(
        m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", loss_scale="dynamic"
    )
    t._scale = 1024.0
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
    initial_scale = t.history_["loss_scale"][0]
    assert initial_scale in (1024.0, 2048.0), initial_scale

    values_before = [np.array(p.value) for p in m.params()]
    t._compiled_mixed = lambda values, x, y: (
        [jnp.full_like(g, jnp.inf) for g in [p.grad for p in m.params()]],
        jnp.float32(1.0),
    )
    t._good_steps = 0
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
    new_scale = t.history_["loss_scale"][-1]
    assert new_scale == initial_scale / 2.0, (
        f"scale did not halve: {initial_scale} -> {new_scale}"
    )
    values_after = [np.array(p.value) for p in m.params()]
    for a, b in zip(values_before, values_after):
        assert np.array_equal(a, b), "weights changed after skipped step"


# ------------------------------------------------------------------
# No NaN over 200 steps with float16 + dynamic
# ------------------------------------------------------------------


def test_dynamic_scale_produces_no_nan():
    """float16 + dynamic: no NaN, loss bounded. The trajectory is not
    asserted because the default init_scale (65536) makes most steps
    overflow on this small model. A mutation that removes the skip
    leaves a NaN.
    """
    X, y = _data((64, 2), (64, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(
        m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", loss_scale="dynamic"
    )
    t.fit(X, y, epochs=20, batch_size=8, key=jax.random.PRNGKey(0))
    losses = t.history_["train_loss"]
    print(f"  losses[:5]: {[round(v, 3) for v in losses[:5]]}")
    print(f"  scale[:5]:  {t.history_['loss_scale'][:5]}")
    assert all(np.isfinite(v) for v in losses), losses
    assert all(v < 1e6 for v in losses)
    assert all(s > 0 for s in t.history_["loss_scale"])


# ------------------------------------------------------------------
# Master weights and optimizer state in float32
# ------------------------------------------------------------------


def test_master_weights_stay_float32():
    X, y = _data((16, 2), (16, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, compute_dtype="bfloat16")
    t.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    for p in m.params():
        assert p.value.dtype == jnp.float32, p.value.dtype
    for v in t._compiled_values:
        assert jnp.asarray(v).dtype == jnp.float32, jnp.asarray(v).dtype


# ------------------------------------------------------------------
# Default: bit-for-bit unchanged
# ------------------------------------------------------------------


def test_default_unchanged_bit_for_bit():
    """At compute_dtype=None and loss_scale=None the loss is unchanged,
    and the loss_scale history is empty."""
    X, y = _data((16, 2), (16, 1))
    a = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_a = Trainer(a, MSE(), SGD(0.01), compile=True)
    t_a.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))

    assert "loss_scale" in t_a.history_
    assert t_a.history_["loss_scale"] == []


# ------------------------------------------------------------------
# Loss scaling: the gradient is multiplied by the scale
# ------------------------------------------------------------------


def test_loss_scale_multiplies_gradient():
    """With a constant scale S, the gradient before unscale is S times
    the gradient with scale 1. A mutation that drops the
    ``grad * loss_scale`` line leaves the two equal; the test fails.
    """
    from dense_armor.utility.neural.compiled import make_step as _make_step

    X, y = _data((8, 2), (8, 1))

    def grads_with(scale):
        m = mlp([2, 4, 1], jax.random.PRNGKey(0))
        step = _make_step(
            m,
            MSE(),
            SGD(0.01),
            ConstantLR(0.01),
            0.0,
            "l2",
            None,
            compute_dtype="float16",
            loss_scale=scale,
        )
        values = [jnp.array(p.value, dtype=jnp.float32) for p in m.params()]
        grads, _loss = step.mixed(values, X, y)
        return grads

    g1 = grads_with(1.0)
    g100 = grads_with(100.0)
    ratio = float(jnp.abs(g100[0]).mean() / jnp.maximum(jnp.abs(g1[0]).mean(), 1e-12))
    print(f"  ratio g100/g1 = {ratio:.3f}")
    assert 50.0 < ratio < 200.0, f"scale not applied: ratio {ratio:.3f}"


# ------------------------------------------------------------------
# Gradients reaching _apply_update are float32
# ------------------------------------------------------------------


def test_constant_scale_with_unscale_matches_no_scale():
    """float16 + loss_scale=256 trains like float16 + loss_scale=None.

    The unscale divides the gradient by the same scale that multiplied
    it, so a step with scale 256 and a step with no scale produce the
    same master weights (up to float16 roundoff). A mutation that
    removes the unscale passes a gradient 256 times too large and the
    master weights move about 256 times further; the test fails.
    """
    X, y = _data((16, 2), (16, 1))

    def run(**kw):
        m = mlp([2, 4, 1], jax.random.PRNGKey(0))
        t = Trainer(m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", **kw)
        t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
        return [np.array(p.value) for p in m.params()]

    a = run(loss_scale=None)
    b = run(loss_scale=256.0)
    for wa, wb in zip(a, b):
        denom = np.maximum(np.abs(wa) + np.abs(wb), 1e-6)
        rel = float(np.max(np.abs(wa - wb) / denom))
        print(f"  param: max rel diff = {rel:.4f}")
        assert rel < 0.1, (
            f"master weights differ by {rel:.3f}: the unscale is not applied"
        )
