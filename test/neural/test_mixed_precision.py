"""Tests for mixed-precision training on the compiled step."""

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

FLOAT16_MIN = 2.0**-24


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


def _deep_sigmoid():
    return mlp([2] + [4] * 12 + [1], jax.random.PRNGKey(0), activation="sigmoid")


def test_underflow_float16_without_scale_is_zero_with_scale_is_not():
    """A deep sigmoid MLP has first-layer gradients below 2^-24 in
    float16; without loss scaling they become zero, with scale 8192
    they survive.
    """
    from dense_armor.utility.neural.compiled import make_step as _ms

    X, y = _data((16, 2), (16, 1))

    def first_grad(scale):
        m = _deep_sigmoid()
        step = _ms(
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
        grads_fp16, _, _ = step.mixed(values, X, y, scale)
        return np.asarray(grads_fp16[0])

    g_no_scale = first_grad(1.0)
    g_8192 = first_grad(8192.0)
    print(
        f"  no scale:   abs max = {abs(g_no_scale).max():.3e}, "
        f"nonzero = {int(np.sum(g_no_scale != 0))}"
    )
    print(
        f"  scale 8192: abs max = {abs(g_8192).max():.3e}, "
        f"nonzero = {int(np.sum(g_8192 != 0))}"
    )
    assert abs(g_no_scale).max() == 0.0, (
        f"the un-scaled gradient is {abs(g_no_scale).max():.3e}, not zero: "
        "the float16 storage of the gradients is not in place"
    )
    assert abs(g_8192).max() > FLOAT16_MIN, (
        f"with scale 8192 the gradient is {abs(g_8192).max():.3e}: "
        "the loss scaling is not applied"
    )


@pytest.mark.parametrize(
    "factory,sx,sy",
    [
        (_mlp, (16, 2), (16, 1)),
        (_transformer, (16, 8, 4), (16, 3)),
    ],
)
def test_bf16_diverges_from_fp32_bit_for_bit(factory, sx, sy):
    """bfloat16 changes the loss because x and W are cast to bfloat16.

    The target is not cast: the loss is computed in float32 as
    Section 3.2 page 5 of the paper asks. The only source of divergence
    is the cast of the input and of the master weights. A mutation that
    removes either cast leaves the two lists bit-for-bit equal; the
    test fails.
    """
    X, y = _data(sx, sy)

    def run(**kw):
        m = factory()
        t = Trainer(m, MSE(), SGD(0.01), compile=True, **kw)
        t.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
        return t.history_["train_loss"]

    fp32 = run()
    bf16 = run(compute_dtype="bfloat16")
    print(f"  fp32: {[round(v, 6) for v in fp32]}")
    print(f"  bf16: {[round(v, 6) for v in bf16]}")
    assert fp32 != bf16, "bfloat16 equals float32: the cast is not applied"


def test_overflow_skips_step_and_halves_scale():
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
    t._compiled_mixed = lambda values, x, y, scale: (
        [jnp.full_like(g, jnp.inf) for g in [p.grad for p in m.params()]],
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


def test_dynamic_scale_produces_no_nan_and_trains():
    """200 steps (25 epochs of 8 batches) of float16 + dynamic.

    With the loss scale passed at runtime (not captured as a constant in
    the jitted graph), the scale adapts and the loss decreases over 25
    epochs. A mutation that captures the scale as a constant leaves the
    scale stuck and the loss frozen; the test fails.
    """
    X, y = _data((64, 2), (64, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(
        m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", loss_scale="dynamic"
    )
    t.fit(X, y, epochs=25, batch_size=8, key=jax.random.PRNGKey(0))
    losses = t.history_["train_loss"]
    scales = t.history_["loss_scale"]
    print(f"  epochs (of 8 steps): {len(losses)}")
    print(f"  losses[:3]: {[round(v, 4) for v in losses[:3]]}")
    print(f"  losses[-3:]: {[round(v, 4) for v in losses[-3:]]}")
    print(f"  scale[:3]: {[float(s) for s in scales[:3]]}")
    print(f"  scale[-3:]: {[float(s) for s in scales[-3:]]}")
    assert len(losses) == 25, f"expected 25 epochs, got {len(losses)}"
    assert all(np.isfinite(v) for v in losses), losses
    assert all(s > 0 for s in scales)
    assert losses[-1] < losses[0] * 0.9, (
        f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_master_weights_stay_float32():
    X, y = _data((16, 2), (16, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, compute_dtype="bfloat16")
    t.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    for p in m.params():
        assert p.value.dtype == jnp.float32, p.value.dtype
    for v in t._compiled_values:
        assert jnp.asarray(v).dtype == jnp.float32, jnp.asarray(v).dtype


def test_default_unchanged_bit_for_bit():
    """At compute_dtype=None and loss_scale=None the compiled path gives
    the same loss as the loop path, on the same data.

    The existing test test_compiled_step.py::test_default_unchanged_bit_for_bit
    compares compile=False against the values stored from master and
    passes on this branch. Here the comparison is compile=True (with the
    mixed-precision defaults that must not change anything) against
    compile=False, with the same data. The tolerance is the one of that
    test: 1e-9 with x64 on, 1e-5 with x64 off, relative.
    """
    X, y = _data((16, 2), (16, 1))
    a = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_a = Trainer(
        a,
        MSE(),
        SGD(0.01),
        compile=True,
        compute_dtype=None,
        loss_scale=None,
    )
    t_a.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
    assert t_a.history_["loss_scale"] == []

    b = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_b = Trainer(
        b,
        MSE(),
        SGD(0.01),
        compile=False,
        compute_dtype=None,
        loss_scale=None,
    )
    t_b.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))

    tol = 1e-9 if jax.config.jax_enable_x64 else 1e-5
    losses_a = t_a.history_["train_loss"]
    losses_b = t_b.history_["train_loss"]
    print(f"  compile=True  (default): {[round(v, 6) for v in losses_a]}")
    print(f"  compile=False (default): {[round(v, 6) for v in losses_b]}")
    assert len(losses_a) == len(losses_b) == 3
    for got, want in zip(losses_a, losses_b):
        rel = abs(got - want) / max(abs(want), 1e-12)
        assert rel <= tol, f"compile=True {got!r} != compile=False {want!r} (rel {rel})"


def test_loss_scale_multiplies_gradient():
    from dense_armor.utility.neural.compiled import make_step as _ms

    X, y = _data((8, 2), (8, 1))

    def grads_with(scale):
        m = mlp([2, 4, 1], jax.random.PRNGKey(0))
        step = _ms(
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
        _, grads_fp32, _ = step.mixed(values, X, y, scale)
        return grads_fp32

    g1 = grads_with(1.0)
    g100 = grads_with(100.0)
    ratio = float(jnp.abs(g100[0]).mean() / jnp.maximum(jnp.abs(g1[0]).mean(), 1e-12))
    print(f"  ratio g100/g1 = {ratio:.3f}")
    assert 50.0 < ratio < 200.0, ratio


def test_grads_are_float32_after_mixed_step():
    from dense_armor.utility.neural.compiled import make_step as _ms

    X, y = _data((8, 2), (8, 1))
    m = _deep_sigmoid()
    step = _ms(
        m,
        MSE(),
        SGD(0.01),
        ConstantLR(0.01),
        0.0,
        "l2",
        None,
        compute_dtype="float16",
        loss_scale=8192.0,
    )
    values = [jnp.array(p.value, dtype=jnp.float32) for p in m.params()]
    _, grads_fp32, _ = step.mixed(values, X, y, 8192.0)
    for g in grads_fp32:
        assert jnp.asarray(g).dtype == jnp.float32, jnp.asarray(g).dtype


def test_unscale_makes_constant_scale_match_no_scale():
    """float16 + loss_scale=64 trains like float16 + loss_scale=None.

    A small MLP, so no step overflows and the master weights change.
    The unscale divides the gradient by the same 64 that multiplied
    it, so the master weights after one step match up to float16
    roundoff. A mutation that removes the unscale passes a gradient 64
    times too large and the weights move about 64 times further; the
    test fails.
    """
    X, y = _data((16, 2), (16, 1))

    def run(**kw):
        m = mlp([2, 4, 1], jax.random.PRNGKey(0))
        t = Trainer(m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", **kw)
        t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
        return [np.array(p.value) for p in m.params()]

    a = run(loss_scale=None)
    b = run(loss_scale=64.0)
    for wa, wb in zip(a, b):
        denom = np.maximum(np.abs(wa) + np.abs(wb), 1e-6)
        rel = float(np.max(np.abs(wa - wb) / denom))
        print(f"  param: max rel diff = {rel:.4f}")
        assert rel < 0.1, (
            f"master weights differ by {rel:.3f}: the unscale is not applied"
        )


def test_weights_and_input_are_cast_to_compute_dtype_in_forward():
    """During the mixed forward, both the master weights and the input
    are in the compute dtype.

    A mutation that removes the cast of the weights leaves them in
    float32 during the forward; a mutation that removes the cast of the
    input leaves the input in float32. The spy records the dtypes that
    the forward receives and uses; the test fails on either mutation.
    """
    from dense_armor.utility.neural.compiled import make_step as _ms

    X, y = _data((8, 2), (8, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    seen = {"weight_dtypes": [], "input_dtypes": []}
    orig = m.forward

    def spy(x, training=True):
        seen["input_dtypes"].append(jnp.asarray(x).dtype)
        seen["weight_dtypes"].extend(p.value.dtype for p in m.params())
        return orig(x, training=training)

    m.forward = spy
    step = _ms(
        m,
        MSE(),
        SGD(0.01),
        ConstantLR(0.01),
        0.0,
        "l2",
        None,
        compute_dtype="bfloat16",
        loss_scale=1.0,
    )
    values = [jnp.array(p.value, dtype=jnp.float32) for p in m.params()]
    step.mixed(values, X, y, 1.0)
    print(f"  input dtypes: {[str(d) for d in seen['input_dtypes']]}")
    print(f"  weight dtypes: {[str(d) for d in seen['weight_dtypes']]}")
    assert all(d == jnp.bfloat16 for d in seen["input_dtypes"]), (
        "the input is not cast to bfloat16"
    )
    assert all(d == jnp.bfloat16 for d in seen["weight_dtypes"]), (
        "the weights are not cast to bfloat16"
    )


def test_read_write_values_with_dtype():
    """_read_values and _write_values with an explicit dtype."""
    import jax.random

    from dense_armor.utility.neural.compiled import _read_values, _write_values
    from dense_armor.utility.neural.models import mlp

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    values = _read_values(m, dtype=jnp.float16)
    for v in values:
        assert jnp.asarray(v).dtype == jnp.float16
    _write_values(m, values, dtype=jnp.bfloat16)
    for p_ in m.params():
        assert p_.value.dtype == jnp.bfloat16
    values2 = _read_values(m)
    for v in values2:
        assert jnp.asarray(v).dtype == jnp.bfloat16


def test_invalid_compute_dtype_raises():
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.trainer import Trainer

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="compute_dtype"):
        Trainer(m, MSE(), SGD(0.01), compute_dtype="float64")


def test_invalid_loss_scale_string_raises():
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.trainer import Trainer

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="loss_scale"):
        Trainer(m, MSE(), SGD(0.01), loss_scale="constant")


def test_negative_loss_scale_raises():
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.trainer import Trainer

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="loss_scale"):
        Trainer(m, MSE(), SGD(0.01), loss_scale=-1.0)


def test_non_mixed_compiled_returns_plain_step():
    """At compute_dtype=None and loss_scale=1.0 the make_step returns
    the plain jitted step, not the mixed one."""
    from dense_armor.utility.neural.compiled import make_step
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.schedules import ConstantLR
    from dense_armor.utility.optim import SGD

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    step = make_step(
        m,
        MSE(),
        SGD(0.01),
        ConstantLR(0.01),
        0.0,
        "l2",
        None,
        compute_dtype=None,
        loss_scale=1.0,
    )
    assert step.plain is not None
    assert step.mixed is None


def test_mixed_step_returns_mixed_attribute():
    from dense_armor.utility.neural.compiled import make_step
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.schedules import ConstantLR
    from dense_armor.utility.optim import SGD

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    step = make_step(
        m,
        MSE(),
        SGD(0.01),
        ConstantLR(0.01),
        0.0,
        "l2",
        None,
        compute_dtype="bfloat16",
        loss_scale=1.0,
    )
    assert hasattr(step, "mixed")


def test_invalid_compute_dtype_in_make_step_raises():
    from dense_armor.utility.neural.compiled import make_step
    from dense_armor.utility.neural.models import mlp
    from dense_armor.utility.neural.schedules import ConstantLR
    from dense_armor.utility.optim import SGD

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="compute_dtype"):
        make_step(
            m,
            MSE(),
            SGD(0.01),
            ConstantLR(0.01),
            0.0,
            "l2",
            None,
            compute_dtype="float64",
            loss_scale=1.0,
        )
