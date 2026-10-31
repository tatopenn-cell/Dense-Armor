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
from dense_armor.utility.optim import SGD, Adam

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
def test_bf16_trajectory_follows_fp32(factory, sx, sy):
    """The loss of bfloat16 against float32 at steps 1, 10, 50 and 200.

    One step per epoch (the batch is the whole set), same seed. The
    bfloat16 losses stay within 5% relative of the float32 losses, and
    differ from them at step 200: the cast is applied.
    """
    X, y = _data(sx, sy)

    def run(**kw):
        m = factory()
        t = Trainer(m, MSE(), SGD(0.01), compile=True, **kw)
        t.fit(X, y, epochs=200, batch_size=len(X), key=jax.random.PRNGKey(0))
        return t.history_["train_loss"]

    fp32 = run()
    bf16 = run(compute_dtype="bfloat16")
    for step in (1, 10, 50, 200):
        a, b = fp32[step - 1], bf16[step - 1]
        rel = abs(a - b) / abs(a)
        print(f"  step {step}: fp32 {a:.6f}  bf16 {b:.6f}  rel {rel:.5f}")
        assert rel <= 0.05, (step, a, b, rel)
    assert fp32[199] != bf16[199], "bfloat16 equals float32: the cast is not applied"


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


STORED_MASTER_LOSSES_X64_TRUE = [
    1.8287844438741407,
    1.4960505551335324,
    1.325572929081289,
]
STORED_MASTER_LOSSES_X64_FALSE = [
    4.009725093841553,
    3.231865406036377,
    2.795619010925293,
]


def test_default_matches_the_values_stored_from_master():
    """At compute_dtype=None and loss_scale=None the compiled trainer gives
    the losses stored from master.

    The values are the three epoch losses of the MLP [2, 4, 1] with seed
    0, MSE, SGD(0.01), compile=True, batch 8 on 16 samples of
    ``_data((16, 2), (16, 1))``, shuffle key 0, taken by running master
    before this branch. The comparison is relative, 1e-9 with x64 on and
    1e-5 with x64 off, because the last digit changes with the version
    of Python. The keys of ``history_`` are the three of master.
    """
    X, y = _data((16, 2), (16, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
    if jax.config.jax_enable_x64:
        expected, tol = STORED_MASTER_LOSSES_X64_TRUE, 1e-9
    else:
        expected, tol = STORED_MASTER_LOSSES_X64_FALSE, 1e-5
    assert t.history_["train_loss"] == pytest.approx(expected, rel=tol)
    assert list(t.history_) == ["train_loss", "val_loss", "lr"]


def test_loss_scale_without_compute_dtype_trains_like_the_plain_step():
    X, y = _data((64, 2), (64, 1))

    def run(**kw):
        m = _mlp()
        t = Trainer(m, MSE(), SGD(0.01), compile=True, **kw)
        t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
        return t

    plain = run().history_["train_loss"]
    const = run(loss_scale=256.0)
    dyn = run(loss_scale="dynamic")
    assert const.history_["train_loss"] == pytest.approx(plain, rel=1e-4)
    assert dyn.history_["train_loss"] == pytest.approx(plain, rel=1e-4)
    assert const.history_["loss_scale"] == [256.0] * 5
    assert dyn.history_["loss_scale"] == [65536.0] * 5


@pytest.mark.parametrize(
    "kw",
    [
        {"compute_dtype": "bfloat16"},
        {"loss_scale": 2.0},
        {"loss_scale": "dynamic"},
    ],
)
def test_mixed_precision_requires_compile(kw):
    with pytest.raises(ValueError, match="compile=True"):
        Trainer(_mlp(), MSE(), SGD(0.01), **kw)


def test_growth_step_uses_the_scale_that_multiplied_the_loss():
    """After 2000 good steps the scale doubles for the next step.

    The step that triggers the growth was computed with the old scale, so
    its weight update equals the one of a constant scale of the same
    value.
    """
    X, y = _data((8, 2), (8, 1))

    def one_step(good_steps=0, scale=None, **kw):
        m = _mlp()
        t = Trainer(m, MSE(), SGD(0.01), compile=True, compute_dtype="float16", **kw)
        if scale is not None:
            t._scale = scale
        t._good_steps = good_steps
        w0 = [np.array(p.value) for p in m.params()]
        t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
        return [np.array(p.value) - a for p, a in zip(m.params(), w0)], t

    d_const, _ = one_step(loss_scale=1024.0)
    d_grow, t_grow = one_step(good_steps=1999, scale=1024.0, loss_scale="dynamic")
    assert t_grow._scale == 2048.0
    assert t_grow._good_steps == 0
    for a, b in zip(d_const, d_grow):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)
    assert max(float(np.abs(d).max()) for d in d_grow) > 0.0


def test_optimizer_state_stays_float32():
    X, y = _data((16, 2), (16, 1))
    t = Trainer(_mlp(), MSE(), Adam(0.001), compile=True, compute_dtype="bfloat16")
    t.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    state = t._compiled_opt_state
    for a in state["m"] + state["v"]:
        assert jnp.asarray(a).dtype == jnp.float32, jnp.asarray(a).dtype
    assert jnp.asarray(state["t"]).dtype == jnp.int32
    assert int(state["t"]) == 4


def test_loss_scale_multiplies_the_stored_gradient_and_the_update_is_unscaled():
    """The float16 gradient is ``scale`` times larger; the float32 gradient
    handed to the optimizer is divided by the same ``scale``."""
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
        grads_fp16, grads_fp32, _ = step.mixed(values, X, y, scale)
        return grads_fp16, grads_fp32

    h1, g1 = grads_with(1.0)
    h100, g100 = grads_with(100.0)
    stored = float(jnp.abs(h100[0]).mean() / jnp.abs(h1[0]).mean())
    handed = float(jnp.abs(g100[0]).mean() / jnp.abs(g1[0]).mean())
    print(
        f"  stored fp16 ratio {stored:.3f}, fp32 ratio after the unscale {handed:.3f}"
    )
    assert stored == pytest.approx(100.0, rel=0.02)
    assert handed == pytest.approx(1.0, rel=0.02)


def test_clip_acts_on_the_unscaled_gradient():
    """The global norm is clipped after the unscale: a scale of 1024 and a
    scale of 1 give the same clipped update."""
    X, y = _data((16, 2), (16, 1))

    def run(scale):
        m = _mlp()
        t = Trainer(
            m,
            MSE(),
            SGD(0.01),
            compile=True,
            compute_dtype="float16",
            loss_scale=scale,
            clip_norm=0.05,
        )
        t.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
        return [np.array(p.value) for p in m.params()]

    for a, b in zip(run(1.0), run(1024.0)):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-6)


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


def _make(**kw):
    from dense_armor.utility.neural.compiled import make_step

    return make_step(_mlp(), MSE(), SGD(0.01), ConstantLR(0.01), 0.0, "l2", None, **kw)


def test_plain_make_step_has_no_mixed_step():
    step = _make()
    assert step.plain is not None
    assert step.mixed is None


def test_mixed_make_step_has_no_plain_step():
    X, y = _data((8, 2), (8, 1))
    values = [jnp.asarray(p.value) for p in _mlp().params()]
    for kw in ({"compute_dtype": "bfloat16"}, {"loss_scale": 1.0}):
        step = _make(**kw)
        assert step.mixed is not None
        assert step.plain is None
        with pytest.raises(TypeError, match="mixed"):
            step(values, None, X, y, 0.0)


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
