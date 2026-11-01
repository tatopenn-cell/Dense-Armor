"""Tests of the data stream and of the gradient checkpointing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.attention import TransformerBlock
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp, transformer_encoder
from dense_armor.utility.neural.stream import DataStream
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, Adam


def _tol():
    return 1e-9 if jax.config.jax_enable_x64 else 1e-5


def _arrays(n=23):
    X = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    y = np.arange(n)
    return X, y


def _collect(stream):
    ys = []
    sizes = []
    for xb, yb in stream:
        assert len(xb) == len(yb)
        ys.extend(int(v) for v in yb)
        sizes.append(len(yb))
    return ys, sizes


@pytest.mark.parametrize("buffer", [0, 1, 7, 23, 50])
@pytest.mark.parametrize("batch", [1, 5, 8, 23, 40])
def test_every_sample_is_served_exactly_once_per_epoch(buffer, batch):
    X, y = _arrays()
    stream = DataStream((X, y), batch, shuffle_buffer=buffer, key=jax.random.PRNGKey(0))
    for _ in range(2):
        ys, sizes = _collect(stream)
        assert sorted(ys) == list(range(23))
        assert sizes == [batch] * (23 // batch) + ([23 % batch] if 23 % batch else [])


def test_the_features_follow_their_labels_when_shuffled():
    X, y = _arrays()
    stream = DataStream((X, y), 4, shuffle_buffer=9, key=jax.random.PRNGKey(1))
    for xb, yb in stream:
        np.testing.assert_array_equal(xb[:, 0], yb * 3.0)


def test_without_a_buffer_the_order_is_the_order_of_the_source():
    X, y = _arrays()
    ys, _ = _collect(DataStream((X, y), 4))
    assert ys == list(range(23))


def test_a_buffer_of_one_sample_keeps_the_order():
    X, y = _arrays()
    stream = DataStream((X, y), 4, shuffle_buffer=1, key=jax.random.PRNGKey(0))
    ys, _ = _collect(stream)
    assert ys == list(range(23))


def test_a_buffer_changes_the_order_and_a_larger_buffer_mixes_more():
    X, y = _arrays(200)
    small = _collect(DataStream((X, y), 10, 5, jax.random.PRNGKey(0)))[0]
    large = _collect(DataStream((X, y), 10, 200, jax.random.PRNGKey(0)))[0]
    assert small != list(range(200))
    assert large != list(range(200))

    def displacement(order):
        return float(np.mean(np.abs(np.array(order) - np.arange(200))))

    assert displacement(large) > displacement(small) > 0.0


def test_the_same_key_gives_the_same_order_and_each_epoch_a_new_one():
    X, y = _arrays(60)
    a = DataStream((X, y), 6, 20, jax.random.PRNGKey(3))
    b = DataStream((X, y), 6, 20, jax.random.PRNGKey(3))
    a1, a2 = _collect(a)[0], _collect(a)[0]
    b1 = _collect(b)[0]
    assert a1 == b1
    assert a1 != a2
    assert sorted(a2) == list(range(60))


def test_a_memory_mapped_file_is_read_in_batches(tmp_path):
    X, y = _arrays(50)
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", y)
    stream = DataStream((tmp_path / "x.npy", str(tmp_path / "y.npy")), 16)
    assert isinstance(stream.X, np.memmap)
    assert isinstance(stream.y, np.memmap)
    got = [(xb, yb) for xb, yb in stream]
    assert [len(yb) for _, yb in got] == [16, 16, 16, 2]
    np.testing.assert_array_equal(np.concatenate([xb for xb, _ in got]), X)
    np.testing.assert_array_equal(np.concatenate([yb for _, yb in got]), y)


def test_a_shuffled_memory_map_serves_every_sample_once(tmp_path):
    X, y = _arrays(50)
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", y)
    stream = DataStream(
        (tmp_path / "x.npy", tmp_path / "y.npy"), 8, 12, jax.random.PRNGKey(0)
    )
    ys, _ = _collect(stream)
    assert sorted(ys) == list(range(50))


def test_a_generator_source_is_restarted_at_every_epoch():
    X, y = _arrays(11)

    def make():
        return ((X[i], y[i]) for i in range(11))

    stream = DataStream(make, 4)
    for _ in range(2):
        ys, sizes = _collect(stream)
        assert ys == list(range(11))
        assert sizes == [4, 4, 3]


def test_a_shuffled_generator_serves_every_sample_once():
    X, y = _arrays(31)
    stream = DataStream(lambda: zip(X, y), 5, 10, jax.random.PRNGKey(0))
    ys, _ = _collect(stream)
    assert sorted(ys) == list(range(31))
    assert ys != list(range(31))


def test_the_length_and_the_number_of_samples():
    X, y = _arrays(23)
    stream = DataStream((X, y), 5)
    assert len(stream) == 5
    assert stream.n_samples == 23
    generator = DataStream(lambda: zip(X, y), 5)
    assert generator.n_samples is None
    with pytest.raises(TypeError, match="not known"):
        len(generator)


@pytest.mark.parametrize(
    "args,match",
    [
        (dict(batch_size=0), "batch_size"),
        (dict(batch_size=4, shuffle_buffer=-1), "shuffle_buffer"),
        (dict(batch_size=4, shuffle_buffer=3), "key"),
    ],
)
def test_invalid_arguments_raise(args, match):
    with pytest.raises(ValueError, match=match):
        DataStream(_arrays(), **args)


def test_a_source_that_is_not_a_pair_or_a_function_raises():
    with pytest.raises(ValueError, match="pair"):
        DataStream(np.zeros((4, 2)), 2)


def test_x_and_y_of_different_length_raise():
    with pytest.raises(ValueError, match="different lengths"):
        DataStream((np.zeros((4, 2)), np.zeros(5)), 2)


STORED_MASTER_BLOCK_X64_TRUE = (
    79.99944916169923,
    106.86571140363814,
    610.2347613344646,
)
STORED_MASTER_BLOCK_X64_FALSE = (79.9996337890625, 65.77505493164062, 601.5282487869263)


def _block_numbers(block):
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (2, 5, 8))
    y = block.forward(x)
    dx = block.backward(g)
    grads = sum(float(jnp.sum(jnp.abs(p.grad))) for p in block.params())
    return float(jnp.sum(y * y)), float(jnp.sum(dx * dx)), grads


def test_the_default_block_gives_the_values_stored_from_master():
    block = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
    assert block.remat is False
    if jax.config.jax_enable_x64:
        expected = STORED_MASTER_BLOCK_X64_TRUE
    else:
        expected = STORED_MASTER_BLOCK_X64_FALSE
    assert _block_numbers(block) == pytest.approx(expected, rel=_tol())


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(5, 8), (2, 5, 8)])
def test_remat_gives_the_same_output_and_gradients(causal, shape):
    key = jax.random.PRNGKey(0)
    plain = TransformerBlock(8, 2, 16, key, causal=causal)
    remat = TransformerBlock(8, 2, 16, key, causal=causal, remat=True)
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    g = jax.random.normal(jax.random.PRNGKey(2), shape)
    y_plain, y_remat = plain.forward(x), remat.forward(x)
    dx_plain, dx_remat = plain.backward(g), remat.backward(g)
    tol = 1e-6 if jax.config.jax_enable_x64 else 1e-5
    np.testing.assert_allclose(y_remat, y_plain, rtol=tol, atol=tol)
    np.testing.assert_allclose(dx_remat, dx_plain, rtol=tol, atol=tol)
    for p, q in zip(plain.params(), remat.params()):
        np.testing.assert_allclose(q.grad, p.grad, rtol=tol, atol=tol)
        assert bool(jnp.array_equal(p.value, q.value))


def test_remat_keeps_no_activation_after_the_forward():
    block = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), remat=True)
    block.forward(jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8)))
    assert block._cache is None
    assert block.attn._cache is None
    assert block.ln1.xhat_ is None
    assert block.ln2.xhat_ is None
    assert block.ffn1.x_ is None
    assert block.ffn2.x_ is None
    assert block._vjp is not None


def test_a_second_backward_adds_to_the_gradients():
    block = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), remat=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
    block.forward(x)
    block.backward(g)
    once = [np.array(p.grad) for p in block.params()]
    block.backward(g)
    for a, p in zip(once, block.params()):
        np.testing.assert_allclose(np.array(p.grad), 2.0 * a, rtol=1e-5, atol=1e-6)


def test_remat_recomputes_the_forward_in_the_backward():
    """The traced forward and backward of a remat block contain a checkpoint
    and more matrix products than the plain block: the forward is run again."""
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (5, 8))

    def trace(remat):
        b = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), remat=remat)

        def run(xx, gg):
            return b.forward(xx), b.backward(gg)

        return str(jax.make_jaxpr(run)(x, g))

    plain, remat = trace(False), trace(True)
    assert "remat" not in plain
    assert "remat" in remat
    assert remat.count("dot_general") > plain.count("dot_general")


def test_remat_with_dropout_raises():
    with pytest.raises(ValueError, match="dropout"):
        TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), dropout=0.1, remat=True)


def test_remat_backward_before_forward_raises():
    block = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0), remat=True)
    with pytest.raises(RuntimeError, match="forward"):
        block.backward(jnp.zeros((5, 8)))


def test_the_encoder_builds_remat_blocks():
    m = transformer_encoder(4, 8, 2, 16, 3, 2, jax.random.PRNGKey(0), 8, remat=True)
    blocks = [layer for layer in m.layers if isinstance(layer, TransformerBlock)]
    assert len(blocks) == 3
    assert all(b.remat for b in blocks)
    m0 = transformer_encoder(4, 8, 2, 16, 3, 2, jax.random.PRNGKey(0), 8)
    assert not any(b.remat for b in m0.layers if isinstance(b, TransformerBlock))


def test_the_compiled_trainer_follows_the_same_trajectory_with_remat():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 8, 4)).astype(np.float32)
    y = rng.normal(size=(16, 3)).astype(np.float32)

    def run(remat):
        m = transformer_encoder(
            4, 8, 2, 16, 2, 3, jax.random.PRNGKey(0), 8, remat=remat
        )
        t = Trainer(m, MSE(), SGD(0.01), compile=True)
        t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
        return t.history_["train_loss"]

    assert run(True) == pytest.approx(run(False), rel=1e-4)


def _train_data(n=64):
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(n, 2)).astype(np.float32),
        rng.normal(size=(n, 1)).astype(np.float32),
    )


def _fit(compile, X, y, epochs, batch_size, opt=None, **kw):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_kw = {
        k: kw.pop(k)
        for k in ("accumulate", "clip_norm", "compute_dtype", "loss_scale")
        if k in kw
    }
    t = Trainer(m, MSE(), opt or SGD(0.05), compile=compile, **t_kw)
    t.fit(X, y, epochs=epochs, batch_size=batch_size, key=jax.random.PRNGKey(0), **kw)
    return t, [np.array(p.value) for p in m.params()]


def _same(a, b):
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-7)


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


def test_the_default_trainer_gives_the_values_stored_from_master():
    """compile=True, accumulate=1 and no stream: the three epoch losses of the
    MLP [2, 4, 1] on 16 samples, SGD(0.01), batch 8, shuffle key 0, taken by
    running master."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 2)).astype(np.float32)
    y = rng.normal(size=(16, 1)).astype(np.float32)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    assert t.accumulate == 1
    t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
    if jax.config.jax_enable_x64:
        expected, tol = STORED_MASTER_LOSSES_X64_TRUE, 1e-9
    else:
        expected, tol = STORED_MASTER_LOSSES_X64_FALSE, 1e-5
    assert t.history_["train_loss"] == pytest.approx(expected, rel=tol)
    assert list(t.history_) == ["train_loss", "val_loss", "lr"]


@pytest.mark.parametrize("compile", [False, True])
def test_accumulate_4_of_8_equals_one_batch_of_32_after_10_updates(compile):
    X, y = _train_data(64)
    t1, w1 = _fit(compile, X, y, 5, 32, shuffle=False)
    t4, w4 = _fit(compile, X, y, 5, 8, shuffle=False, accumulate=4)
    print(
        f"\n  weights differ by {max(float(np.abs(a - b).max()) for a, b in zip(w1, w4))}"
    )
    _same(w1, w4)
    assert t4.history_["train_loss"] == pytest.approx(
        t1.history_["train_loss"], rel=1e-5
    )


@pytest.mark.parametrize("compile", [False, True])
def test_the_clip_acts_on_the_mean_gradient_of_the_accumulation(compile):
    X, y = _train_data(64)
    _, w1 = _fit(compile, X, y, 5, 32, shuffle=False, clip_norm=0.05)
    _, w4 = _fit(compile, X, y, 5, 8, shuffle=False, accumulate=4, clip_norm=0.05)
    _, w0 = _fit(compile, X, y, 5, 32, shuffle=False)
    _same(w1, w4)
    assert max(float(np.abs(a - b).max()) for a, b in zip(w1, w0)) > 1e-3


def test_the_last_group_of_an_epoch_averages_the_micro_batches_that_are_left():
    X, y = _train_data(24)
    t, _ = _fit(True, X, y, 1, 8, opt=Adam(0.01), shuffle=False, accumulate=2)
    assert int(t._compiled_opt_state["t"]) == 2
    t, _ = _fit(True, X, y, 2, 8, opt=Adam(0.01), shuffle=False, accumulate=2)
    assert int(t._compiled_opt_state["t"]) == 4


def test_accumulate_below_one_raises():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="accumulate must be at least 1"):
        Trainer(m, MSE(), SGD(0.01), compile=True, accumulate=0)


@pytest.mark.parametrize(
    "kw",
    [
        {"compute_dtype": "float16", "loss_scale": 1024.0},
        {"compute_dtype": "bfloat16"},
        {"loss_scale": 256.0},
    ],
)
def test_mixed_accumulation_follows_the_mixed_step_of_the_large_batch(kw):
    """4 micro-batches of 8 in the mixed way against one mixed batch of 32.

    The gradients are stored in the compute dtype per micro-batch, so the
    two differ by the rounding of that dtype, not by more."""
    X, y = _train_data(64)
    t1, w1 = _fit(True, X, y, 5, 32, shuffle=False, **kw)
    t4, w4 = _fit(True, X, y, 5, 8, shuffle=False, accumulate=4, **kw)
    tol = {"float16": 2e-2, "bfloat16": 5e-2}.get(kw.get("compute_dtype"), 1e-5)
    for a, b in zip(w1, w4):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol * 0.1)
    assert t4.history_["train_loss"] == pytest.approx(
        t1.history_["train_loss"], rel=tol
    )


def test_mixed_accumulation_with_the_dynamic_scale_trains():
    """The skipped groups of the first epochs differ from the skipped steps of
    the large batch, so the two runs are not compared; the run must train."""
    X, y = _train_data(64)
    t, _ = _fit(
        True,
        X,
        y,
        12,
        8,
        shuffle=False,
        accumulate=4,
        compute_dtype="float16",
        loss_scale="dynamic",
    )
    losses = t.history_["train_loss"]
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < 0.8 * losses[0]
    assert t.history_["loss_scale"][-1] < 65536.0


def test_one_overflow_in_a_group_skips_the_whole_group_and_halves_the_scale_once():
    X, y = _train_data(32)
    X = X.copy()
    X[8:16] *= 1e30
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(
        m,
        MSE(),
        SGD(0.05),
        compile=True,
        accumulate=4,
        compute_dtype="float16",
        loss_scale="dynamic",
    )
    t._scale = 1024.0
    before = [np.array(p.value, dtype=np.float32) for p in m.params()]
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    after = [np.array(p.value, dtype=np.float32) for p in m.params()]
    for a, b in zip(before, after):
        assert np.array_equal(a, b)
    assert t.history_["loss_scale"] == [512.0]


def test_a_group_without_overflow_after_a_skipped_one_updates_the_weights():
    X, y = _train_data(64)
    X = X.copy()
    X[0:8] *= 1e30
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(
        m,
        MSE(),
        SGD(0.05),
        compile=True,
        accumulate=4,
        compute_dtype="float16",
        loss_scale="dynamic",
    )
    t._scale = 1024.0
    before = [np.array(p.value, dtype=np.float32) for p in m.params()]
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    after = [np.array(p.value, dtype=np.float32) for p in m.params()]
    assert max(float(np.abs(a - b).max()) for a, b in zip(before, after)) > 0.0
    assert t.history_["loss_scale"] == [512.0]


def _make_acc(**kw):
    from dense_armor.utility.neural.compiled import make_accumulating_step
    from dense_armor.utility.neural.schedules import ConstantLR

    m = kw.pop("model", None) or mlp([2, 4, 1], jax.random.PRNGKey(0))
    opt = kw.pop("opt", None) or SGD(0.01)
    return make_accumulating_step(
        m, MSE(), opt, ConstantLR(0.01), 0.0, "l2", None, **kw
    )


def test_the_accumulating_step_rejects_an_unsupported_optimizer():
    class Weird:
        pass

    with pytest.raises(ValueError, match="does not support optimizer Weird"):
        _make_acc(opt=Weird())


def test_the_accumulating_step_rejects_a_model_with_state():
    from dense_armor.utility.neural.layers import Dropout

    with pytest.raises(ValueError, match="Dropout"):
        _make_acc(model=Dropout(0.1, jax.random.PRNGKey(0)))


def test_the_accumulating_step_rejects_an_unknown_compute_dtype():
    with pytest.raises(ValueError, match="compute_dtype"):
        _make_acc(compute_dtype="float64")


def _make_parallel(**kw):
    from dense_armor.utility.neural.compiled import make_step_parallel
    from dense_armor.utility.neural.schedules import ConstantLR

    m = kw.pop("model", None) or mlp([2, 4, 1], jax.random.PRNGKey(0))
    opt = kw.pop("opt", None) or SGD(0.01)
    n = kw.pop("n_devices", 2)
    return make_step_parallel(m, MSE(), opt, ConstantLR(0.01), 0.0, "l2", None, n, **kw)


def test_the_parallel_step_rejects_an_unsupported_optimizer():
    class Weird:
        pass

    with pytest.raises(ValueError, match="does not support optimizer Weird"):
        _make_parallel(opt=Weird())


def test_the_parallel_step_rejects_a_model_with_state():
    from dense_armor.utility.neural.layers import Dropout

    with pytest.raises(ValueError, match="Dropout"):
        _make_parallel(model=Dropout(0.1, jax.random.PRNGKey(0)))


def test_the_parallel_step_rejects_fewer_than_two_devices():
    with pytest.raises(ValueError, match="n_devices must be >= 2"):
        _make_parallel(n_devices=1)


def test_an_epoch_without_a_batch_raises():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    with pytest.raises(ValueError, match="no batch"):
        t.fit(
            None,
            None,
            epochs=1,
            batch_size=None,
            key=jax.random.PRNGKey(0),
            stream=DataStream(lambda: iter(()), 4),
        )


@pytest.mark.parametrize("compile", [False, True])
def test_fit_on_a_stream_equals_fit_on_the_arrays(compile):
    X, y = _train_data(64)
    _, w_arrays = _fit(compile, X, y, 5, 8, shuffle=False)
    _, w_stream = _fit(compile, None, None, 5, None, stream=DataStream((X, y), 8))
    _same(w_arrays, w_stream)


@pytest.mark.parametrize("compile", [False, True])
def test_fit_on_a_generator_stream_equals_fit_on_the_arrays(compile):
    X, y = _train_data(64)
    _, w_arrays = _fit(compile, X, y, 3, 8, shuffle=False)
    stream = DataStream(lambda: zip(X, y), 8)
    _, w_stream = _fit(compile, None, None, 3, None, stream=stream)
    _same(w_arrays, w_stream)


def test_fit_on_a_stream_with_accumulation_equals_one_large_batch():
    X, y = _train_data(64)
    _, w1 = _fit(True, X, y, 5, 32, shuffle=False)
    _, w4 = _fit(True, None, None, 5, None, stream=DataStream((X, y), 8), accumulate=4)
    _same(w1, w4)


def test_fit_on_a_shuffled_memory_mapped_stream_trains(tmp_path):
    X, y = _train_data(64)
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", y)
    stream = DataStream(
        (tmp_path / "x.npy", tmp_path / "y.npy"), 8, 16, jax.random.PRNGKey(0)
    )
    t, _ = _fit(True, None, None, 12, None, stream=stream)
    losses = t.history_["train_loss"]
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < 0.9 * losses[0]


def test_fit_on_a_stream_runs_with_a_validation_set_and_early_stopping():
    X, y = _train_data(64)
    t, _ = _fit(
        True,
        None,
        None,
        6,
        None,
        stream=DataStream((X, y), 8),
        X_val=X[:8],
        y_val=y[:8],
        patience=2,
    )
    assert t.best_epoch_ is not None
    assert not np.isnan(t.history_["val_loss"][0])


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"X": np.zeros((8, 2)), "y": np.zeros((8, 1))}, "not both"),
        ({"batch_size": 4}, "differs from the stream"),
        ({"stream": None}, "required without a stream"),
    ],
)
def test_fit_argument_errors(kw, match):
    X, y = _train_data(16)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    args = {
        "X": None,
        "y": None,
        "epochs": 1,
        "batch_size": None,
        "stream": DataStream((X, y), 8),
    }
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        t.fit(key=jax.random.PRNGKey(0), **args)


def test_fit_without_a_stream_needs_a_batch_size():
    X, y = _train_data(16)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    with pytest.raises(ValueError, match="batch_size is required"):
        t.fit(X, y, epochs=1, batch_size=None, key=jax.random.PRNGKey(0))
