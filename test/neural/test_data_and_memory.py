"""Tests of the data stream and of the gradient checkpointing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.attention import TransformerBlock
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import transformer_encoder
from dense_armor.utility.neural.stream import DataStream
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD


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
