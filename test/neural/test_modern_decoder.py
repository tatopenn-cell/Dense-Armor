"""Tests for the modern decoder, its static cache step and the cached generate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.generate import generate
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.losses_lm import NextTokenLoss
from dense_armor.utility.neural.modern_decoder import (
    ModernDecoderBlock,
    modern_decoder,
)
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam

VOCAB = 11
IDS = jnp.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=jnp.int32)


def _model(remat=False, key=0):
    return modern_decoder(VOCAB, 8, 2, 2, 12, key=jax.random.PRNGKey(key), remat=remat)


class _ForwardOnly:
    """The same model seen through ``forward`` only: the path without a cache."""

    def __init__(self, model):
        self._m = model
        self.max_len = model.max_len

    def forward(self, ids):
        return self._m.forward(ids, training=False)


def test_builder_param_count_by_hand():
    m = _model()
    n = sum(int(p.value.size) for p in m.params())
    emb = VOCAB * 8
    block = 8 + 4 * 8 * 8 + 8 + 3 * 8 * 16
    head = 8 * VOCAB + VOCAB
    assert n == emb + 2 * block + 8 + head == 1507


def test_builder_forward_shape_single_and_batch():
    m = _model()
    assert m.forward(IDS).shape == (8, VOCAB)
    assert m.forward(jnp.stack([IDS, IDS])).shape == (2, 8, VOCAB)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_layers=0),
        dict(d_model=9, n_heads=2),
        dict(d_model=6, n_heads=2),
    ],
)
def test_builder_rejects_bad_sizes(kwargs):
    args = dict(vocab=VOCAB, d_model=8, n_heads=2, n_layers=1, max_len=4)
    args.update(kwargs)
    with pytest.raises(ValueError):
        modern_decoder(**args, key=jax.random.PRNGKey(0))


def test_block_gradcheck_on_the_input():
    b = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))

    def bwd(xx, y, g):
        b.forward(xx)
        return b.backward(g)

    assert gradcheck(b.forward, bwd, x) < 1e-4


def test_block_gradient_of_a_weight_matches_finite_differences():
    b = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (5, 8))

    def loss(w):
        old = b.attn.W_Q.value
        b.attn.W_Q.value = w
        out = float(jnp.sum(g * b.forward(x)))
        b.attn.W_Q.value = old
        return out

    for p in b.params():
        p.grad = jnp.zeros_like(p.value)
    b.forward(x)
    b.backward(g)
    w = b.attn.W_Q.value
    d = np.zeros(w.shape)
    d[1, 2] = 1e-6
    fd = (loss(w + d) - loss(w - d)) / 2e-6
    assert abs(float(b.attn.W_Q.grad[1, 2]) - fd) < 1e-5 * max(1.0, abs(fd))


def test_remat_gives_the_same_output_and_gradients():
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    g = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
    plain = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
    remat = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0), remat=True)
    for blk in (plain, remat):
        for p in blk.params():
            p.grad = jnp.zeros_like(p.value)
    np.testing.assert_allclose(remat.forward(x), plain.forward(x), atol=1e-12)
    np.testing.assert_allclose(remat.backward(g), plain.backward(g), atol=1e-10)
    for a, b in zip(remat.params(), plain.params()):
        np.testing.assert_allclose(a.grad, b.grad, atol=1e-10)


def test_remat_backward_before_forward_raises():
    b = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0), remat=True)
    with pytest.raises(RuntimeError):
        b.backward(jnp.zeros((5, 8)))


def test_output_at_t_ignores_the_tokens_after_t():
    m = _model()
    other = IDS.at[5:].set(0)
    a = m.forward(IDS)
    b = m.forward(other)
    np.testing.assert_allclose(a[:5], b[:5], atol=1e-12)
    assert float(jnp.max(jnp.abs(a[5:] - b[5:]))) > 1e-6


def test_step_static_logits_equal_the_full_forward_at_every_position():
    m = _model()
    full = m.forward(IDS)
    caches = m.init_static_cache(12)
    for t in range(IDS.shape[0]):
        logits, caches = m.step_static(IDS[t], caches, t)
        np.testing.assert_allclose(logits, full[t], atol=1e-9)


def test_step_compiled_equals_step_static():
    m = _model()
    c1 = m.init_static_cache(12)
    c2 = m.init_static_cache(12)
    for t in range(IDS.shape[0]):
        a, c1 = m.step_static(IDS[t], c1, t)
        b, c2 = m.step_compiled(IDS[t], c2, t)
        np.testing.assert_allclose(b, a, atol=1e-9)


def test_rows_after_the_position_are_masked():
    m = _model()
    clean = m.init_static_cache(12)
    junk = [
        (
            jax.random.normal(jax.random.PRNGKey(7), k.shape),
            jax.random.normal(jax.random.PRNGKey(8), v.shape),
        )
        for k, v in clean
    ]
    a, _ = m.step_static(IDS[0], clean, 0)
    b, _ = m.step_static(IDS[0], junk, 0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_the_cache_has_a_fixed_shape_and_is_written_at_the_position():
    m = _model()
    caches = m.init_static_cache(12)
    assert [tuple(k.shape) for k, _ in caches] == [(12, 8), (12, 8)]
    _, new = m.step_static(IDS[0], caches, 4)
    k = new[0][0]
    assert k.shape == (12, 8)
    assert float(jnp.max(jnp.abs(k[4]))) > 0.0
    assert float(jnp.max(jnp.abs(jnp.delete(k, 4, axis=0)))) == 0.0


def test_step_compiled_reads_the_current_weights():
    m = _model()
    before, _ = m.step_compiled(IDS[0], m.init_static_cache(12), 0)
    m.head.W.value = m.head.W.value * 2.0
    after, _ = m.step_compiled(IDS[0], m.init_static_cache(12), 0)
    eager, _ = m.step_static(IDS[0], m.init_static_cache(12), 0)
    np.testing.assert_allclose(after, eager, atol=1e-9)
    assert float(jnp.max(jnp.abs(after - before))) > 1e-6


def test_step_compiled_leaves_the_model_usable_for_training():
    m = _model()
    m.step_compiled(IDS[0], m.init_static_cache(12), 0)
    m.forward(IDS)
    for p in m.params():
        p.grad = jnp.zeros_like(p.value)
    m.backward(jnp.ones((8, VOCAB)))
    assert all(bool(jnp.all(jnp.isfinite(p.grad))) for p in m.params())


def test_generate_cached_equals_generate_without_cache_greedy():
    m = _model()
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    fast = generate(m, prompt, max_new=7)
    slow = generate(_ForwardOnly(m), prompt, max_new=7)
    assert fast.shape == (10,)
    np.testing.assert_array_equal(fast, slow)


def test_generate_cached_equals_generate_without_cache_sampled():
    m = _model()
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    key = jax.random.PRNGKey(5)
    fast = generate(m, prompt, max_new=7, temperature=1.0, key=key)
    slow = generate(_ForwardOnly(m), prompt, max_new=7, temperature=1.0, key=key)
    np.testing.assert_array_equal(fast, slow)


def test_generate_cached_stops_at_max_len_and_handles_the_edges():
    m = _model()
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    assert generate(m, prompt, max_new=50).shape == (12,)
    assert generate(m, prompt, max_new=0).shape == (3,)
    full = jnp.arange(12, dtype=jnp.int32) % VOCAB
    np.testing.assert_array_equal(generate(m, full, max_new=5), full)
    assert generate(m, prompt, max_new=2, max_len=4).shape == (4,)


def test_generate_without_a_cache_is_unchanged_for_a_plain_model():
    m = _model()
    out = generate(_ForwardOnly(m), jnp.array([1, 2], dtype=jnp.int32), max_new=3)
    assert out.shape == (5,)


def test_trainer_trains_the_modern_decoder_and_remat_gives_the_same_loss():
    tokens = jnp.asarray(np.tile(np.arange(4), (8, 2)), dtype=jnp.int32)
    losses = {}
    for remat in (False, True):
        m = modern_decoder(4, 16, 2, 1, 8, key=jax.random.PRNGKey(0), remat=remat)
        t = Trainer(m, NextTokenLoss(), Adam(0.05), compile=False)
        t.fit(tokens, tokens, epochs=3, batch_size=4, key=jax.random.PRNGKey(0))
        losses[remat] = t.history_["train_loss"]
    np.testing.assert_allclose(losses[True], losses[False], rtol=1e-6)
    assert losses[False][-1] < losses[False][0]


def test_repr_names_the_sizes():
    assert "n_layers=2" in repr(_model())
    assert "remat=True" in repr(
        ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0), True)
    )
