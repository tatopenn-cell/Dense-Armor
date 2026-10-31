"""Tests for the decoder block, the LM builder, the next-token loss, generate."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from dense_armor.utility.neural.decoder import TransformerDecoderBlock
from dense_armor.utility.neural.generate import generate
from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.losses_lm import NextTokenLoss, perplexity
from dense_armor.utility.neural.models import _TiedHead, transformer_decoder
from dense_armor.utility.neural.layers import Embedding
from dense_armor.utility.optim import Adam


def test_decoder_block_shape_without_cross():
    b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    assert b.forward(x).shape == (4, 8)
    assert len(b.params()) == 12


def test_decoder_block_shape_with_cross():
    b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0), cross=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    mem = jax.random.normal(jax.random.PRNGKey(2), (3, 8))
    assert b.forward(x, memory=mem).shape == (4, 8)
    assert len(b.params()) == 18


def test_decoder_block_cross_requires_memory():
    b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0), cross=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    with pytest.raises(ValueError, match="memory"):
        b.forward(x)


def test_backward_before_forward_raises():
    b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError):
        b.backward(jnp.ones((4, 8)))


def test_causality_exact():
    """Changing tokens after t leaves the logits up to t bit-for-bit equal."""
    m = transformer_decoder(8, 16, 2, 1, 6, key=jax.random.PRNGKey(0))
    a = jnp.array([0, 1, 2, 3, 4, 5])
    b = jnp.array([0, 1, 2, 7, 6, 5])
    la = m.forward(a)
    lb = m.forward(b)
    for t in range(3):
        assert np.array_equal(np.asarray(la[t]), np.asarray(lb[t])), (
            f"position {t} changed"
        )


def test_gradcheck_decoder_block_no_cross():
    b = TransformerDecoderBlock(4, 2, 8, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))

    def fwd(xx):
        return b.forward(xx)

    def bwd(xx, y, g):
        b.forward(xx)
        return b.backward(g)

    assert gradcheck(fwd, bwd, x) < 1e-4


def test_gradcheck_decoder_block_cross():
    b = TransformerDecoderBlock(4, 2, 8, jax.random.PRNGKey(0), cross=True)
    mem = jax.random.normal(jax.random.PRNGKey(2), (3, 4))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))

    def fwd(xx):
        return b.forward(xx, memory=mem)

    def bwd(xx, y, g):
        b.forward(xx, memory=mem)
        return b.backward(g)

    assert gradcheck(fwd, bwd, x) < 1e-4


def test_gradcheck_next_token_loss():
    logits = jax.random.normal(jax.random.PRNGKey(3), (1, 4, 5))
    y = jnp.array([[1, 2, 3, 4]])

    def fwd(zz):
        L = NextTokenLoss()
        return L.forward(zz, y)

    def bwd(zz, out, g):
        L = NextTokenLoss()
        L.forward(zz, y)
        return g * L.backward()

    assert gradcheck(fwd, bwd, logits) < 1e-4


def test_gradcheck_tied_head():
    emb = Embedding(4, 6, jax.random.PRNGKey(0))
    head = _TiedHead(emb)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6))

    def fwd(xx):
        return head.forward(xx)

    def bwd(xx, y, g):
        head.forward(xx)
        return head.backward(g)

    assert gradcheck(fwd, bwd, x) < 1e-4


def test_tied_head_adds_to_E_grad():
    emb = Embedding(4, 6, jax.random.PRNGKey(0))
    head = _TiedHead(emb)
    E_before = np.asarray(emb.E.grad).copy()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6))
    head.forward(x)
    head.backward(jnp.ones((2, 4)))
    E_after = np.asarray(emb.E.grad)
    assert not np.allclose(E_before, E_after), "E.grad not touched by the tied head"


def test_next_token_loss_hand_case():
    L = NextTokenLoss()
    logits = jnp.array([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    y = jnp.array([[1, 0, 1]])
    assert abs(float(L.forward(logits, y)) - 0.1269) < 1e-3
    assert abs(perplexity(0.1269) - 1.1353) < 1e-3


def test_next_token_loss_ignore_index():
    L = NextTokenLoss()
    logits = jnp.array([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    y = jnp.array([[1, -100, 1]])
    assert abs(float(L.forward(logits, y)) - 0.1269) < 1e-3


def test_next_token_loss_all_ignore_raises():
    L = NextTokenLoss()
    logits = jnp.zeros((1, 3, 2))
    y = jnp.array([[-100, -100, -100]])
    with pytest.raises(ValueError, match="valid target"):
        L.forward(logits, y)


def test_next_token_loss_T1_raises():
    L = NextTokenLoss()
    with pytest.raises(ValueError, match="valid target"):
        L.forward(jnp.zeros((1, 1, 2)), jnp.array([[0]]))


def test_next_token_loss_grad_sum_zero():
    L = NextTokenLoss()
    logits = jax.random.normal(jax.random.PRNGKey(3), (1, 4, 5))
    y = jnp.array([[1, 2, 3, 4]])
    L.forward(logits, y)
    g = L.backward()
    s = float(jnp.sum(g[..., :-1, :], axis=-1).max())
    assert abs(s) < 1e-4


def test_generate_empty_prompt_raises():
    m = transformer_decoder(4, 8, 2, 1, 6, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="empty"):
        generate(m, jnp.array([]), max_new=1, max_len=6)


def test_generate_bounds_cap():
    m = transformer_decoder(4, 8, 2, 1, 4, key=jax.random.PRNGKey(0))
    out = generate(m, jnp.array([0, 1, 2]), max_new=10, max_len=4)
    assert out.shape == (4,)


def test_generate_T_equal_max_len():
    m = transformer_decoder(4, 8, 2, 1, 4, key=jax.random.PRNGKey(0))
    out = generate(m, jnp.array([0, 1, 2, 3]), max_new=5, max_len=4)
    assert out.shape == (4,)


def test_batch_and_single_same_logits():
    m = transformer_decoder(8, 16, 2, 1, 6, key=jax.random.PRNGKey(0))
    a = jnp.array([0, 1, 2, 3, 4, 5])
    la = m.forward(a)
    lb = m.forward(a[None])[0]
    assert np.array_equal(np.asarray(la), np.asarray(lb))


def _period4_batch(seed: int, T: int = 8) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(T) % 4
    return jnp.asarray(x.astype(np.int32))


def _train_until(seed: int, max_steps: int = 300, lr: float = 0.05) -> float:
    m = transformer_decoder(4, 16, 2, 1, 8, key=jax.random.PRNGKey(seed))
    tokens = _period4_batch(seed)
    opts = [Adam(lr=lr) for _ in m.params()]
    L = NextTokenLoss()
    last = None
    for _ in range(max_steps):
        m.zero_grad()
        logits = m.forward(tokens)
        last = float(L.forward(logits, tokens))
        if last < 0.05:
            return last
        g_logits = L.backward()
        m.backward(g_logits)
        for p, o in zip(m.params(), opts):
            p.update(o)
    return float(last) if last is not None else float("nan")


def test_learning_repeating_pattern_4():
    loss = _train_until(seed=0, max_steps=300)
    assert loss < 0.05, f"loss {loss:.4f} >= 0.05 after 300 steps"


def test_generate_reproduces_pattern():
    m = transformer_decoder(4, 16, 2, 1, 8, key=jax.random.PRNGKey(0))
    tokens = _period4_batch(0)
    opts = [Adam(lr=0.05) for _ in m.params()]
    L = NextTokenLoss()
    for _ in range(300):
        m.zero_grad()
        logits = m.forward(tokens)
        loss = float(L.forward(logits, tokens))
        if loss < 0.05:
            break
        m.backward(L.backward())
        for p, o in zip(m.params(), opts):
            p.update(o)
    prompt = jnp.array([0, 1])
    out = generate(m, prompt, max_new=5, max_len=8)
    expected = np.array([0, 1, 2, 3, 0, 1, 2])
    assert np.array_equal(np.asarray(out), expected[: out.shape[0]]), (
        f"generated {np.asarray(out).tolist()}, expected {expected[:out.shape[0]].tolist()}"
    )
