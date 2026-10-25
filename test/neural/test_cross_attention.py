"""Tests for MultiHeadCrossAttention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.attention import MultiHeadSelfAttention
from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.gradcheck import gradcheck

KEY = jax.random.PRNGKey(0)


def _identity_model(d_context: int = 2) -> MultiHeadCrossAttention:
    m = MultiHeadCrossAttention(2, 1, KEY, d_context=d_context)
    for p in m.params():
        p.value = jnp.eye(2)
    return m


def test_hand_case():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = m.forward(xq, xc)
    expected_w = np.array([[0.40111, 0.19778, 0.40111], [0.19778, 0.40111, 0.40111]])
    expected_y = np.array([[0.80222, 0.59889], [0.59889, 0.80222]])
    assert np.allclose(np.asarray(m.weights_[0, 0]), expected_w, atol=1e-5)
    assert np.allclose(np.asarray(out), expected_y, atol=1e-5)

    dxq, dxc = m.backward(jnp.ones((2, 2)))
    expected_dxq = np.array([[0.05609, 0.11377], [0.11377, 0.05609]])
    expected_dxc = np.array(
        [[0.48512, 0.54279], [0.54279, 0.48512], [0.97209, 0.97209]]
    )
    assert np.allclose(np.asarray(dxq), expected_dxq, atol=1e-5)
    assert np.allclose(np.asarray(dxc), expected_dxc, atol=1e-5)


def test_mask_forbids_column():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mask = jnp.array([[True, False, True], [True, True, True]])
    m.forward(xq, xc, mask=mask)
    w = np.asarray(m.weights_[0, 0])
    assert w[0, 1] == 0.0
    assert np.allclose(w.sum(axis=1), np.ones(2), atol=1e-6)


def test_matches_self_attention_when_context_equals_queries():
    key = jax.random.PRNGKey(1)
    self_attn = MultiHeadSelfAttention(4, 2, key)
    cross = MultiHeadCrossAttention(4, 2, key, d_context=4)
    cross.W_Q.value = self_attn.W_Q.value
    cross.W_K.value = self_attn.W_K.value
    cross.W_V.value = self_attn.W_V.value
    cross.W_O.value = self_attn.W_O.value
    x = jax.random.normal(key, (5, 4))
    y_self = self_attn.forward(x)
    y_cross = cross.forward(x, x)
    assert np.allclose(np.asarray(y_self), np.asarray(y_cross), atol=1e-6)


def test_rows_of_queries_are_independent():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y1 = np.asarray(m.forward(xq, xc))
    xq2 = xq.at[0, 0].set(2.0)
    y2 = np.asarray(m.forward(xq2, xc))
    assert not np.allclose(y1[0], y2[0])
    assert np.allclose(y1[1], y2[1], atol=1e-7)


def test_gradcheck_xq():
    m = _identity_model()
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def f(xq_):
        return m.forward(xq_, xc)

    def b(xq_, y, g):
        m.forward(xq_, xc)
        dxq, _ = m.backward(g)
        return dxq

    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    assert gradcheck(f, b, xq) < 1e-6


def test_gradcheck_xc():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def f(xc_):
        return m.forward(xq, xc_)

    def b(xc_, y, g):
        m.forward(xq, xc_)
        _, dxc = m.backward(g)
        return dxc

    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert gradcheck(f, b, xc) < 1e-6


def test_gradcheck_params():
    key = jax.random.PRNGKey(3)
    xq = jax.random.normal(key, (2, 4))
    xc = jax.random.normal(jax.random.PRNGKey(4), (3, 3))

    for name in ["W_Q", "W_K", "W_V", "W_O"]:

        def f(v, name=name):
            m = MultiHeadCrossAttention(4, 2, key, d_context=3)
            getattr(m, name).value = v
            return m.forward(xq, xc)

        def b(v, y, g, name=name):
            m = MultiHeadCrossAttention(4, 2, key, d_context=3)
            getattr(m, name).value = v
            m.forward(xq, xc)
            for p in m.params():
                p.zero_grad()
            m.backward(g)
            return getattr(m, name).grad

        m0 = MultiHeadCrossAttention(4, 2, key, d_context=3)
        v0 = getattr(m0, name).value
        assert gradcheck(f, b, v0) < 1e-6


def test_single_and_batch():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y_single = np.asarray(m.forward(xq, xc))
    m2 = _identity_model()
    y_batch = np.asarray(m2.forward(xq[None], xc[None])[0])
    assert np.allclose(y_single, y_batch, atol=1e-7)

    dxq1, dxc1 = m.backward(jnp.ones((2, 2)))
    dxq2, dxc2 = m2.backward(jnp.ones((1, 2, 2)))
    assert np.allclose(np.asarray(dxq1), np.asarray(dxq2[0]), atol=1e-6)
    assert np.allclose(np.asarray(dxc1), np.asarray(dxc2[0]), atol=1e-6)


def test_stream_step_matches_forward():
    m = _identity_model()
    xq = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    xc = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cache = m.init_cache()
    for t in range(3):
        cache = m.append_context(cache, xc[t])
        y_step = np.asarray(m.step(xq[0], cache))
        y_fwd = np.asarray(m.forward(xq, xc[: t + 1])[0])
        assert np.allclose(y_step, y_fwd, atol=1e-5)


def test_repr():
    m = _identity_model()
    s = repr(m)
    assert "MultiHeadCrossAttention" in s
    assert "d_context=2" in s


def test_errors():
    with pytest.raises(ValueError):
        MultiHeadCrossAttention(3, 2, KEY)
    with pytest.raises(ValueError):
        MultiHeadCrossAttention(2, 1, KEY, d_context=0)

    m = _identity_model()
    with pytest.raises(RuntimeError):
        m.backward(jnp.ones((2, 2)))

    m = _identity_model()
    xq = jnp.ones((2, 2))
    xc = jnp.ones((3, 2))
    with pytest.raises(ValueError):
        m.forward(xq, xc, mask=jnp.ones((3, 2), dtype=bool))

    m = _identity_model()
    with pytest.raises(ValueError):
        m.forward(jnp.ones((1, 2, 2)), jnp.ones((2, 3, 2)))


def test_learning():
    """Train a model that copies the xc row selected by xq[..., 0].

    The xc rows are one-hot so that each row is identifiable by content:
    the cross-attention is permutation-invariant on the context, so a row
    cannot be selected by position.
    """
    import jax.numpy as jnp

    from dense_armor.utility.neural.losses import MSE
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import Adam

    class CrossCopy:
        def __init__(self, key):
            self.att = MultiHeadCrossAttention(4, 2, key, d_context=4)

        def forward(self, x, training=True):
            return self.att.forward(x[..., :4], x[..., 4:], training=training)

        def backward(self, g):
            dxq, dxc = self.att.backward(g)
            return jnp.concatenate([dxq, dxc], axis=-1)

        def params(self):
            return self.att.params()

        def zero_grad(self):
            for p in self.params():
                p.zero_grad()

    key = jax.random.PRNGKey(7)
    B, T, _ = 32, 4, 4
    kxq, _ = jax.random.split(key)
    idx = jax.random.randint(kxq, (B, T), 0, T)
    xq = jax.nn.one_hot(idx, T)
    xc = 3.0 * jnp.tile(jax.nn.one_hot(jnp.arange(T), T), (B, 1, 1))
    y = xc[jnp.arange(B)[:, None], idx]

    X = jnp.concatenate([xq, xc], axis=-1)

    m = CrossCopy(jax.random.PRNGKey(11))
    trainer = Trainer(m, MSE(), Adam(lr=0.01))
    before = trainer.evaluate(X, y)
    trainer.fit(X, y, epochs=200, batch_size=32, key=jax.random.PRNGKey(0))
    after = trainer.evaluate(X, y)
    print(f"learning: before={before:.4f} after={after:.4f}")
    assert after < before / 10.0
