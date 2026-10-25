"""Tests for BottleneckFusion."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.bottleneck import BottleneckFusion
from dense_armor.utility.neural.gradcheck import gradcheck

KEY = jax.random.PRNGKey(0)


def _tiny_model(n_bottleneck: int = 2) -> BottleneckFusion:
    return BottleneckFusion(4, 2, n_bottleneck, KEY, d_ff=8)


def test_forward_matches_manual_composition():
    m = _tiny_model()
    key = jax.random.PRNGKey(1)
    a = jax.random.normal(key, (3, 4))
    b = jax.random.normal(key, (5, 4))
    a_out, b_out, fsn_new = m.forward(a, b)

    ab = jnp.concatenate([a, m.fsn.value], axis=0)
    bb = jnp.concatenate([b, m.fsn.value], axis=0)
    a_full = m.block_a.forward(ab)
    b_full = m.block_b.forward(bb)
    a_man = a_full[:3]
    b_man = b_full[:5]
    fsn_man = 0.5 * (a_full[3:] + b_full[5:])
    assert np.allclose(np.asarray(a_out), np.asarray(a_man), atol=1e-6)
    assert np.allclose(np.asarray(b_out), np.asarray(b_man), atol=1e-6)
    assert np.allclose(np.asarray(fsn_new), np.asarray(fsn_man), atol=1e-6)


def test_flow_restricted_to_bottleneck_one_layer():
    m = _tiny_model()
    a = jax.random.normal(KEY, (3, 4))
    b1 = jax.random.normal(jax.random.PRNGKey(2), (5, 4))
    b2 = jax.random.normal(jax.random.PRNGKey(3), (5, 4))
    a_out1, _, fsn1 = m.forward(a, b1)
    a_out2, _, fsn2 = m.forward(a, b2)
    d = float(np.max(np.abs(np.asarray(a_out1) - np.asarray(a_out2))))
    print(f"one layer: max |a'(b1) - a'(b2)| = {d}")
    assert d == 0.0
    assert not np.allclose(np.asarray(fsn1), np.asarray(fsn2))


def test_flow_through_bottleneck_two_layers():
    m1 = _tiny_model()
    m2 = _tiny_model()
    a = jax.random.normal(KEY, (3, 4))
    b1 = jax.random.normal(jax.random.PRNGKey(2), (5, 4))
    b2 = jax.random.normal(jax.random.PRNGKey(3), (5, 4))
    a1_1, _, fsn1_1 = m1.forward(a, b1)
    a2_1, _, _ = m2.forward(a1_1, b1, fsn=fsn1_1)
    a1_2, _, fsn1_2 = m1.forward(a, b2)
    a2_2, _, _ = m2.forward(a1_2, b2, fsn=fsn1_2)
    d = float(np.max(np.abs(np.asarray(a2_1) - np.asarray(a2_2))))
    print(f"two layers: max |a''(b1) - a''(b2)| = {d}")
    assert d > 0.0


def test_shapes_n_bottleneck():
    for nb in (1, 4, 16):
        m = BottleneckFusion(4, 2, nb, KEY, d_ff=8)
        a = jnp.ones((3, 4))
        b = jnp.ones((5, 4))
        a_out, b_out, fsn_out = m.forward(a, b)
        assert a_out.shape == (3, 4)
        assert b_out.shape == (5, 4)
        assert fsn_out.shape == (nb, 4)


def test_single_and_batch():
    m = _tiny_model()
    a = jax.random.normal(KEY, (3, 4))
    b = jax.random.normal(KEY, (5, 4))
    y1 = m.forward(a, b)
    m2 = _tiny_model()
    y2 = m2.forward(a[None], b[None])
    for i in range(3):
        assert np.allclose(np.asarray(y1[i]), np.asarray(y2[i][0]), atol=1e-6)


def test_errors():
    with pytest.raises(ValueError):
        BottleneckFusion(4, 2, 0, KEY)
    m = _tiny_model()
    with pytest.raises(RuntimeError):
        m.backward(jnp.ones((3, 4)), jnp.ones((5, 4)), None)


def test_params():
    m = _tiny_model()
    ps = m.params()
    assert ps[0] is m.fsn
    assert len(ps) == 1 + len(m.block_a.params()) + len(m.block_b.params())


def test_repr():
    m = _tiny_model()
    s = repr(m)
    assert "BottleneckFusion" in s
    assert "n_bottleneck=2" in s


def test_gradcheck_a():
    m = _tiny_model()
    b = jax.random.normal(KEY, (5, 4))

    def f(a_):
        return m.forward(a_, b)[0]

    def bw(a_, y, g):
        m.forward(a_, b)
        da, _, _ = m.backward(g, jnp.zeros((5, 4)), None)
        return da

    a = jax.random.normal(KEY, (3, 4))
    assert gradcheck(f, bw, a) < 1e-5


def test_gradcheck_b():
    m = _tiny_model()
    a = jax.random.normal(KEY, (3, 4))

    def f(b_):
        return m.forward(a, b_)[1]

    def bw(b_, y, g):
        m.forward(a, b_)
        _, db, _ = m.backward(jnp.zeros((3, 4)), g, None)
        return db

    b = jax.random.normal(KEY, (5, 4))
    assert gradcheck(f, bw, b) < 1e-5


def test_learning():
    """Two layers in sequence, predict a function of b alone."""
    from dense_armor.utility.neural.losses import MSE
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import Adam

    class TwoLayers:
        def __init__(self, key):
            k1, k2 = jax.random.split(key)
            self.m1 = BottleneckFusion(4, 2, 2, k1, d_ff=8)
            self.m2 = BottleneckFusion(4, 2, 2, k2, d_ff=8)
            self.Ta = 3
            self.Tb = 5

        def forward(self, x, training=True):
            a = x[:, : self.Ta, :]
            b = x[:, self.Ta :, :]
            a1, _, fsn1 = self.m1.forward(a, b, training=training)
            a2, _, _ = self.m2.forward(a1, b, fsn=fsn1, training=training)
            return jnp.mean(a2, axis=1)

        def backward(self, g):
            B = g.shape[0]
            D = g.shape[1]
            ga2 = jnp.broadcast_to(g[:, None, :] / self.Ta, (B, self.Ta, D))
            gb2 = jnp.zeros((B, self.Tb, D))
            gfsn2 = jnp.zeros((B, 2, D))
            da1, _, dfsn1 = self.m2.backward(ga2, gb2, gfsn2)
            self.m1.backward(da1, jnp.zeros((B, self.Tb, D)), dfsn1)

        def params(self):
            return self.m1.params() + self.m2.params()

        def zero_grad(self):
            for pp in self.params():
                pp.zero_grad()

    key = jax.random.PRNGKey(7)
    B, Ta, Tb, D = 32, 3, 5, 4
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (B, Ta, D))
    b = jax.random.normal(k2, (B, Tb, D))
    y = jnp.sum(b, axis=1)
    X = jnp.concatenate([a, b], axis=1)

    m = TwoLayers(jax.random.PRNGKey(11))
    trainer = Trainer(m, MSE(), Adam(lr=0.01))
    before = trainer.evaluate(X, y)
    trainer.fit(X, y, epochs=80, batch_size=16, key=jax.random.PRNGKey(0))
    after = trainer.evaluate(X, y)
    print(f"learning: before={before:.4f} after={after:.4f}")
    assert after < before * 0.5


def test_d_ff_none_defaults_to_4_times_d_model():
    m = BottleneckFusion(4, 2, 2, KEY)
    assert m.d_ff == 16
    a = jnp.ones((3, 4))
    b = jnp.ones((5, 4))
    a_out, b_out, fsn_out = m.forward(a, b)
    assert np.asarray(a_out).shape == (3, 4)
    assert np.asarray(fsn_out).shape == (2, 4)


def test_2d_fsn_with_3d_streams():
    m = _tiny_model()
    a = jnp.ones((2, 3, 4))
    b = jnp.ones((2, 5, 4))
    fsn = jnp.zeros((2, 4))
    a_out, b_out, fsn_out = m.forward(a, b, fsn=fsn)
    assert a_out.shape == (2, 3, 4)
    assert b_out.shape == (2, 5, 4)
    assert fsn_out.shape == (2, 4)
    da, db, dfsn = m.backward(
        jnp.ones((2, 3, 4)), jnp.ones((2, 5, 4)), jnp.ones((2, 4))
    )
    assert da.shape == (2, 3, 4)
    assert db.shape == (2, 5, 4)
    assert dfsn.shape == (2, 4)


def test_gradcheck_fsn():
    m = BottleneckFusion(2, 1, 2, jax.random.PRNGKey(0), d_ff=2)
    a = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    b = jax.random.normal(jax.random.PRNGKey(0), (5, 2))

    def f(fsn_):
        m.fsn.value = fsn_
        return m.forward(a, b, fsn=None)[2]

    def bw(fsn_, y, gg):
        m.fsn.value = fsn_
        m.forward(a, b, fsn=None)
        for pp in m.params():
            pp.zero_grad()
        m.backward(jnp.zeros((3, 2)), jnp.zeros((5, 2)), gg)
        return m.fsn.grad

    fsn0 = 0.02 * jax.random.normal(jax.random.PRNGKey(0), (2, 2))
    assert gradcheck(f, bw, fsn0) < 1e-4
