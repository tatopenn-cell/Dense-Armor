"""Tests for losses_more.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.losses import (
    BCEWithLogits,
    SoftmaxCrossEntropy,
)
from dense_armor.utility.neural.losses_more import (
    FocalLoss,
    HingeLoss,
    KLDivLoss,
    L1Loss,
    LabelSmoothingCE,
    focal_forward,
    hinge_forward,
    kldiv_forward,
    label_smoothing_ce_forward,
)


def test_l1_hand_case():
    l = L1Loss()
    v = float(l.forward(jnp.array([1.0, 3.0]), jnp.array([2.0, 1.0])))
    g = np.asarray(l.backward())
    print(f"L1: {v:.4f}, grad {g.tolist()}")
    assert abs(v - 1.5) < 1e-6
    assert np.allclose(g, [-0.5, 0.5])


def test_label_smoothing_hand_case():
    ls = LabelSmoothingCE(0.1)
    v = float(ls.forward(jnp.array([[2.0, 0.0, 0.0]]), jnp.array([0])))
    g = np.asarray(ls.backward())
    print(f"LSCE: {v:.7f}, grad {g.tolist()}")
    assert abs(v - 0.3728781) < 1e-6
    assert np.allclose(g, [[-0.1463473, 0.0731736, 0.0731736]], atol=1e-6)


def test_label_smoothing_eps0_equals_ce():
    ls = LabelSmoothingCE(0.0)
    ce = SoftmaxCrossEntropy()
    logits = jnp.array([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
    labels = jnp.array([1, 2])
    v_ls = float(ls.forward(logits, labels))
    v_ce = float(ce.forward(logits, labels))
    print(f"LSCE(0)={v_ls:.7f} vs CE={v_ce:.7f}")
    assert abs(v_ls - v_ce) < 1e-6


def test_kldiv_hand_case():
    kl = KLDivLoss()
    v = float(kl.forward(jnp.array([[2.0, 0.0, 0.0]]), jnp.array([[0.7, 0.2, 0.1]])))
    g = np.asarray(kl.backward())
    print(f"KL: {v:.7f}, grad {g.tolist()}")
    assert abs(v - 0.0377262) < 1e-6
    assert np.allclose(g, [[0.086986, -0.093493, 0.006507]], atol=1e-6)


def test_hinge_hand_case():
    h = HingeLoss()
    v = float(h.forward(jnp.array([0.5, -2.0]), jnp.array([1.0, 1.0])))
    g = np.asarray(h.backward())
    print(f"Hinge: {v:.4f}, grad {g.tolist()}")
    assert abs(v - 1.75) < 1e-6
    assert np.allclose(g, [-0.5, -0.5])


def test_focal_hand_case():
    f = FocalLoss()
    v = float(f.forward(jnp.array([0.0]), jnp.array([1.0])))
    print(f"Focal(0, y=1): {v:.7f}")
    assert abs(v - 0.1732868) < 1e-6

    f2 = FocalLoss(alpha=0.25)
    v2 = float(f2.forward(jnp.array([0.0]), jnp.array([1.0])))
    print(f"Focal(0, y=1, alpha=0.25): {v2:.7f}")
    assert abs(v2 - 0.0433217) < 1e-6

    f3 = FocalLoss()
    v3 = float(f3.forward(jnp.array([2.0]), jnp.array([1.0])))
    print(f"Focal(2, y=1): {v3:.8f}")
    assert abs(v3 - 0.00180356) < 1e-6


def test_focal_gamma0_equals_bce():
    f = FocalLoss(gamma=0.0)
    b = BCEWithLogits()
    logits = jnp.array([[0.5, -1.0, 2.0]])
    y = jnp.array([[1.0, 0.0, 1.0]])
    v_f = float(f.forward(logits, y))
    v_b = float(b.forward(logits, y))
    print(f"Focal(gamma=0)={v_f:.7f} vs BCE={v_b:.7f}")
    assert abs(v_f - v_b) < 1e-6


def test_l1_gradient_jax():
    rng = np.random.default_rng(0)
    pred = jnp.asarray(rng.normal(0.0, 1.0, 5).astype(np.float32))
    tgt = jnp.asarray(rng.normal(0.0, 1.0, 5).astype(np.float32))
    l = L1Loss()
    _ = l.forward(pred, tgt)
    g = np.asarray(l.backward())

    def f(p):
        return jnp.mean(jnp.abs(p - tgt))

    g_ref = np.asarray(jax.grad(f)(pred))
    print(f"L1 grad diff: {np.max(np.abs(g - g_ref)):.2e}")
    assert np.allclose(g, g_ref, atol=1e-6)


def test_label_smoothing_gradient_jax():
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(0.0, 1.0, (2, 3)).astype(np.float32))
    labels = jnp.array([0, 2])
    ls = LabelSmoothingCE(0.1)
    _ = ls.forward(logits, labels)
    g = np.asarray(ls.backward())

    def f(z):
        return label_smoothing_ce_forward(z, labels, 0.1)

    g_ref = np.asarray(jax.grad(f)(logits))
    print(f"LSCE grad diff: {np.max(np.abs(g - g_ref)):.2e}")
    assert np.allclose(g, g_ref, atol=1e-6)


def test_kl_gradient_jax():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(0.0, 1.0, (2, 3)).astype(np.float32))
    tgt = jnp.asarray(rng.dirichlet(jnp.ones(3), 2).astype(np.float32))
    kl = KLDivLoss()
    _ = kl.forward(logits, tgt)
    g = np.asarray(kl.backward())

    def f(z):
        return kldiv_forward(z, tgt)

    g_ref = np.asarray(jax.grad(f)(logits))
    print(f"KL grad diff: {np.max(np.abs(g - g_ref)):.2e}")
    assert np.allclose(g, g_ref, atol=1e-6)


def test_hinge_gradient_jax():
    rng = np.random.default_rng(3)
    pred = jnp.asarray(rng.normal(0.0, 1.0, 4).astype(np.float32))
    y = jnp.array([1.0, -1.0, 1.0, 1.0])
    h = HingeLoss()
    _ = h.forward(pred, y)
    g = np.asarray(h.backward())

    def f(p):
        return hinge_forward(p, y)

    g_ref = np.asarray(jax.grad(f)(pred))
    print(f"Hinge grad diff: {np.max(np.abs(g - g_ref)):.2e}")
    assert np.allclose(g, g_ref, atol=1e-6)


def test_focal_gradient_jax():
    rng = np.random.default_rng(4)
    logits = jnp.asarray(rng.normal(0.0, 1.0, (3, 2)).astype(np.float32))
    y = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    f = FocalLoss(gamma=2.0, alpha=0.25)
    _ = f.forward(logits, y)
    g = np.asarray(f.backward())

    def loss(z):
        return focal_forward(z, y, 2.0, 0.25)

    g_ref = np.asarray(jax.grad(loss)(logits))
    print(f"Focal grad diff: {np.max(np.abs(g - g_ref)):.2e}")
    assert np.allclose(g, g_ref, atol=1e-5)


def test_value_errors():
    with pytest.raises(ValueError):
        LabelSmoothingCE(-0.1)
    with pytest.raises(ValueError):
        LabelSmoothingCE(1.0)
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(alpha=-0.1)
    with pytest.raises(ValueError):
        FocalLoss(alpha=1.1)


def test_backward_before_forward_raises():
    for inst in [
        L1Loss(),
        LabelSmoothingCE(0.1),
        KLDivLoss(),
        HingeLoss(),
        FocalLoss(),
    ]:
        with pytest.raises(RuntimeError):
            inst.backward()
