"""Hand cases of the vision classes of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.vision as V
from dense_armor.ml.vision import (
    PCA,
    DenseFlow,
    GlobalMotion,
    MotionMask,
    RandomProjection,
)


def test_doctests():
    r = doctest.testmod(V)
    assert r.failed == 0 and r.attempted >= 5


def test_flow_zero_and_normalisation():
    a = np.arange(36.0).reshape(6, 6)
    m = DenseFlow().fit(a, a)
    assert not m.u_.any() and not m.v_.any() and not m.mag_.any()
    s = np.tile(np.sin(2 * np.pi * np.arange(16) / 16), (16, 1))
    f1 = DenseFlow(alpha=0.1, n_iter=60, normalize_intensity=True).fit(
        s, np.roll(s, 1, axis=1)
    )
    f2 = DenseFlow(alpha=0.1, n_iter=60, normalize_intensity=True).fit(
        50 * s, 50 * np.roll(s, 1, axis=1)
    )
    assert np.allclose(f1.u_, f2.u_, atol=1e-6)


def test_flow_one_step_hand_case():
    a = np.zeros((4, 4))
    a[:, 1] = 1.0
    b = np.roll(a, 1, axis=1)
    m = DenseFlow(alpha=1.0, n_iter=1).fit(a, b)
    Ix = np.zeros((4, 4))
    Ix[:, :-1] = a[:, 1:] - a[:, :-1]
    It = b - a
    assert np.allclose(m.u_, -Ix * It / (1.0 + Ix**2 + 1e-9))
    assert not m.v_.any()


def test_mask_hand_case():
    u = np.tile([0.1, 0.0], (8, 4))
    u[3:5, 3:5] = 2.0
    z = np.zeros((8, 8))
    m = MotionMask(dilation=0).fit(u, z)
    assert m.mask_.sum() == 4 and m.mask_[3:5, 3:5].all()
    d = MotionMask(dilation=1).fit(u, z)
    assert d.mask_.sum() == 12
    assert not MotionMask().fit(z, z).mask_.any()


def test_global_motion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 32))
    m = GlobalMotion(refine=False).fit(a, np.roll(a, (3, -2), axis=(0, 1)))
    assert (m.dx_, m.dy_) == (-2.0, 3.0) and m.peak_ > 0.9
    h = GlobalMotion(window=16).fit(a, np.roll(a, (1, 1), axis=(0, 1)))
    assert abs(h.dx_ - 1) < 1.0
    z = GlobalMotion().fit(np.ones((8, 8)), np.ones((8, 8)))
    assert (z.dx_, z.dy_, z.peak_) == (0.0, 0.0, 0.0)


def test_random_projection():
    X = np.arange(12.0).reshape(2, 6)
    m = RandomProjection(k=3, seed=0).fit(X)
    W = np.random.default_rng(0).standard_normal((6, 3)) / np.sqrt(3)
    assert np.allclose(m.W_, W) and np.allclose(m.transform(X), X @ W)
    big = np.random.default_rng(1).normal(size=(2, 4000))
    p = RandomProjection(k=400, seed=2).fit(big).transform(big)
    assert abs(np.linalg.norm(p[0]) / np.linalg.norm(big[0]) - 1) < 0.15
    with pytest.raises(ValueError):
        RandomProjection(k=0)


def test_pca_hand_case():
    X = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.1], [-2.0, -0.1]]
    m = PCA(k=2).fit(X)
    assert np.allclose(m.mean_, 0.0)
    assert np.allclose(m.explained_variance_.sum(), 2.505)
    assert np.allclose(m.components_ @ m.components_.T, np.eye(2))
    assert np.allclose(m.transform(X)[:, 0], np.array(X) @ m.components_[0])
    with pytest.raises(ValueError):
        PCA(k=0)


def test_stopped():
    assert len(V.STOPPED) == 23 and all(V.STOPPED.values())
