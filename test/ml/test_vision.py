"""Hand cases of the vision classes of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.vision as V
from dense_armor.ml.vision import (
    PCA,
    DenseFlow,
    RandomProjection,
)


def test_doctests():
    r = doctest.testmod(V)
    assert r.failed == 0 and r.attempted >= 3


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


def test_colour_frames_use_the_luminance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 8, 3))
    b = np.roll(a, 1, axis=1)
    lum = lambda f: f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114  # noqa: E731
    c = DenseFlow(n_iter=5).fit(a, b)
    g = DenseFlow(n_iter=5).fit(lum(a), lum(b))
    assert np.allclose(c.u_, g.u_)
    one = DenseFlow(n_iter=5).fit(a[..., :1], b[..., :1])
    assert np.allclose(one.u_, DenseFlow(n_iter=5).fit(a[..., 0], b[..., 0]).u_)


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
    assert len(V.STOPPED) == 25 and all(V.STOPPED.values())
