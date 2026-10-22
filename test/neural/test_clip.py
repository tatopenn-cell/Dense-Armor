"""Tests for clip.py and the clip_norm parameter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.clip import clip_grad_norm
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE, SoftmaxCrossEntropy
from dense_armor.utility.neural.online import OnlineNetClassifier, OnlineNetRegressor
from dense_armor.utility.neural.online_autoencoder import OnlineAutoencoder
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

W_TRAINER = [
    [
        [1.5797994136810303, 1.9963921308517456, -0.4169407784938812],
        [-0.03376825153827667, 0.38205891847610474, -1.1690658330917358],
    ],
    [-0.040536124259233475, -0.028111785650253296, 0.011090670712292194],
    [[0.07026466727256775], [0.4783298671245575], [-0.7056853771209717]],
    [0.12216528505086899],
]
W_REGRESSOR = [[[1.0374104976654053], [1.8503371477127075]], [0.03244306519627571]]
W_CLASSIFIER = [
    [
        [1.2392603158950806, 2.01365327835083, -0.03860075771808624],
        [-0.19496333599090576, 0.22495244443416595, -0.9046047329902649],
    ],
    [0.04462721198797226, 0.09335601329803467, -0.13798320293426514],
]
W_AUTOENCODER = [
    [[1.3338395357131958], [1.82252836227417]],
    [0.20452862977981567],
    [[0.35472267866134644, 0.3120631277561188]],
    [0.22590959072113037, 0.07189716398715973],
]


def _fixed_data():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (8, 2)).astype(np.float32)
    y = (X[:, 0:1] + 2 * X[:, 1:2]).astype(np.float32)
    return X, y


def test_clip_hand_case():
    p = Param(jnp.array([0.0, 0.0]))
    p.grad = jnp.array([3.0, 4.0])
    norm = clip_grad_norm([p], 1.0)
    assert abs(norm - 5.0) < 1e-6
    assert np.allclose(np.asarray(p.grad), [0.6, 0.8], atol=1e-6)


def test_clip_below_threshold_keeps_gradient():
    p = Param(jnp.array([0.0, 0.0]))
    p.grad = jnp.array([3.0, 4.0])
    norm = clip_grad_norm([p], 10.0)
    assert abs(norm - 5.0) < 1e-6
    assert np.allclose(np.asarray(p.grad), [3.0, 4.0], atol=1e-9)


def test_clip_global_norm_two_params():
    p1 = Param(jnp.array([0.0]))
    p1.grad = jnp.array([3.0])
    p2 = Param(jnp.array([0.0]))
    p2.grad = jnp.array([4.0])
    norm = clip_grad_norm([p1, p2], 1.0)
    assert abs(norm - 5.0) < 1e-6
    assert np.allclose(np.asarray(p1.grad), [0.6], atol=1e-6)
    assert np.allclose(np.asarray(p2.grad), [0.8], atol=1e-6)


def test_clip_zero_gradient_stays_zero():
    p = Param(jnp.array([0.0, 0.0]))
    p.grad = jnp.array([0.0, 0.0])
    norm = clip_grad_norm([p], 1.0)
    assert norm == 0.0
    assert np.allclose(np.asarray(p.grad), [0.0, 0.0])


def test_clip_at_boundary_keeps_norm():
    p = Param(jnp.array([0.0, 0.0]))
    p.grad = jnp.array([0.6, 0.8])
    norm = clip_grad_norm([p], 1.0)
    assert abs(norm - 1.0) < 1e-6
    new_norm = float(np.linalg.norm(np.asarray(p.grad)))
    assert abs(new_norm - 1.0) < 1e-6


def test_clip_value_error():
    p = Param(jnp.array([0.0]))
    p.grad = jnp.array([1.0])
    with pytest.raises(ValueError):
        clip_grad_norm([p], 0.0)
    with pytest.raises(ValueError):
        clip_grad_norm([p], -1.0)


def _fit_variant(clip):
    X, y = _fixed_data()
    m = Sequential(
        [Dense(2, 3, jax.random.PRNGKey(0)), Dense(3, 1, jax.random.PRNGKey(1))]
    )
    t = Trainer(m, MSE(), SGD(0.1), clip_norm=clip)
    t.fit(X, y, epochs=3, batch_size=4, key=jax.random.PRNGKey(2), shuffle=False)
    return [np.asarray(p.value).copy() for p in m.params()]


def _regressor_variant(clip):
    X, y = _fixed_data()
    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    on = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["a", "b"], clip_norm=clip)
    for i in range(8):
        on.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(y[i, 0]))
    return [np.asarray(p.value).copy() for p in on.model_.params()]


def _classifier_variant(clip):
    X, y = _fixed_data()
    y_3c = (np.abs(y[:, 0]) * 10).astype(int) % 3
    m = Sequential([Dense(2, 3, jax.random.PRNGKey(0))])
    on = OnlineNetClassifier(
        m,
        SoftmaxCrossEntropy(),
        SGD(0.1),
        classes=[0, 1, 2],
        feature_keys=["a", "b"],
        clip_norm=clip,
    )
    for i in range(8):
        on.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y_3c[i]))
    return [np.asarray(p.value).copy() for p in on.model_.params()]


def _autoencoder_variant(clip):
    X, _ = _fixed_data()
    enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
    on = OnlineAutoencoder(
        Autoencoder(enc, dec), SGD(0.1), feature_keys=["a", "b"], clip_norm=clip
    )
    for i in range(8):
        on.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])})
    return [np.asarray(p.value).copy() for p in on.model_.params()]


def test_trainer_clip_norm_none_unchanged():
    a = _fit_variant(clip=None)
    b = _fit_variant(clip=None)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)


def test_online_regressor_clip_norm_none_unchanged():
    a = _regressor_variant(clip=None)
    b = _regressor_variant(clip=None)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)


def test_online_classifier_clip_norm_none_unchanged():
    a = _classifier_variant(clip=None)
    b = _classifier_variant(clip=None)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)


def test_online_autoencoder_clip_norm_none_unchanged():
    a = _autoencoder_variant(clip=None)
    b = _autoencoder_variant(clip=None)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)


def test_trainer_clip_norm_bounds_step():
    X, y = _fixed_data()
    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    before = [np.asarray(p.value).copy() for p in m.params()]
    t = Trainer(m, MSE(), SGD(1.0), clip_norm=0.5)
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    delta_sq = sum(
        float(((np.asarray(p.value) - b) ** 2).sum())
        for p, b in zip(m.params(), before)
    )
    delta = float(np.sqrt(delta_sq))
    assert delta <= 0.5 + 1e-6, f"delta {delta} > 0.5 + 1e-6"


def test_online_regressor_clip_norm_bounds_step():
    X, y = _fixed_data()
    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    on = OnlineNetRegressor(m, MSE(), SGD(1.0), feature_keys=["a", "b"], clip_norm=0.5)
    before = [np.asarray(p.value).copy() for p in on.model_.params()]
    on.learn_one({"a": float(X[0, 0]), "b": float(X[0, 1])}, float(y[0, 0]))
    delta_sq = sum(
        float(((np.asarray(p.value) - b) ** 2).sum())
        for p, b in zip(on.model_.params(), before)
    )
    delta = float(np.sqrt(delta_sq))
    assert delta <= 0.5 + 1e-6, f"delta {delta} > 0.5 + 1e-6"


def test_exploding_gradient_clipped():
    from dense_armor.utility.neural.recurrent import RNN

    r = RNN(1, 1, jax.random.PRNGKey(0))
    r.W.value = jnp.array([[1.0]])
    r.U.value = jnp.array([[1.5]])
    r.b.value = jnp.array([0.0])
    x = jnp.full((40, 1), 1e-6)
    g = jnp.ones((40, 1))
    for p in r.params():
        p.zero_grad()
    _ = r.forward(x, training=True)
    r.backward(g)
    pre = float(
        np.sqrt(sum(float((np.asarray(p.grad) ** 2).sum()) for p in r.params()))
    )
    print(f"pre-clip norm: {pre:.3e}")
    assert pre > 1000.0, f"pre-clip norm {pre} not > 1000"
    clip_grad_norm(r.params(), 1.0)
    post = float(
        np.sqrt(sum(float((np.asarray(p.grad) ** 2).sum()) for p in r.params()))
    )
    print(f"post-clip norm: {post:.6f}")
    assert abs(post - 1.0) < 1e-5, f"post-clip norm {post} != 1.0"


def test_clip_norm_value_errors():
    X, y = _fixed_data()
    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    with pytest.raises(ValueError):
        Trainer(m, MSE(), SGD(0.1), clip_norm=0.0)
    with pytest.raises(ValueError):
        Trainer(m, MSE(), SGD(0.1), clip_norm=-1.0)
    with pytest.raises(ValueError):
        OnlineNetRegressor(m, MSE(), SGD(0.1), clip_norm=0.0)
    m2 = Sequential([Dense(2, 3, jax.random.PRNGKey(0))])
    with pytest.raises(ValueError):
        OnlineNetClassifier(
            m2, SoftmaxCrossEntropy(), SGD(0.1), classes=[0, 1, 2], clip_norm=0.0
        )
    enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
    with pytest.raises(ValueError):
        OnlineAutoencoder(Autoencoder(enc, dec), SGD(0.1), clip_norm=0.0)
