"""Tests for trainer.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.layers import Dropout
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.schedules import (
    ConstantLR,
    CosineWarmRestarts,
)
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam, SGD


def _data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0:1] + 2 * X[:, 1:2]).astype(np.float32)
    return X, y


def test_trainer_history_has_three_keys():
    X, y = _data()
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01))
    t.fit(X, y, epochs=3, batch_size=5, key=jax.random.PRNGKey(1))
    assert list(t.history_.keys()) == ["train_loss", "val_loss", "lr"]
    assert len(t.history_["train_loss"]) == 3
    assert len(t.history_["lr"]) == 3


def test_default_schedule_is_constant_lr_of_optimizer():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.05))
    assert isinstance(t.schedule, ConstantLR)
    assert t.schedule.lr == 0.05


def test_full_batch_matches_manual_loop():
    X, y = _data(8)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.1))
    t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    loss = MSE()
    opt = SGD(0.1)
    Xj = jnp.asarray(X)
    yj = jnp.asarray(y)
    for _ in range(3):
        m2.zero_grad()
        pred = m2.forward(Xj, training=True)
        _ = loss.forward(pred, yj)
        m2.backward(loss.backward())
        for p in m2.params():
            p.update(opt)
    w_m = [np.asarray(p.value).copy() for p in m2.params()]

    for a, b in zip(w_t, w_m):
        assert np.allclose(a, b, atol=1e-6)


def test_one_sample_matches_manual_loop():
    X, y = _data(4)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.1))
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    loss = MSE()
    opt = SGD(0.1)
    for i in range(4):
        m2.zero_grad()
        pred = m2.forward(jnp.asarray(X[i:i+1]), training=True)
        _ = loss.forward(pred, jnp.asarray(y[i:i+1]))
        m2.backward(loss.backward())
        for p in m2.params():
            p.update(opt)
    w_m = [np.asarray(p.value).copy() for p in m2.params()]

    for a, b in zip(w_t, w_m):
        assert np.allclose(a, b, atol=1e-6)


def test_l2_equals_decoupled_with_factor_lambda_over_alpha():
    X, y = _data(10)
    lam = 0.1
    alpha = 0.1

    m_l2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_l2 = Trainer(m_l2, MSE(), SGD(alpha),
                   weight_decay=lam / alpha, decay="l2")
    t_l2.fit(X, y, epochs=2, batch_size=10, key=jax.random.PRNGKey(0),
             shuffle=False)
    w_l2 = [np.asarray(p.value).copy() for p in m_l2.params()]

    m_dc = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t_dc = Trainer(m_dc, MSE(), SGD(alpha),
                   weight_decay=lam, decay="decoupled")
    t_dc.fit(X, y, epochs=2, batch_size=10, key=jax.random.PRNGKey(0),
             shuffle=False)
    w_dc = [np.asarray(p.value).copy() for p in m_dc.params()]

    for a, b in zip(w_l2, w_dc):
        assert np.allclose(a, b, atol=1e-5)


def test_decoupled_shrinks_weights_by_hand():
    X = jnp.zeros((2, 2))
    y = jnp.zeros((2, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    W0 = np.asarray(m.params()[0].value).copy()
    t = Trainer(m, MSE(), SGD(0.1), weight_decay=0.1, decay="decoupled")
    t.fit(X, y, epochs=1, batch_size=2, key=jax.random.PRNGKey(0),
          shuffle=False)
    W1 = np.asarray(m.params()[0].value)
    expected = W0 * (1 - 0.1 * 1.0) - 0.1 * 0.0
    assert np.allclose(W1, expected, atol=1e-6)


def test_l2_does_not_touch_biases():
    X = jnp.zeros((2, 2))
    y = jnp.zeros((2, 1))
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    b0 = np.asarray(m.params()[1].value).copy()
    t = Trainer(m, MSE(), SGD(0.1), weight_decay=1.0, decay="l2")
    t.fit(X, y, epochs=1, batch_size=2, key=jax.random.PRNGKey(0),
          shuffle=False)
    b1 = np.asarray(m.params()[1].value)
    assert np.allclose(b1, b0, atol=1e-8)


def test_schedule_in_history_follows_cosine():
    X, y = _data(10)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    c = CosineWarmRestarts(0.05, 0.0, 5)
    t = Trainer(m, MSE(), SGD(0.01), schedule=c)
    t.fit(X, y, epochs=3, batch_size=10, key=jax.random.PRNGKey(0),
          shuffle=False)
    for e in range(3):
        assert abs(t.history_["lr"][e] - c(e)) < 1e-12


def test_early_stopping_restores_best_weights():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0.0, 1.0, (30, 2)).astype(np.float32)
    y_tr = rng.normal(0.0, 1.0, (30, 1)).astype(np.float32)
    X_val = rng.normal(0.0, 1.0, (30, 2)).astype(np.float32)
    y_val = rng.normal(0.0, 1.0, (30, 1)).astype(np.float32)

    m = mlp([2, 32, 32, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), Adam(0.01))
    t.fit(X_tr, y_tr, epochs=200, batch_size=10, key=jax.random.PRNGKey(0),
          X_val=X_val, y_val=y_val, patience=5)
    print(f"stopped at {t.stopped_epoch_}, best at {t.best_epoch_}")
    assert t.stopped_epoch_ < 200
    best_val = t.history_["val_loss"][t.best_epoch_]
    assert abs(t.evaluate(X_val, y_val) - best_val) < 1e-6


def test_without_patience_runs_all_epochs():
    X, y = _data(10)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01))
    t.fit(X, y, epochs=10, batch_size=5, key=jax.random.PRNGKey(0))
    assert len(t.history_["train_loss"]) == 10


def test_same_key_twice_gives_same_history():
    X, y = _data(10)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t1 = Trainer(m1, MSE(), SGD(0.01))
    t1.fit(X, y, epochs=3, batch_size=5, key=jax.random.PRNGKey(42))
    m2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t2 = Trainer(m2, MSE(), SGD(0.01))
    t2.fit(X, y, epochs=3, batch_size=5, key=jax.random.PRNGKey(42))
    assert np.allclose(t1.history_["train_loss"], t2.history_["train_loss"])


def test_dropout_is_active_in_fit_and_off_in_evaluate():
    X, y = _data(20)
    d = Dropout(0.5, jax.random.PRNGKey(0))
    inner = mlp([2, 4, 1], jax.random.PRNGKey(0))

    class WithDropout:
        def forward(self, x, training=True):
            return d.forward(inner.forward(x, training=training), training=training)

        def backward(self, g):
            return inner.backward(d.backward(g))

        def params(self):
            return inner.params()

        def zero_grad(self):
            for p in inner.params():
                p.zero_grad()

    t = Trainer(WithDropout(), MSE(), SGD(0.01))
    v1 = t.evaluate(X, y)
    v2 = t.evaluate(X, y)
    assert v1 == v2


def test_predict_returns_model_output():
    X, y = _data(5)
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01))
    out = t.predict(X)
    expected = m.forward(X, training=False)
    assert np.allclose(np.asarray(out), np.asarray(expected))


def test_value_errors():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="decay must be"):
        Trainer(m, MSE(), SGD(0.1), decay="foo")
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        Trainer(m, MSE(), SGD(0.1), weight_decay=-1.0)

    t = Trainer(m, MSE(), SGD(0.1))
    X, y = _data(5)
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        t.fit(X, y, epochs=0, batch_size=2, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="different lengths"):
        t.fit(X, y[:3], epochs=1, batch_size=2, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="patience requires"):
        t.fit(X, y, epochs=1, batch_size=2, key=jax.random.PRNGKey(0),
              patience=3)
    with pytest.raises(ValueError, match="patience must be at least 1"):
        t.fit(X, y, epochs=1, batch_size=2, key=jax.random.PRNGKey(0),
              X_val=X, y_val=y, patience=0)


def test_two_moons_reaches_accuracy():
    rng = np.random.default_rng(42)
    n = 100
    a = rng.normal([-0.5, 0.5], 0.15, (n // 2, 2)).astype(np.float32)
    b = rng.normal([0.5, -0.5], 0.15, (n // 2, 2)).astype(np.float32)
    X = np.vstack([a, b])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float32)
    y = y.reshape(-1, 1)

    m = mlp([2, 16, 16, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), Adam(0.01))
    t.fit(X, y, epochs=200, batch_size=20, key=jax.random.PRNGKey(0))
    pred = np.asarray(t.predict(X)).reshape(-1)
    acc = float(((pred > 0.5) == (y.reshape(-1) > 0.5)).mean())
    print(f"accuracy: {acc:.4f}")
    assert acc >= 0.9
