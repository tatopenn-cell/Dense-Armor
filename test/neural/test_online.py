"""Tests for online.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE, BCEWithLogits
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.online import (
    OnlineNetClassifier,
    OnlineNetRegressor,
)
from dense_armor.utility.neural.schedules import ConstantLR, CosineWarmRestarts
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, Adam


def _data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0:1] + 0.5 * X[:, 1:2]).astype(np.float32)
    dicts = [{"a": float(X[i, 0]), "b": float(X[i, 1])} for i in range(n)]
    return X, y, dicts


def test_online_sgd_matches_trainer():
    X, y, dicts = _data()
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, MSE(), SGD(0.1))
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r = OnlineNetRegressor(m2, MSE(), SGD(0.1), feature_keys=["a", "b"])
    for d, yi in zip(dicts, y.reshape(-1)):
        r.learn_one(d, float(yi))
    w_o = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_online_adam_matches_trainer():
    X, y, dicts = _data()
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, MSE(), Adam(0.01))
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r = OnlineNetRegressor(m2, MSE(), Adam(0.01), feature_keys=["a", "b"])
    for d, yi in zip(dicts, y.reshape(-1)):
        r.learn_one(d, float(yi))
    w_o = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_online_decoupled_matches_trainer():
    X, y, dicts = _data(10)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, MSE(), SGD(0.1), weight_decay=0.1, decay="decoupled")
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r = OnlineNetRegressor(
        m2,
        MSE(),
        SGD(0.1),
        feature_keys=["a", "b"],
        weight_decay=0.1,
        decay="decoupled",
    )
    for d, yi in zip(dicts, y.reshape(-1)):
        r.learn_one(d, float(yi))
    w_o = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_online_hand_step():
    d = Dense(1, 1, jax.random.PRNGKey(0))
    d.W.value = jnp.array([[2.0]])
    d.b.value = jnp.array([0.0])
    r = OnlineNetRegressor(d, MSE(), SGD(0.1), feature_keys=["x"])
    r.learn_one({"x": 1.0}, 0.0)
    assert abs(float(r.model_.params()[0].value[0, 0]) - 1.6) < 1e-6
    assert abs(float(r.model_.params()[1].value[0]) - (-0.4)) < 1e-6


def test_online_hand_step_l2():
    d = Dense(1, 1, jax.random.PRNGKey(0))
    d.W.value = jnp.array([[2.0]])
    d.b.value = jnp.array([0.0])
    r = OnlineNetRegressor(
        d, MSE(), SGD(0.1), feature_keys=["x"], weight_decay=0.1, decay="l2"
    )
    r.learn_one({"x": 1.0}, 0.0)
    expected_w = 2.0 - 0.1 * (4.0 + 0.1 * 2.0)
    assert abs(float(r.model_.params()[0].value[0, 0]) - expected_w) < 1e-6


def test_online_schedule_lr():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    sched = CosineWarmRestarts(0.1, 0.0, 10)
    r = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b"], schedule=sched)
    for i in range(6):
        r.learn_one({"a": float(i), "b": 0.0}, 0.0)
    lr5 = r.schedule(float(5))
    assert abs(lr5 - 0.05) < 1e-6


def test_missing_values():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["a", "b"])
    w0 = [np.asarray(p.value).copy() for p in r.model_.params()]
    r.learn_one({"a": 1.0}, 0.5)
    r.learn_one({"a": 1.0, "b": float("nan")}, 0.5)
    r.learn_one({"a": 1.0, "b": "x"}, 0.5)
    r.learn_one({"a": 1.0, "b": 0.0}, float("nan"))
    r.learn_one(None, 0.5)
    assert r.n_missing_ == 5
    w1 = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w0, w1):
        assert np.allclose(a, b)


def test_classifier_proba_one_logit():
    d = Dense(1, 1, jax.random.PRNGKey(0))
    d.W.value = jnp.array([[1.0]])
    d.b.value = jnp.array([0.0])
    c = OnlineNetClassifier(
        d, BCEWithLogits(), SGD(0.1), classes=[0, 1], feature_keys=["x"]
    )
    logit = 1.0
    p1 = 1.0 / (1.0 + np.exp(-logit))
    proba = c.predict_proba_one({"x": 1.0})
    assert abs(proba[0] - (1.0 - p1)) < 1e-5
    assert abs(proba[1] - p1) < 1e-5


def test_classifier_argmax_three_logits():
    from dense_armor.utility.neural.losses import SoftmaxCrossEntropy

    d = Dense(1, 3, jax.random.PRNGKey(0))
    d.W.value = jnp.array([[0.1, 0.5, 0.3]])
    d.b.value = jnp.zeros(3)
    c = OnlineNetClassifier(
        d, SoftmaxCrossEntropy(), SGD(0.1), classes=["x", "y", "z"], feature_keys=["a"]
    )
    assert c.predict_one({"a": 1.0}) == "y"


def test_progressive_validation_improves():
    rng = np.random.default_rng(3)
    n = 1000
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    w_true = np.array([2.0, -1.0], dtype=np.float32)
    y = (X @ w_true + rng.normal(0.0, 0.1, n)).astype(np.float32)

    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b"])
    first_errors, last_errors = [], []
    for i in range(n):
        d = {"a": float(X[i, 0]), "b": float(X[i, 1])}
        pred = r.predict_one(d)
        err = (pred - y[i]) ** 2
        if i < 100:
            first_errors.append(err)
        elif i >= n - 100:
            last_errors.append(err)
        r.learn_one(d, float(y[i]))
    print(f"first-100 MSE: {np.mean(first_errors):.4f}")
    print(f"last-100 MSE:  {np.mean(last_errors):.4f}")
    assert np.mean(last_errors) < np.mean(first_errors)


def test_classifier_accuracy():
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    m = mlp([2, 8, 1], jax.random.PRNGKey(0))
    c = OnlineNetClassifier(
        m, BCEWithLogits(), SGD(0.05), classes=[0, 1], feature_keys=["a", "b"]
    )
    last_ok = 0
    last_n = 200
    for i in range(n):
        d = {"a": float(X[i, 0]), "b": float(X[i, 1])}
        pred = c.predict_one(d)
        if i >= n - last_n and pred == int(y[i]):
            last_ok += 1
        c.learn_one(d, int(y[i]))
    acc = last_ok / last_n
    print(f"last-200 accuracy: {acc:.4f}")
    assert acc >= 0.9


def test_check_estimator_regressor():
    from dense_armor.checks import check_estimator

    m = mlp([3, 4, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b", "c"])
    out = check_estimator(r)
    assert out["not_applicable"] == []
    assert r._unit_test_skips() == set()


def test_check_estimator_classifier():
    from dense_armor.checks import check_estimator

    m = mlp([3, 1], jax.random.PRNGKey(0))
    c = OnlineNetClassifier(
        m, BCEWithLogits(), SGD(0.01), classes=[0, 1], feature_keys=["a", "b", "c"]
    )
    out = check_estimator(c)
    assert out["not_applicable"] == []
    assert c._unit_test_skips() == set()


def test_deterministic():
    X, y, dicts = _data(20)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r1 = OnlineNetRegressor(m1, MSE(), SGD(0.1), feature_keys=["a", "b"])
    for d, yi in zip(dicts, y.reshape(-1)):
        r1.learn_one(d, float(yi))
    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r2 = OnlineNetRegressor(m2, MSE(), SGD(0.1), feature_keys=["a", "b"])
    for d, yi in zip(dicts, y.reshape(-1)):
        r2.learn_one(d, float(yi))
    assert r1.n_seen_ == r2.n_seen_
    for p1, p2 in zip(r1.model_.params(), r2.model_.params()):
        assert np.allclose(np.asarray(p1.value), np.asarray(p2.value))


def test_eqwrap_hash_and_repr():
    from dense_armor.utility.neural.online import _EqWrap

    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    w = _EqWrap(m)
    assert isinstance(hash(w), int)
    assert repr(w) == repr(m)


def test_values_equal_handles_wraps_and_types():
    from dense_armor.utility.neural.online import _EqWrap, _values_equal

    a = _EqWrap(ConstantLR(0.1))
    b = ConstantLR(0.1)
    assert _values_equal(a, b)
    assert _values_equal(b, a)
    assert not _values_equal(1, "1")


def test_values_equal_param_counts_differ():
    from dense_armor.utility.neural.online import _values_equal

    m1 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    m2 = mlp([3, 4, 1], jax.random.PRNGKey(0))
    assert not _values_equal(m1, m2)


def test_value_errors_both_classes():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="decay must be"):
        OnlineNetRegressor(m, MSE(), SGD(0.1), decay="bad")
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        OnlineNetRegressor(m, MSE(), SGD(0.1), weight_decay=-1.0)
    with pytest.raises(ValueError, match="decay must be"):
        OnlineNetClassifier(m, BCEWithLogits(), SGD(0.1), classes=[0, 1], decay="bad")
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        OnlineNetClassifier(
            m, BCEWithLogits(), SGD(0.1), classes=[0, 1], weight_decay=-1.0
        )
    with pytest.raises(ValueError, match="classes needs at least two"):
        OnlineNetClassifier(m, BCEWithLogits(), SGD(0.1), classes=[0])


def test_first_x_not_dict_and_y_not_in_classes():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1))
    r.learn_one(None, 0.0)
    assert r.n_missing_ == 1
    r.learn_one({"a": 1.0, "b": 0.0}, 0.0)
    assert r.n_seen_ == 1

    c = OnlineNetClassifier(
        mlp([2, 1], jax.random.PRNGKey(0)), BCEWithLogits(), SGD(0.1), classes=[0, 1]
    )
    c.learn_one({"a": 1.0, "b": 0.0}, 2)
    assert c.n_missing_ == 1
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.n_seen_ == 1


def test_predict_before_first_learn():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1))
    assert r.predict_one({"a": 1.0, "b": 0.0}) == 0.0
    assert r.predict_one({"a": 1.0, "b": 0.0}, return_std=True) == (0.0, 0.0)

    c = OnlineNetClassifier(
        mlp([2, 1], jax.random.PRNGKey(0)), BCEWithLogits(), SGD(0.1), classes=[0, 1]
    )
    assert c.predict_proba_one({"a": 1.0}) == {}
    assert c.predict_one({"a": 1.0}) is None


def test_predict_with_missing_feature():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["a", "b"])
    r.learn_one({"a": 1.0, "b": 0.0}, 0.5)
    assert r.predict_one({}) == 0.0
    assert r.predict_one({}, return_std=True) == (0.0, 0.0)
    out = r.predict_one({"a": 1.0, "b": 0.0}, return_std=True)
    assert len(out) == 2

    c = OnlineNetClassifier(
        m, BCEWithLogits(), SGD(0.1), classes=[0, 1], feature_keys=["a", "b"]
    )
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.predict_proba_one({}) == {}


def test_multiclass_target_and_softmax():
    from dense_armor.utility.neural.losses import SoftmaxCrossEntropy

    m = mlp([2, 3], jax.random.PRNGKey(0))
    c = OnlineNetClassifier(
        m,
        SoftmaxCrossEntropy(),
        SGD(0.1),
        classes=["x", "y", "z"],
        feature_keys=["a", "b"],
    )
    for _ in range(3):
        c.learn_one({"a": 1.0, "b": 0.0}, "y")
    proba = c.predict_proba_one({"a": 1.0, "b": 0.0})
    assert set(proba) == {"x", "y", "z"}
    assert abs(sum(proba.values()) - 1.0) < 1e-5


def test_decoupled_online():
    X, y, dicts = _data(10)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, MSE(), SGD(0.1), weight_decay=0.1, decay="decoupled")
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r = OnlineNetRegressor(
        m2,
        MSE(),
        SGD(0.1),
        feature_keys=["a", "b"],
        weight_decay=0.1,
        decay="decoupled",
    )
    for d, yi in zip(dicts, y.reshape(-1)):
        r.learn_one(d, float(yi))
    w_o = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_online_l2_online():
    X, y, dicts = _data(10)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, MSE(), SGD(0.1), weight_decay=0.5, decay="l2")
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(42))
    r = OnlineNetRegressor(
        m2, MSE(), SGD(0.1), feature_keys=["a", "b"], weight_decay=0.5, decay="l2"
    )
    for d, yi in zip(dicts, y.reshape(-1)):
        r.learn_one(d, float(yi))
    w_o = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_regressor_y_not_numeric_counts_as_missing():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["a", "b"])
    w0 = [np.asarray(p.value).copy() for p in r.model_.params()]
    r.learn_one({"a": 1.0, "b": 0.0}, None)
    r.learn_one({"a": 1.0, "b": 0.0}, "a")
    r.learn_one({"a": 1.0, "b": 0.0}, float("nan"))
    assert r.n_missing_ == 3
    assert r.n_seen_ == 0
    w1 = [np.asarray(p.value).copy() for p in r.model_.params()]
    for a, b in zip(w0, w1):
        assert np.allclose(a, b)


def test_values_equal_wrap_mismatch_and_param_count():
    from dense_armor.utility.neural.online import _EqWrap, _values_equal

    m1 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    m2 = mlp([3, 4, 1], jax.random.PRNGKey(0))
    assert not _values_equal(m1, m2)
    assert not _values_equal(_EqWrap(1), "1")
    assert _values_equal(_EqWrap(ConstantLR(0.1)), ConstantLR(0.1))


def test_classifier_decoupled_online():
    X, y, dicts = _data(10)
    y_int = np.array([1 if v > 0 else 0 for v in y.reshape(-1)])
    m1 = mlp([2, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, BCEWithLogits(), SGD(0.1), weight_decay=0.1, decay="decoupled")
    t.fit(
        X,
        y_int.reshape(-1, 1),
        epochs=1,
        batch_size=1,
        key=jax.random.PRNGKey(0),
        shuffle=False,
    )
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 1], jax.random.PRNGKey(42))
    c = OnlineNetClassifier(
        m2,
        BCEWithLogits(),
        SGD(0.1),
        classes=[0, 1],
        feature_keys=["a", "b"],
        weight_decay=0.1,
        decay="decoupled",
    )
    for d, yi in zip(dicts, y_int):
        c.learn_one(d, int(yi))
    w_o = [np.asarray(p.value).copy() for p in c.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_classifier_y_not_in_classes():
    m = mlp([2, 1], jax.random.PRNGKey(0))
    c = OnlineNetClassifier(
        m, BCEWithLogits(), SGD(0.1), classes=[0, 1], feature_keys=["a", "b"]
    )
    c.learn_one({"a": 1.0, "b": 0.0}, 5)
    assert c.n_missing_ == 1
    c.learn_one({"a": 1.0, "b": 0.0}, None)
    assert c.n_missing_ == 2
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.n_seen_ == 1


def test_regressor_first_x_not_dict():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1))
    r.learn_one(None, 0.0)
    assert r.n_missing_ == 1
    r.learn_one({"a": 1.0, "b": 0.0}, 0.0)
    assert r.n_seen_ == 1


def test_regressor_read_features_returns_none():
    m = mlp([2, 2, 1], jax.random.PRNGKey(0))
    r = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["a", "b"])
    r.learn_one({"a": 1.0}, 0.0)
    assert r.n_missing_ == 1
    r.learn_one({"a": 1.0, "b": None}, 0.0)
    assert r.n_missing_ == 2
    r.learn_one({"a": 1.0, "b": "not-a-number"}, 0.0)
    assert r.n_missing_ == 3
    r.learn_one({"a": 1.0, "b": 0.0}, 0.0)
    assert r.n_seen_ == 1


def test_values_equal_param_count_differs():
    """Line 90: same type, params() length differs -> False."""
    from dense_armor.utility.neural.online import _values_equal

    class P:
        def __init__(self, v):
            self.value = v

    class M:
        def __init__(self, n):
            self.n = n

        def params(self):
            return [P(0.0) for _ in range(self.n)]

    assert not _values_equal(M(2), M(1))
    assert _values_equal(M(2), M(2))


def test_values_equal_dict_keys_differ():
    """Line 99: same type, __dict__ keys differ -> False."""
    from dense_armor.utility.neural.online import _values_equal

    class D:
        def __init__(self, extra=False):
            self.x = 1
            if extra:
                self.y = 2

    assert not _values_equal(D(False), D(True))
    assert _values_equal(D(False), D(False))


def test_values_equal_nested_values_differ():
    """Line 102: same type, same keys, a nested value differs -> False."""
    from dense_armor.utility.neural.online import _values_equal

    class A:
        def __init__(self, v):
            self.x = v

    assert not _values_equal(A(1), A(2))
    assert _values_equal(A(1), A(1))


def test_classifier_l2_weight_decay_online():
    """Line 396: the classifier L2 branch matches the Trainer."""
    X, y, dicts = _data(10)
    y_int = np.array([1 if v > 0 else 0 for v in y.reshape(-1)])

    m1 = mlp([2, 1], jax.random.PRNGKey(42))
    t = Trainer(m1, BCEWithLogits(), SGD(0.1), weight_decay=0.5, decay="l2")
    t.fit(
        X,
        y_int.reshape(-1, 1),
        epochs=1,
        batch_size=1,
        key=jax.random.PRNGKey(0),
        shuffle=False,
    )
    w_t = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 1], jax.random.PRNGKey(42))
    c = OnlineNetClassifier(
        m2,
        BCEWithLogits(),
        SGD(0.1),
        classes=[0, 1],
        feature_keys=["a", "b"],
        weight_decay=0.5,
        decay="l2",
    )
    for d, yi in zip(dicts, y_int):
        c.learn_one(d, int(yi))
    w_o = [np.asarray(p.value).copy() for p in c.model_.params()]
    for a, b in zip(w_t, w_o):
        assert np.allclose(a, b, atol=1e-6)


def test_classifier_first_x_not_dict():
    """Lines 437-438: first learn with x not a dict -> n_missing."""
    m = mlp([2, 1], jax.random.PRNGKey(0))
    c = OnlineNetClassifier(m, BCEWithLogits(), SGD(0.1), classes=[0, 1])
    c.learn_one(None, 1)
    assert c.n_missing_ == 1
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.n_seen_ == 1
