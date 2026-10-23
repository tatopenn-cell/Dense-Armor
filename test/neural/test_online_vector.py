"""Tests for online_vector.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.matryoshka import MatryoshkaLoss
from dense_armor.utility.neural.online import OnlineNetRegressor
from dense_armor.utility.neural.online_vector import OnlineNetVector
from dense_armor.utility.neural.quantize import VectorQuantizer
from dense_armor.utility.neural.recurrent import LSTM
from dense_armor.utility.neural.seq2seq import Seq2Seq
from dense_armor.utility.optim import SGD, Adam


def test_hand_case():
    m = Dense(2, 2, jax.random.PRNGKey(0))
    m.W.value = jnp.eye(2)
    m.b.value = jnp.zeros(2)
    on = OnlineNetVector(m, MSE(), SGD(0.1))
    on.learn_one(jnp.array([3.0, 4.0]), jnp.array([3.0, 0.0]))
    print(f"W: {np.asarray(on.model_.W.value).tolist()}")
    print(f"b: {np.asarray(on.model_.b.value).tolist()}")
    assert np.allclose(on.model_.W.value, [[1.0, -1.2], [0.0, -0.6]], atol=1e-6)
    assert np.allclose(on.model_.b.value, [0.0, -0.4], atol=1e-6)


def test_equals_scalar_regressor():
    m1 = Dense(3, 1, jax.random.PRNGKey(0))
    r1 = OnlineNetRegressor(m1, MSE(), SGD(0.01), feature_keys=["a", "b", "c"])
    m2 = Dense(3, 1, jax.random.PRNGKey(0))
    v2 = OnlineNetVector(m2, MSE(), SGD(0.01), feature_keys=["a", "b", "c"])
    for i in range(20):
        d = {"a": float(i), "b": float(i + 1), "c": float(i + 2)}
        y = float(i * 2)
        r1.learn_one(d, y)
        v2.learn_one(d, jnp.array([y]))
    for p1, p2 in zip(r1.model_.params(), v2.model_.params()):
        assert np.allclose(p1.value, p2.value, atol=1e-6), f"{p1.value} vs {p2.value}"


@pytest.mark.training
def test_seq2seq_learning():
    rng = np.random.default_rng(0)
    s = Seq2Seq(
        LSTM(1, 4, jax.random.PRNGKey(0)),
        LSTM(4, 4, jax.random.PRNGKey(1)),
        Dense(4, 1, jax.random.PRNGKey(2)),
        horizon=4,
    )
    on = OnlineNetVector(s, MSE(), Adam(0.01))
    errs = []
    for i in range(300):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        t_src = np.arange(8, dtype=np.float32)
        t_tgt = np.arange(8, 12, dtype=np.float32)
        x = np.sin(t_src + phase) + 0.05 * rng.normal(0.0, 1.0, 8).astype(np.float32)
        y = np.sin(t_tgt + phase)
        pred = on.predict_one(jnp.asarray(x.reshape(8, 1)))
        errs.append(float(np.mean((np.asarray(pred).reshape(-1) - y) ** 2)))
        on.learn_one(jnp.asarray(x.reshape(8, 1)), jnp.asarray(y.reshape(4, 1)))
    first = float(np.mean(errs[:50]))
    last = float(np.mean(errs[-50:]))
    print(f"Seq2Seq online: first={first:.4f} last={last:.4f}")
    assert last < first, f"{last} not < {first}"


@pytest.mark.training
def test_quantizer_autoencoder_learning():
    rng = np.random.default_rng(0)
    m = Sequential(
        [
            Dense(6, 2, jax.random.PRNGKey(0)),
            VectorQuantizer(4, 2, jax.random.PRNGKey(1)),
            Dense(2, 6, jax.random.PRNGKey(2)),
        ]
    )
    on = OnlineNetVector(m, MSE(), Adam(0.01))
    A = rng.normal(0.0, 1.0, (2, 6)).astype(np.float32)
    Z = rng.normal(0.0, 1.0, (400, 2)).astype(np.float32)
    X = Z @ A
    errs = []
    for i in range(400):
        xi = jnp.asarray(X[i])
        pred = on.predict_one(xi)
        errs.append(float(np.mean((np.asarray(pred).reshape(-1) - X[i]) ** 2)))
        on.learn_one(xi, xi)
    first = float(np.mean(errs[:100]))
    last = float(np.mean(errs[-100:]))
    print(f"VQ autoencoder: first={first:.4f} last={last:.4f}")
    assert last < first, f"{last} not < {first}"


@pytest.mark.training
@pytest.mark.training
def test_matryoshka_learning():
    rng = np.random.default_rng(0)
    m = Dense(8, 8, jax.random.PRNGKey(0))
    m.W.value = jnp.eye(8)
    m.b.value = jnp.zeros(8)
    ml = MatryoshkaLoss([2, 4, 8], 8, 4, jax.random.PRNGKey(1))
    on = OnlineNetVector(m, ml, Adam(0.05))
    means = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    y_all = rng.integers(0, 4, 400)
    X_all = np.zeros((400, 8), dtype=np.float32)
    X_all[:, :2] = means[y_all] + 0.3 * rng.normal(0.0, 1.0, (400, 2)).astype(
        np.float32
    )
    X_all[:, 2:] = 0.3 * rng.normal(0.0, 1.0, (400, 6)).astype(np.float32)
    W_before = np.asarray(ml.heads_[2].value).copy()
    for i in range(400):
        on.learn_one(jnp.asarray(X_all[i]), jnp.array(int(y_all[i])))
    W_after = np.asarray(ml.heads_[2].value).copy()
    print(f"head 8 changed by {np.max(np.abs(W_after - W_before)):.3e}")
    assert not np.allclose(W_before, W_after)
    ok = 0
    for i in range(300, 400):
        z_i = on.predict_one(jnp.asarray(X_all[i]))
        logits = ml.logits(jnp.asarray(z_i).reshape(-1), 8)
        if int(jnp.argmax(logits)) == int(y_all[i]):
            ok += 1
    acc = ok / 100
    print(f"head 8 accuracy on last 100: {acc:.3f}")
    assert acc > 0.5, f"accuracy {acc} not > 0.5"


def test_missing_values():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01), feature_keys=["a", "b"])
    on.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    n0 = on.n_seen_
    m0 = on.n_missing_
    for bad_x, bad_y in [
        (None, 3.0),
        ({"a": 1.0}, 3.0),
        ({"a": 1.0, "b": float("nan")}, 3.0),
        ({"a": 1.0, "b": float("inf")}, 3.0),
        ({"a": 1.0, "b": 2.0}, None),
        ({"a": 1.0, "b": 2.0}, float("nan")),
        ({"a": 1.0, "b": 2.0}, float("inf")),
    ]:
        on.learn_one(bad_x, bad_y)
    print(f"n_seen: {on.n_seen_}, n_missing: {on.n_missing_}")
    assert on.n_seen_ == n0
    assert on.n_missing_ == m0 + 7


def test_clip_norm_bounds_step():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(1.0), clip_norm=0.5)
    before = [np.asarray(p.value).copy() for p in on.model_.params()]
    on.learn_one(jnp.array([1.0, 2.0]), jnp.array([3.0]))
    delta_sq = sum(
        float(((np.asarray(p.value) - b) ** 2).sum())
        for p, b in zip(on.model_.params(), before)
    )
    delta = float(np.sqrt(delta_sq))
    print(f"delta after clip: {delta:.6f}")
    assert delta <= 0.5 + 1e-6, f"delta {delta} > 0.5 + 1e-6"


def test_bound_memory_constant():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01), feature_keys=["a", "b"])

    def count(est):
        n = 0
        for p in est.model_.params():
            n += int(p.value.size)
        for opt in est.optimizers_:
            for attr in ("_m", "_v", "m", "v"):
                if hasattr(opt, attr):
                    arr = getattr(opt, attr)
                    if hasattr(arr, "size") and arr.size > 0:
                        n += int(arr.size)
        return n

    for _ in range(200):
        on.learn_one({"a": 1.0, "b": 2.0}, jnp.array([3.0]))
    n_200 = count(on)
    for _ in range(1800):
        on.learn_one({"a": 1.0, "b": 2.0}, jnp.array([3.0]))
    n_2000 = count(on)
    print(f"n_200={n_200} n_2000={n_2000}")
    assert on.n_seen_ == 2000
    assert n_200 == n_2000


def test_check_estimator_passes():
    from dense_armor.checks import check_estimator

    m = Dense(3, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01), feature_keys=["a", "b", "c"])
    out = check_estimator(on)
    print(
        f"check_estimator: ran {len(out['run'])} checks, "
        f"not_applicable {len(out['not_applicable'])}, "
        f"skipped {len(out['skipped'])}"
    )
    assert out["not_applicable"] == []
    assert out["failed"] == []
    assert on._unit_test_skips() == set()


def test_value_errors():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        OnlineNetVector(m, MSE(), SGD(0.1), decay="wrong")
    with pytest.raises(ValueError):
        OnlineNetVector(m, MSE(), SGD(0.1), weight_decay=-1.0)
    with pytest.raises(ValueError):
        OnlineNetVector(m, MSE(), SGD(0.1), clip_norm=0.0)


def test_state_dict_roundtrip():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    for i in range(5):
        on.learn_one({"a": float(i), "b": float(i + 1)}, jnp.array([float(i)]))
    n_seen = on.n_seen_
    assert on.feature_keys_ == ["a", "b"]
    state = on.state_dict()
    assert "model_" in state
    assert "optimizers_" in state
    assert state["n_seen_"] == n_seen
    assert state["feature_keys_"] == ["a", "b"]
    m2 = Dense(2, 1, jax.random.PRNGKey(99))
    on2 = OnlineNetVector(m2, MSE(), SGD(0.01))
    on2.load_state_dict(state)
    assert on2.n_seen_ == n_seen
    assert on2.feature_keys_ == ["a", "b"]
    for p1, p2 in zip(on.model_.params(), on2.model_.params()):
        assert np.allclose(p1.value, p2.value)


def test_weight_decay_l2():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.1), weight_decay=0.1, decay="l2")
    W_before = np.asarray(m.W.value).copy()
    on.learn_one(jnp.array([1.0, 2.0]), jnp.array([0.0]))
    W_after = np.asarray(on.model_.W.value)
    print(f"L2 decay: |dW| = {np.max(np.abs(W_after - W_before)):.4f}")
    assert not np.allclose(W_before, W_after)


def test_weight_decay_decoupled():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.1), weight_decay=0.1, decay="decoupled")
    W_before = np.asarray(m.W.value).copy()
    on.learn_one(jnp.array([1.0, 2.0]), jnp.array([0.0]))
    W_after = np.asarray(on.model_.W.value)
    print(f"decoupled decay: |dW| = {np.max(np.abs(W_after - W_before)):.4f}")
    assert not np.allclose(W_before, W_after)


def test_read_x_bad_array():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    on.learn_one({"a": 1.0, "b": 2.0}, jnp.array([0.0]))
    n0 = on.n_missing_
    on.learn_one("not_an_array", jnp.array([0.0]))
    on.learn_one(jnp.array([float("nan"), 1.0]), jnp.array([0.0]))
    on.learn_one(jnp.array([1.0, 2.0]), "not_a_number")
    print(f"bad inputs: n_missing delta = {on.n_missing_ - n0}")
    assert on.n_missing_ == n0 + 3


def test_read_x_dict_after_first():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    on.learn_one({"a": 1.0, "b": 2.0}, jnp.array([0.0]))
    assert on.feature_keys_ is not None
    on.learn_one({"a": 2.0, "b": 3.0}, jnp.array([1.0]))
    assert on.n_seen_ == 2


def test_predict_one_dict_and_return_std():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    # prima di learn_one: feature_keys_ is None
    y0 = on.predict_one({"a": 1.0, "b": 2.0})
    assert float(jnp.asarray(y0).reshape(-1)[0]) == 0.0
    on.learn_one({"a": 1.0, "b": 2.0}, jnp.array([3.0]))
    y1 = on.predict_one({"a": 1.0, "b": 2.0})
    print(f"predict_one dict: {float(jnp.asarray(y1).reshape(-1)[0]):.4f}")
    assert np.isfinite(float(jnp.asarray(y1).reshape(-1)[0]))
    y_std, s = on.predict_one({"a": 1.0, "b": 2.0}, return_std=True)
    assert s == 0.0
    # dict incompleto
    y_bad = on.predict_one({"a": 1.0})
    assert float(jnp.asarray(y_bad).reshape(-1)[0]) == 0.0


def test_predict_one_bad_array():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    on.learn_one(jnp.array([1.0, 2.0]), jnp.array([3.0]))
    y = on.predict_one("not_an_array")
    assert float(jnp.asarray(y).reshape(-1)[0]) == 0.0
    # return_std con array
    y2, s = on.predict_one(jnp.array([1.0, 2.0]), return_std=True)
    assert s == 0.0


def test_predict_one_empty():
    m = Dense(2, 1, jax.random.PRNGKey(0))
    on = OnlineNetVector(m, MSE(), SGD(0.01))
    y = on.predict_one({"a": 1.0, "b": 2.0})
    arr = jnp.asarray(y)
    assert float(arr.reshape(-1)[0]) == 0.0
