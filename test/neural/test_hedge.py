"""Tests for hedge.py: HedgeNetClassifier (Sahoo et al. 2018)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.hedge import HedgeNetClassifier


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    return X, y


def test_hedge_alphas_init_uniform():
    c = HedgeNetClassifier([2, 4, 4, 4], jax.random.PRNGKey(0), classes=[0, 1])
    assert c.L_ == 3
    np.testing.assert_allclose(np.asarray(c.alphas_), np.full(4, 0.25), atol=1e-6)
    assert abs(float(np.sum(c.alphas_)) - 1.0) < 1e-6


def test_hedge_alphas_sum_to_one_and_min_bound():
    X, y = _data(200, seed=1)
    c = HedgeNetClassifier(
        [2, 4, 4],
        jax.random.PRNGKey(0),
        classes=[0, 1],
        beta=0.5,
        smoothing=0.2,
        feature_keys=["a", "b"],
    )
    for i in range(200):
        c.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
        assert abs(float(np.sum(c.alphas_)) - 1.0) < 1e-6
        assert float(np.min(np.asarray(c.alphas_))) >= (0.2 / 2) - 1e-9


def test_hedge_hand_round():
    c = HedgeNetClassifier(
        [1, 1],
        jax.random.PRNGKey(0),
        classes=[0, 1],
        beta=0.5,
        smoothing=0.2,
        lr=0.1,
        feature_keys=["a"],
    )
    c.Ws_[0].value = jnp.array([[0.5]])
    c.Thetas_[0].value = jnp.array([[1.0, -1.0]])
    c.Thetas_[1].value = jnp.array([[0.3, -0.3]])
    c.alphas_ = jnp.array([0.5, 0.5])

    x = jnp.array([1.0])
    y_idx = 1

    h1 = float(jax.nn.relu(0.5 * x[0]))
    logits0 = np.asarray(c.Thetas_[0].value.T @ x)
    logits1 = np.asarray(c.Thetas_[1].value.T @ jnp.array([h1]))
    p0 = np.exp(logits0) / np.exp(logits0).sum()
    p1 = np.exp(logits1) / np.exp(logits1).sum()

    a0, a1 = 0.5, 0.5
    L0 = 1.0 - p0[y_idx]
    L1 = 1.0 - p1[y_idx]
    new_a0 = max(a0 * (0.5**L0), 0.2 / 1)
    new_a1 = max(a1 * (0.5**L1), 0.2 / 1)
    Z = new_a0 + new_a1
    expected_alphas = np.array([new_a0 / Z, new_a1 / Z])

    onehot = np.array([0.0, 1.0])
    d_logits_0 = (p0 - onehot) * a0
    d_logits_1 = (p1 - onehot) * a1
    dTheta0 = np.outer(x, d_logits_0)
    dTheta1 = np.outer(np.array([h1]), d_logits_1)
    expected_Theta0 = np.asarray(c.Thetas_[0].value) - 0.1 * dTheta0
    expected_Theta1 = np.asarray(c.Thetas_[1].value) - 0.1 * dTheta1

    d_h1 = np.asarray(c.Thetas_[1].value) @ d_logits_1
    d_pre1 = d_h1 * (h1 > 0)
    dW0 = np.outer(x, d_pre1)
    expected_W0 = np.asarray(c.Ws_[0].value) - 0.1 * dW0

    c.learn_one({"a": float(x[0])}, 1)
    np.testing.assert_allclose(np.asarray(c.alphas_), expected_alphas, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(c.Thetas_[0].value), expected_Theta0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(c.Thetas_[1].value), expected_Theta1, atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(c.Ws_[0].value), expected_W0, atol=1e-5)


def test_hedge_eq3_against_jax_grad():
    c = HedgeNetClassifier(
        [2, 3, 3],
        jax.random.PRNGKey(0),
        classes=[0, 1],
        lr=0.1,
        feature_keys=["a", "b"],
    )
    vec = jnp.array([0.5, -0.3])
    y_idx = 1
    alphas = np.asarray(c.alphas_)

    def loss_of_W0(W0):
        hs = [vec]
        h1 = jax.nn.relu(W0.T @ hs[0])
        hs.append(h1)
        h2 = jax.nn.relu(c.Ws_[1].value.T @ hs[1])
        hs.append(h2)
        total = 0.0
        for j in range(3):
            logits = hs[j] @ c.Thetas_[j].value
            p = jax.nn.softmax(logits)
            total = total + alphas[j] * (-jnp.log(p[y_idx]))
        return total

    grad_W0 = jax.grad(loss_of_W0)(c.Ws_[0].value)
    hs = c._forward_all(vec)
    probs = c._probas_from_hs(hs)
    onehot = jnp.zeros((2,)).at[y_idx].set(1.0)
    dWs = [jnp.zeros_like(w.value) for w in c.Ws_]
    for j in range(c.L_ + 1):
        d_logits = (probs[j] - onehot) * c.alphas_[j]
        d_h = c.Thetas_[j].value @ d_logits
        for lvl in range(j, 0, -1):
            d_pre = d_h * (hs[lvl] > 0).astype(d_h.dtype)
            dWs[lvl - 1] = dWs[lvl - 1] + jnp.outer(hs[lvl - 1], d_pre)
            d_h = c.Ws_[lvl - 1].value @ d_pre
    manual = -0.1 * np.asarray(dWs[0])
    np.testing.assert_allclose(manual, -0.1 * np.asarray(grad_W0), atol=1e-5)


def test_hedge_depth_selection_on_linear_stream():
    rng = np.random.default_rng(2)
    n = 300
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)
    c = HedgeNetClassifier(
        [2, 8, 8, 8],
        jax.random.PRNGKey(0),
        classes=[0, 1],
        beta=0.99,
        lr=0.05,
        smoothing=0.2,
        feature_keys=["a", "b"],
    )
    snapshots = {}
    for i in range(n):
        c.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
        if i + 1 in (1, 10, 100, 300):
            snapshots[i + 1] = np.asarray(c.alphas_).copy()
    print("alphas at 1, 10, 100, 300:")
    for k, v in snapshots.items():
        print(f"  {k}: {np.round(v, 4).tolist()}")
    shallow = float(snapshots[300][0] + snapshots[300][1])
    deepest = float(snapshots[300][3])
    print(f"sum shallow(0,1)={shallow:.4f}, deepest(3)={deepest:.4f}")
    assert shallow > deepest


def test_hedge_learning_accuracy():
    rng = np.random.default_rng(3)
    n = 1000
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)
    c = HedgeNetClassifier(
        [2, 8, 8],
        jax.random.PRNGKey(0),
        classes=[0, 1],
        beta=0.9,
        lr=0.05,
        feature_keys=["a", "b"],
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
    assert acc >= 0.85


def test_hedge_value_errors():
    with pytest.raises(ValueError, match="at least two entries"):
        HedgeNetClassifier([2], jax.random.PRNGKey(0), classes=[0, 1])
    with pytest.raises(ValueError, match="at least two labels"):
        HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0])
    with pytest.raises(ValueError, match="lr must be positive"):
        HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1], lr=0.0)
    with pytest.raises(ValueError, match="beta must be in"):
        HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1], beta=1.5)
    with pytest.raises(ValueError, match="smoothing must be in"):
        HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1], smoothing=0.0)


def test_hedge_missing_values():
    c = HedgeNetClassifier(
        [2, 4], jax.random.PRNGKey(0), classes=[0, 1], feature_keys=["a", "b"]
    )
    w0 = [np.asarray(p.value).copy() for p in c.params()]
    c.learn_one({"a": 1.0}, 1)
    c.learn_one({"a": 1.0, "b": float("nan")}, 1)
    c.learn_one({"a": 1.0, "b": "x"}, 1)
    c.learn_one({"a": 1.0, "b": 0.0}, 5)
    assert c.n_missing_ == 4
    assert c.n_seen_ == 0
    w1 = [np.asarray(p.value).copy() for p in c.params()]
    for a, b in zip(w0, w1):
        assert np.allclose(a, b)


def test_hedge_check_estimator():
    from dense_armor.checks import check_estimator

    c = HedgeNetClassifier(
        [3, 4], jax.random.PRNGKey(0), classes=(0, 1), feature_keys=("a", "b", "c")
    )
    out = check_estimator(c)
    assert out["not_applicable"] == []
    assert c._unit_test_skips() == set()


def test_hedge_deterministic():
    X, y = _data(30, seed=5)
    c1 = HedgeNetClassifier(
        [2, 4, 4], jax.random.PRNGKey(7), classes=[0, 1], feature_keys=["a", "b"]
    )
    c2 = HedgeNetClassifier(
        [2, 4, 4], jax.random.PRNGKey(7), classes=[0, 1], feature_keys=["a", "b"]
    )
    for i in range(30):
        c1.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
        c2.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
    np.testing.assert_allclose(np.asarray(c1.alphas_), np.asarray(c2.alphas_))


def test_hedge_first_x_not_dict_and_read_returns_none():
    c = HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1])
    c.learn_one(None, 1)
    assert c.n_missing_ == 1
    assert c.n_seen_ == 0


def test_hedge_predict_before_learn():
    c = HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1])
    assert c.predict_proba_one({"a": 1.0, "b": 0.0}) == {}
    assert c.predict_one({"a": 1.0, "b": 0.0}) is None

    c2 = HedgeNetClassifier(
        [2, 4], jax.random.PRNGKey(0), classes=[0, 1], feature_keys=["a", "b"]
    )
    proba = c2.predict_proba_one({"a": 1.0, "b": 0.0})
    assert set(proba) == {0, 1}
    assert abs(sum(proba.values()) - 1.0) < 1e-6


def test_hedge_missing_after_learn():
    c = HedgeNetClassifier(
        [2, 4], jax.random.PRNGKey(0), classes=[0, 1], feature_keys=["a", "b"]
    )
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.predict_proba_one({}) == {}
    assert c.predict_one({}) is None


def test_hedge_read_returns_none_on_missing_and_non_numeric():
    c = HedgeNetClassifier(
        [2, 4], jax.random.PRNGKey(0), classes=[0, 1], feature_keys=["a", "b"]
    )
    w0 = [np.asarray(p.value).copy() for p in c.params()]
    c.learn_one({"a": 1.0}, 1)
    c.learn_one({"a": 1.0, "b": None}, 1)
    c.learn_one({"a": 1.0, "b": "x"}, 1)
    c.learn_one({"a": 1.0, "b": float("nan")}, 1)
    assert c.n_missing_ == 4
    w1 = [np.asarray(p.value).copy() for p in c.params()]
    for a, b in zip(w0, w1):
        assert np.allclose(a, b)


def test_hedge_read_x_not_dict():
    """Line 140: _read returns None when x is not a dict."""
    c = HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1])
    assert c._read(None) is None
    assert c._read("not-a-dict") is None
    c.learn_one(None, 1)
    assert c.n_missing_ == 1
    c.learn_one({"a": 1.0, "b": 0.0}, 1)
    assert c.n_seen_ == 1
