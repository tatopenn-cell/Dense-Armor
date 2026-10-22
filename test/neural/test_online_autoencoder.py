"""Tests for online_autoencoder.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.roles import AnomalyDetector
from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.online_autoencoder import OnlineAutoencoder
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, Adam


def _hand():
    enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    enc.layers[0].W.value = jnp.array([[1.0], [0.0]])
    enc.layers[0].b.value = jnp.array([0.0])
    dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
    dec.layers[0].W.value = jnp.array([[1.0, 0.0]])
    dec.layers[0].b.value = jnp.array([0.0, 0.0])
    return Autoencoder(enc, dec)


def test_is_anomaly_detector():
    oa = OnlineAutoencoder(_hand(), SGD(0.1), feature_keys=["a", "b"])
    assert isinstance(oa, AnomalyDetector)


def test_hand_case():
    oa = OnlineAutoencoder(_hand(), SGD(0.1), feature_keys=["a", "b"])
    assert abs(oa.score_one({"a": 3.0, "b": 4.0}) - 8.0) < 1e-6

    oa.learn_one({"a": 3.0, "b": 4.0})
    dec_w = np.asarray(oa.model_.decoder.layers[0].W.value)
    dec_b = np.asarray(oa.model_.decoder.layers[0].b.value)
    enc_w = np.asarray(oa.model_.encoder.layers[0].W.value)
    assert np.allclose(dec_w, np.array([[1.0, 1.2]]), atol=1e-6)
    assert np.allclose(dec_b, np.array([0.0, 0.4]), atol=1e-6)
    assert np.allclose(enc_w, np.array([[1.0], [0.0]]), atol=1e-6)

    assert abs(oa.score_one({"a": 3.0, "b": 4.0})) < 1e-6
    assert abs(oa.score_one({"a": 1.0, "b": 0.0}) - 1.28) < 1e-6

    z = oa.encode_one({"a": 3.0, "b": 4.0})
    assert set(z) == {"c0"}
    assert abs(z["c0"] - 3.0) < 1e-6


def test_online_equals_batch():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (5, 2)).astype(np.float32)
    dicts = [{"a": float(X[i, 0]), "b": float(X[i, 1])} for i in range(5)]

    m1 = _hand()
    oa = OnlineAutoencoder(m1, SGD(0.1), feature_keys=["a", "b"])
    for d in dicts:
        oa.learn_one(d)

    m2 = _hand()
    t = Trainer(m2, MSE(), SGD(0.1))
    t.fit(X, X, epochs=1, batch_size=1, key=jax.random.PRNGKey(0), shuffle=False)

    for a, b in zip(oa.model_.params(), m2.params()):
        assert np.allclose(np.asarray(a.value), np.asarray(b.value), atol=1e-6)


def test_missing_values_count():
    oa = OnlineAutoencoder(_hand(), SGD(0.1), feature_keys=["a", "b"])
    w0 = [np.asarray(p.value).copy() for p in oa.model_.params()]
    for bad in [
        {"a": 1.0},
        {"a": 1.0, "b": None},
        {"a": 1.0, "b": "x"},
        {"a": 1.0, "b": float("inf")},
    ]:
        oa.learn_one(bad)
        assert oa.score_one(bad) == 0.0
    assert oa.n_missing_ == 4
    assert oa.n_seen_ == 0
    for a, b in zip([np.asarray(p.value) for p in oa.model_.params()], w0):
        assert np.allclose(a, b)


def _arrays_and_elements(oa):
    n_arrays = 0
    n_elements = 0
    for prm in oa.model_.params():
        n_arrays += 1
        n_elements += int(prm.value.size)
    for opt in oa.optimizers_:
        for attr in ("m", "v", "_m", "_v"):
            if not hasattr(opt, attr):
                continue
            arr = getattr(opt, attr)
            size = int(getattr(arr, "size", 0))
            if size == 0:
                continue
            n_arrays += 1
            n_elements += size
    return n_arrays, n_elements


def test_bound_memory_constant_after_many_samples():
    oa = OnlineAutoencoder(_hand(), Adam(0.01), feature_keys=["a", "b"])
    for _ in range(200):
        oa.learn_one({"a": 1.0, "b": 0.0})
    n_arrays_200, n_elements_200 = _arrays_and_elements(oa)
    for _ in range(1800):
        oa.learn_one({"a": 1.0, "b": 0.0})
    n_arrays_2000, n_elements_2000 = _arrays_and_elements(oa)
    assert oa.n_seen_ == 2000
    assert n_arrays_200 == n_arrays_2000, (
        f"array count grew: {n_arrays_200} -> {n_arrays_2000}"
    )
    assert n_elements_200 == n_elements_2000, (
        f"element count grew: {n_elements_200} -> {n_elements_2000}"
    )
    n_params = sum(int(prm.value.size) for prm in oa.model_.params())
    n_opt_state = n_elements_2000 - n_params
    assert n_opt_state > 0, (
        f"test counts only params ({n_params}), not optimizer state: "
        f"the optimizer must expose _m and _v (Adam) so the count includes state"
    )
    print(
        f"arrays={n_arrays_2000} elements={n_elements_2000} "
        f"params={n_params} opt_state={n_opt_state}"
    )


@pytest.mark.training
def test_detects_a_point_off_the_plane():
    rng = np.random.default_rng(0)
    A = rng.normal(0.0, 1.0, (2, 6)).astype(np.float32)
    Z = rng.normal(0.0, 1.0, (400, 2)).astype(np.float32)
    X = Z @ A

    enc = Sequential([Dense(6, 2, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(2, 6, jax.random.PRNGKey(1))])
    ae = Autoencoder(enc, dec)
    oa = OnlineAutoencoder(ae, Adam(0.01), feature_keys=[f"p{i}" for i in range(6)])
    for row in X:
        oa.learn_one({f"p{i}": float(row[i]) for i in range(6)})

    p_in = {
        "p0": float(X[0, 0]),
        "p1": float(X[0, 1]),
        "p2": float(X[0, 2]),
        "p3": float(X[0, 3]),
        "p4": float(X[0, 4]),
        "p5": float(X[0, 5]),
    }
    _, _, vt = np.linalg.svd(A, full_matrices=True)
    ortho = vt[2:].sum(axis=0)
    ortho = ortho / np.linalg.norm(ortho)
    assert abs(float(np.dot(ortho, A[0]))) < 1e-6

    p_out = {f"p{i}": p_in[f"p{i}"] + float(ortho[i]) for i in range(6)}
    s_in = oa.score_one(p_in)
    s_out = oa.score_one(p_out)
    print(f"score in-plane: {s_in:.4f}, score off-plane: {s_out:.4f}")
    assert s_in < s_out


def test_check_estimator_passes():
    from dense_armor.checks import check_estimator

    enc = Sequential([Dense(3, 2, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(2, 3, jax.random.PRNGKey(1))])
    oa = OnlineAutoencoder(
        Autoencoder(enc, dec), SGD(0.01), feature_keys=("a", "b", "c")
    )
    out = check_estimator(oa)
    assert out["not_applicable"] == []
    assert oa._unit_test_skips() == set()


def test_encode_one_empty_when_cannot_predict():
    oa = OnlineAutoencoder(_hand(), SGD(0.1))
    assert oa.encode_one({"a": 1.0, "b": 0.0}) == {}
    assert oa.score_one({"a": 1.0, "b": 0.0}) == 0.0


def test_value_error_no_encode():
    class BadModel:
        def forward(self, x, training=True):
            return x

        def backward(self, g):
            return g

        def params(self):
            return []

    with pytest.raises(ValueError, match="encode and decode"):
        OnlineAutoencoder(BadModel(), SGD(0.1))


def test_read_returns_none_when_x_not_dict():
    """Line 37: _read with x not a dict."""
    from dense_armor.utility.neural.online_autoencoder import _read

    v, k = _read(None, ["a"])
    assert v is None
    v, k = _read("not-a-dict", None)
    assert v is None


def test_first_x_not_dict_counts_missing():
    """Lines 151-154: first learn with x not a dict -> n_missing."""
    oa = OnlineAutoencoder(_hand(), SGD(0.1))
    oa.learn_one(None)
    assert oa.n_missing_ == 1
    assert oa.n_seen_ == 0
    oa.learn_one({"a": 1.0, "b": 0.0})
    assert oa.n_seen_ == 1


def test_get_params_wrap():
    """Line 163: get_params wraps model/optimizer/schedule."""
    from dense_armor.utility.neural.online import _EqWrap

    oa = OnlineAutoencoder(_hand(), SGD(0.1))
    d = oa.get_params()
    assert isinstance(d["model"], _EqWrap)
    assert isinstance(d["optimizer"], _EqWrap)
    assert isinstance(d["schedule"], _EqWrap)


def test_score_one_after_first_learn_with_missing():
    """Line 202: score_one with keys_ set but x has a missing feature."""
    oa = OnlineAutoencoder(_hand(), SGD(0.1), feature_keys=["a", "b"])
    oa.learn_one({"a": 1.0, "b": 0.0})
    assert oa.score_one({"a": 1.0}) == 0.0


def test_score_one_without_keys_returns_zero():
    """Line 202: score_one with feature_keys=None and no learn_one -> 0.0."""
    oa = OnlineAutoencoder(_hand(), SGD(0.1))
    assert oa.score_one({"a": 1.0, "b": 0.0}) == 0.0


def test_encode_one_with_missing_feature_returns_empty():
    """Line 202: encode_one with keys set but x incomplete."""
    oa = OnlineAutoencoder(_hand(), SGD(0.1), feature_keys=["a", "b"])
    oa.learn_one({"a": 1.0, "b": 0.0})
    assert oa.encode_one({"a": 1.0}) == {}
