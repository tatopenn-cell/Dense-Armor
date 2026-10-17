"""Tests for the field-weighted FM (Pan et al. 2018, eq. (7))."""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.facto import FwFMClassifier, FwFMRegressor
from dense_armor.utility.facto import fwfm as fwfm_module
from dense_armor.utility.optim import SGD

FIELDS = {"a": "F1", "b": "F1", "c": "F2", "d": "F2"}


def _factory_clf():
    return FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, k=2, seed=0)


def _factory_reg():
    return FwFMRegressor(optimizer=SGD(lr=0.1), fields=FIELDS, k=2, seed=0)


def test_model_equals_direct_sum():
    """Eq. (7) page 3: y_hat = w0 + sum w_i x_i + sum pair term."""
    rng = np.random.default_rng(0)
    m = FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, k=3, seed=0)
    m._resolve_keys({"a": 0.0, "c": 0.0})
    m._ensure_state()
    for _ in range(200):
        x = {k: float(rng.normal()) for k in FIELDS if rng.random() < 0.5}
        if len(x) < 2:
            continue
        fast = m._y_hat(x)
        w0, w, V, R = m._unpack()
        active = m._active(x)
        slow = w0
        for a, (i, fi, xi) in enumerate(active):
            slow += w[i] * xi
            for j, fj, xj in active[a + 1 :]:
                lo, hi = (fi, fj) if fi <= fj else (fj, fi)
                slow += float(np.dot(V[i], V[j])) * xi * xj * R[lo, hi]
        assert abs(fast - slow) < 1e-12


def test_R_starts_at_one():
    """My choice, documented: r = 1 so the model starts as a plain FM."""
    m = FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, k=2, seed=0)
    m._resolve_keys({"a": 0.0, "c": 0.0})
    m._ensure_state()
    _, _, _, R = m._unpack()
    assert np.all(R == 1.0)


def test_R_is_symmetric_in_reading():
    """Only the upper triangle is used: F1,F2 equals F2,F1."""
    m = FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, k=2, seed=0)
    m._resolve_keys({"a": 0.0, "c": 0.0})
    m._ensure_state()
    _, _, _, R = m._unpack()
    R[0, 1] = 0.3
    R[1, 0] = 0.9
    x = {"a": 1.0, "c": 1.0}
    score = m._y_hat(x)
    _, _, V, _ = m._unpack()
    i_a = m._key_index["a"]
    i_c = m._key_index["c"]
    expected = float(np.dot(V[i_a], V[i_c])) * 1.0 * 1.0 * 0.3
    assert abs(score - expected) < 1e-10


def test_learns_a_constant_target():
    m = _factory_reg()
    for _ in range(300):
        _ = m.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 2.0)
    assert abs(m.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) - 2.0) < 0.5


def test_predict_before_learning():
    m = _factory_reg()
    assert m.predict_one({"a": 1.0}) == 0.0
    c = _factory_clf()
    assert c.predict_proba_one({"a": 1.0}) == {}
    assert c.predict_one({"a": 1.0}) is None


def test_nan_skipped_and_counted():
    m = _factory_reg()
    _ = m.learn_one({"a": 1.0}, 1.0)
    _ = m.learn_one({"a": float("nan")}, 1.0)
    _ = m.learn_one({"a": 1.0}, float("nan"))
    assert m.n_missing == 2
    assert m.n_seen_ == 1


def test_missing_key_skipped_and_counted():
    m = _factory_reg()
    _ = m.learn_one({"a": 1.0}, 1.0)
    _ = m.learn_one({}, 1.0)
    assert m.n_missing == 1


def test_constructor_arguments_not_overwritten():
    m = FwFMClassifier(optimizer=SGD(lr=0.5), fields=FIELDS, k=4, l2=0.01, seed=7)
    for _ in range(10):
        _ = m.learn_one({"a": 1.0}, 0)
    assert m.k == 4
    assert m.l2 == 0.01
    assert m.seed == 7


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, k=0)
    with pytest.raises(ValueError):
        FwFMClassifier(optimizer=SGD(lr=0.1), fields=FIELDS, l2=-1.0)


def test_check_estimator_classifier():
    r = check_estimator(_factory_clf())
    assert r["not_applicable"] == []


def test_check_estimator_regressor():
    r = check_estimator(_factory_reg())
    assert r["not_applicable"] == []


def test_doctest():
    res = doctest.testmod(fwfm_module, verbose=False)
    assert res.failed == 0
