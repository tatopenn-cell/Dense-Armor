"""Tests for the factorization machine (Rendle 2010, eq. (1)-(4))."""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.facto import FMClassifier, FMRegressor
from dense_armor.utility.facto import fm as fm_module
from dense_armor.utility.optim import SGD


def _factory_reg():
    return FMRegressor(optimizer=SGD(lr=0.05), k=2, seed=0)


def _factory_clf():
    return FMClassifier(optimizer=SGD(lr=0.05), k=2, seed=0)


def test_ok_form_matches_double_sum():
    """Lemma 3.1, page 3: O(kn) equals the O(kn^2) double sum."""
    rng = np.random.default_rng(0)
    fm = FMRegressor(optimizer=SGD(lr=0.05), k=3, seed=1)
    keys = [f"x{i}" for i in range(8)]
    fm._resolve_keys({"x0": 0.0})
    fm._ensure_state(fm._keys)
    for _ in range(200):
        x = {k: float(rng.normal()) for k in keys if rng.random() < 0.4}
        fast = fm._y_hat(x)
        slow = fm._pairwise_naive(x)
        assert abs(fast - slow) < 1e-9, f"x={x}  fast={fast}  slow={slow}"


def test_learns_a_constant_target():
    fm = _factory_reg()
    for _ in range(200):
        _ = fm.learn_one({"a": 1.0, "b": 0.5}, 2.0)
    assert abs(fm.predict_one({"a": 1.0, "b": 0.5}) - 2.0) < 0.3


def test_predict_before_learning_returns_zero():
    fm = FMRegressor(optimizer=SGD(lr=0.1), k=2)
    assert fm.predict_one({"a": 1.0}) == 0.0
    fc = FMClassifier(optimizer=SGD(lr=0.1), k=2)
    assert fc.predict_proba_one({"a": 1.0}) == {}
    assert fc.predict_one({"a": 1.0}) is None


def test_nan_skipped_and_counted():
    fm = _factory_reg()
    _ = fm.learn_one({"a": 1.0}, 1.0)
    _ = fm.learn_one({"a": float("nan")}, 1.0)
    _ = fm.learn_one({"a": float("inf")}, 1.0)
    _ = fm.learn_one({"a": 1.0}, float("nan"))
    assert fm.n_missing == 3
    assert fm.n_seen_ == 1


def test_missing_key_skipped_and_counted():
    fm = _factory_reg()
    _ = fm.learn_one({"a": 1.0}, 1.0)
    _ = fm.learn_one({}, 1.0)
    assert fm.n_missing == 1


def test_constructor_arguments_not_overwritten():
    fm = FMRegressor(optimizer=SGD(lr=0.1), k=4, l2=0.01, seed=7)
    for _ in range(10):
        _ = fm.learn_one({"a": 1.0}, 1.0)
    assert fm.k == 4
    assert fm.l2 == 0.01
    assert fm.seed == 7
    assert fm.get_params()["k"] == 4


def test_clone_is_independent():
    fm = _factory_reg()
    for _ in range(20):
        _ = fm.learn_one({"a": 1.0}, 1.0)
    c = fm.clone()
    assert c._theta is None


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        FMRegressor(optimizer=SGD(), k=0)
    with pytest.raises(ValueError):
        FMRegressor(optimizer=SGD(), l2=-1.0)


def test_check_estimator_regressor():
    r = check_estimator(_factory_reg())
    assert r["not_applicable"] == []


def test_check_estimator_classifier():
    r = check_estimator(_factory_clf())
    assert r["not_applicable"] == []


def test_doctest():
    res = doctest.testmod(fm_module, verbose=False)
    assert res.failed == 0
