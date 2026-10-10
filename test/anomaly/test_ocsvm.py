"""Tests for OneClassSGD (SONAR)."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.ocsvm import OneClassSGD


def _learn_normal(ds, n=800, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        ds.learn_one(
            {
                "a": float(rng.normal(0.5, 0.05)),
                "b": float(rng.normal(0.5, 0.05)),
            }
        )
    return ds


def test_ocsvm_score_float():
    ds = OneClassSGD(n_features_rff=64, nu=0.05, n_init=50, seed=0)
    _learn_normal(ds, 500)
    s = ds.score_one({"a": 0.5, "b": 0.5})
    assert isinstance(s, float)


def test_ocsvm_outlier_scores_higher():
    ds = OneClassSGD(n_features_rff=64, nu=0.05, n_init=50, seed=0)
    _learn_normal(ds, 800)
    s_in = ds.score_one({"a": 0.5, "b": 0.5})
    s_out = ds.score_one({"a": 5.0, "b": 5.0})
    assert s_out > s_in


def test_ocsvm_nan_counted_missing():
    ds = OneClassSGD(n_features_rff=32, nu=0.05, n_init=20, seed=0)
    ds.learn_one({"a": float("nan"), "b": 0.5})
    assert ds.n_missing == 1


def test_ocsvm_rejects_bad_params():
    with pytest.raises(ValueError):
        OneClassSGD(n_features_rff=0)
    with pytest.raises(ValueError):
        OneClassSGD(nu=0.0)
    with pytest.raises(ValueError):
        OneClassSGD(nu=1.0)
    with pytest.raises(ValueError):
        OneClassSGD(step=0.0)
    with pytest.raises(ValueError):
        OneClassSGD(gamma=0.0)
    with pytest.raises(ValueError):
        OneClassSGD(n_init=0)


def test_ocsvm_before_ready_returns_zero():
    ds = OneClassSGD(n_features_rff=32, nu=0.05, n_init=50, seed=0)
    for _ in range(10):
        ds.learn_one({"a": 0.5, "b": 0.5})
    assert ds.score_one({"a": 0.5, "b": 0.5}) == 0.0


def test_ocsvm_rho_tracks_nu():
    ds = OneClassSGD(n_features_rff=32, nu=0.1, n_init=50, seed=0)
    _learn_normal(ds, 2000)
    assert ds._ready
    assert -5.0 <= ds._rho <= 5.0


def test_ocsvm_adaptive_threshold_flags_nothing_at_warmup():
    ds = OneClassSGD(
        n_features_rff=32,
        nu=0.05,
        n_init=20,
        threshold_warmup=200,
        seed=0,
    )
    _learn_normal(ds, 50)
    assert ds.threshold == 1e9


def test_ocsvm_check_estimator():
    check_estimator(OneClassSGD(n_features_rff=32, nu=0.05, n_init=20, seed=0))
