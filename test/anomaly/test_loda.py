"""Tests for LODA."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.loda import LODA


def _learn_normal(ds, n=400, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        ds.learn_one(
            {
                "a": float(rng.normal(0.5, 0.05)),
                "b": float(rng.normal(0.5, 0.05)),
            }
        )
    return ds


def test_loda_score_float():
    ds = LODA(n_projections=30, n_bins=8, window=200, range_init=10, seed=0)
    _learn_normal(ds, 300)
    s = ds.score_one({"a": 0.5, "b": 0.5})
    assert isinstance(s, float)


def test_loda_outlier_scores_higher():
    ds = LODA(n_projections=100, n_bins=10, window=400, range_init=10, seed=0)
    _learn_normal(ds, 600)
    s_in = ds.score_one({"a": 0.5, "b": 0.5})
    s_out = ds.score_one({"a": 5.0, "b": 5.0})
    assert s_out > s_in


def test_loda_nan_counted_missing():
    ds = LODA(n_projections=10, n_bins=5, window=50, range_init=5, seed=0)
    ds.learn_one({"a": float("nan"), "b": 0.5})
    assert ds.n_missing == 1


def test_loda_rejects_bad_params():
    with pytest.raises(ValueError):
        LODA(n_projections=0)
    with pytest.raises(ValueError):
        LODA(n_bins=1)
    with pytest.raises(ValueError):
        LODA(window=0)
    with pytest.raises(ValueError):
        LODA(sparsity=0)
    with pytest.raises(ValueError):
        LODA(range_init=0)


def test_loda_projection_orthonormal():
    ds = LODA(
        n_projections=5,
        n_bins=4,
        window=32,
        sparsity=2,
        range_init=5,
        seed=0,
    )
    _learn_normal(ds, 10)
    assert ds._W is not None
    for j in range(5):
        nrm = float(np.linalg.norm(ds._W[j]))
        assert abs(nrm - 1.0) < 1e-9


def test_loda_explicit_ranges():
    ds = LODA(
        n_projections=10,
        n_bins=5,
        window=20,
        feature_ranges=[(-10.0, 10.0), (-10.0, 10.0)],
        seed=0,
    )
    ds.learn_one({"a": 5.0, "b": -5.0})
    assert ds._ready


def test_loda_online_ranges_learned():
    ds = LODA(n_projections=10, n_bins=5, window=30, range_init=10, seed=0)
    for i in range(30):
        ds.learn_one({"a": float(i), "b": float(-i)})
    assert ds._ready
    assert ds._lo is not None and ds._hi is not None
    assert ds._lo[0] < 0.0
    assert ds._hi[0] > 5.0


def test_loda_buffer_bounded():
    ds = LODA(n_projections=10, n_bins=5, window=16, range_init=5, seed=0)
    _learn_normal(ds, 200)
    assert ds._buf_size == 16


def test_loda_before_ready_returns_zero():
    ds = LODA(n_projections=10, n_bins=5, window=16, range_init=20, seed=0)
    assert ds.score_one({"a": 0.5, "b": 0.5}) == 0.0


def test_loda_check_estimator():
    check_estimator(
        LODA(
            n_projections=20,
            n_bins=8,
            window=64,
            range_init=10,
            seed=0,
        )
    )
