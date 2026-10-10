"""Tests for OnlineIsolationForest."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest


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


def test_oif_score_in_range():
    ds = OnlineIsolationForest(n_trees=8, window=512, max_leaf_samples=8, seed=0)
    _learn_normal(ds, 600)
    s = ds.score_one({"a": 0.5, "b": 0.5})
    assert 0.0 <= s <= 1.0


def test_oif_outlier_scores_higher():
    ds = OnlineIsolationForest(n_trees=32, window=512, max_leaf_samples=8, seed=0)
    _learn_normal(ds, 800)
    s_in = ds.score_one({"a": 0.5, "b": 0.5})
    s_out = ds.score_one({"a": 50.0, "b": 50.0})
    assert s_out > s_in


def test_oif_nan_counted_missing():
    ds = OnlineIsolationForest(n_trees=4, window=64, max_leaf_samples=16, seed=0)
    ds.learn_one({"a": float("nan"), "b": 0.5})
    assert ds.n_missing == 1


def test_oif_rejects_bad_params():
    with pytest.raises(ValueError):
        OnlineIsolationForest(n_trees=0)
    with pytest.raises(ValueError):
        OnlineIsolationForest(window=1)
    with pytest.raises(ValueError):
        OnlineIsolationForest(max_leaf_samples=1)
    with pytest.raises(ValueError):
        OnlineIsolationForest(window=64, max_leaf_samples=128)


def test_oif_buffer_bounded():
    ds = OnlineIsolationForest(n_trees=4, window=32, max_leaf_samples=8, seed=0)
    _learn_normal(ds, 200)
    assert ds._buf is not None
    assert len(ds._buf) == 32


def test_oif_before_ready_returns_one():
    ds = OnlineIsolationForest(n_trees=4, window=32, max_leaf_samples=8, seed=0)
    assert ds.score_one({"a": 0.5, "b": 0.5}) == 1.0


def test_oif_check_estimator():
    check_estimator(
        OnlineIsolationForest(n_trees=4, window=64, max_leaf_samples=8, seed=0)
    )
