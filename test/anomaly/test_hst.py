"""Tests for HalfSpaceTrees."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.hst import HalfSpaceTrees


def test_hst_shapes():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=30, range_init=5, seed=0)
    for _ in range(10):
        hst.learn_one({"a": 0.5, "b": 0.5})
    s = hst.score_one({"a": 0.5, "b": 0.5})
    assert isinstance(s, float)


def test_hst_outlier_scores_higher():
    hst = HalfSpaceTrees(n_trees=15, depth=8, window=100, range_init=10, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        hst.learn_one(
            {
                "a": float(rng.normal(0.0, 0.05)),
                "b": float(rng.normal(0.0, 0.05)),
            }
        )
    s_in = hst.score_one({"a": 0.0, "b": 0.0})
    s_out = hst.score_one({"a": 10.0, "b": 10.0})
    assert s_out > s_in


def test_hst_nan_counted_missing():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=20, range_init=5, seed=0)
    hst.learn_one({"a": float("nan"), "b": 0.5})
    assert hst.n_missing == 1


def test_hst_rejects_bad_params():
    with pytest.raises(ValueError):
        HalfSpaceTrees(n_trees=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(depth=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(window=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(range_init=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(threshold_quantile=0.0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(threshold_quantile=1.0)


def test_hst_explicit_ranges():
    hst = HalfSpaceTrees(
        n_trees=5,
        depth=6,
        window=20,
        feature_ranges=[(-10.0, 10.0), (-10.0, 10.0)],
        seed=0,
    )
    hst.learn_one({"a": 5.0, "b": -5.0})
    s = hst.score_one({"a": 5.0, "b": -5.0})
    assert isinstance(s, float)


def test_hst_online_ranges_learned():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=30, range_init=10, seed=0)
    for i in range(30):
        hst.learn_one({"a": float(i), "b": float(-i)})
    assert hst._ready
    assert hst._lo is not None and hst._hi is not None
    assert hst._lo[0] < 0.0
    assert hst._hi[0] > 5.0


def test_hst_adaptive_threshold_flags_nothing_at_warmup():
    hst = HalfSpaceTrees(
        n_trees=5,
        depth=6,
        window=30,
        range_init=5,
        threshold_warmup=50,
        seed=0,
    )
    for _ in range(10):
        hst.learn_one({"a": 0.5, "b": 0.5})
    assert hst.threshold == 1e9
    assert not hst.is_outlier({"a": 0.5, "b": 0.5})


def test_hst_window_bounds_memory():
    hst = HalfSpaceTrees(n_trees=3, depth=5, window=10, range_init=5, seed=0)
    for i in range(100):
        hst.learn_one({"a": float(i % 10) / 10.0})
    assert hst._buf_size == 10


def test_hst_check_estimator():
    check_estimator(HalfSpaceTrees(n_trees=3, depth=5, window=20, range_init=5, seed=0))
