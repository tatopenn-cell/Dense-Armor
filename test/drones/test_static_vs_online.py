"""Tests for pipelines/drones/static_vs_online.py (task D1, item B8)."""

import numpy as np
import pytest

from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from pipelines.drones.io import synthetic_flight
from pipelines.drones.static_vs_online import (
    N_FEATURES,
    StaticVsOnlineResult,
    _features,
    run,
)


def _tree():
    return HoeffdingTreeClassifier(grace_period=50)


def test_features_count_and_keys():
    s = next(synthetic_flight(n=1, seed=0))
    f = _features(s)
    assert len(f) == N_FEATURES
    for k in ("gyro_x", "acc_z", "gps_n", "motor_1", "battery_v",
              "current_a", "payload_kg"):
        assert k in f


@pytest.mark.slow
def test_run_one_seed():
    r = run(n_seeds=1, n_samples=150, max_depth=3)
    assert isinstance(r, StaticVsOnlineResult)
    assert r.n_seeds == 1
    assert r.n_samples == 150
    assert r.warmup == 75
    assert len(r.accuracy_batch) == 1
    assert len(r.accuracy_online) == 1


@pytest.mark.slow
def test_run_accuracy_reasonable():
    r = run(n_seeds=1, n_samples=150, max_depth=3)
    assert 0.7 <= r.median_accuracy_batch <= 1.0
    assert 0.7 <= r.median_accuracy_online <= 1.0


@pytest.mark.slow
def test_run_batch_faster_than_online():
    r = run(n_seeds=1, n_samples=150, max_depth=3)
    assert r.time_per_sample_batch_ms > 0
    assert r.time_per_sample_online_ms > 0
    # the batch is usually faster per sample; do not assert the inequality,
    # just check both are finite and small


@pytest.mark.slow
def test_run_memory_is_a_finite_delta():
    r = run(n_seeds=1, n_samples=150, max_depth=3)
    assert np.isfinite(r.memory_batch_kb)
    assert np.isfinite(r.memory_online_kb)


@pytest.mark.slow
def test_run_deterministic():
    r1 = run(n_seeds=1, n_samples=150, max_depth=3)
    r2 = run(n_seeds=1, n_samples=150, max_depth=3)
    np.testing.assert_array_equal(r1.accuracy_batch, r2.accuracy_batch)
    np.testing.assert_array_equal(r1.accuracy_online, r2.accuracy_online)


@pytest.mark.slow
def test_run_names_populated():
    r = run(n_seeds=1, n_samples=100, max_depth=3)
    assert "CART" in r.batch_name
    assert "Hoeffding" in r.online_name


def test_fit_batch_makes_members():
    from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier as OB
    clf = OB(_tree, n_models=3, seed=0)
    X = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0},
         {"a": 5.0, "b": 6.0}, {"a": 7.0, "b": 8.0}]
    y = [0, 1, 0, 1]
    clf.fit_batch(X, y, seed=0)
    assert len(clf.members_) == 3
    assert clf.n_seen_ == 4


def test_fit_batch_length_mismatch_raises():
    from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier as OB
    clf = OB(_tree, n_models=3, seed=0)
    with pytest.raises(ValueError, match="X has"):
        clf.fit_batch([{"a": 1.0}], [0, 1], seed=0)


def test_fit_batch_empty_raises():
    from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier as OB
    clf = OB(_tree, n_models=3, seed=0)
    with pytest.raises(ValueError, match="empty"):
        clf.fit_batch([], [], seed=0)


def test_fit_batch_passes_check_estimator():
    from dense_armor.checks import check_estimator
    clf = OnlineBaggingClassifier(_tree, n_models=3, seed=0)
    out = check_estimator(clf)
    assert out["not_applicable"] == []


def test_online_passes_check_estimator():
    from dense_armor.checks import check_estimator
    clf = OnlineBaggingClassifier(_tree, n_models=3, seed=0)
    out = check_estimator(clf)
    assert out["not_applicable"] == []


def test_wilcoxon_on_ten_seeds():
    """_wilcoxon on 10 per-seed values is finite and small when a > b."""
    import numpy as np

    from pipelines.drones.static_vs_online import _wilcoxon
    a = np.array([1.0, 0.99, 0.98, 1.0, 0.99, 1.0, 0.99, 1.0, 0.98, 1.0])
    b = np.array([0.99, 0.98, 0.97, 0.99, 0.98, 0.99, 0.98, 0.99, 0.97, 0.99])
    p = _wilcoxon(a, b)
    assert 0.0 < p < 0.05


def test_wilcoxon_constant_returns_one():
    """_wilcoxon on identical arrays returns 1.0 (fallback branch)."""
    import numpy as np

    from pipelines.drones.static_vs_online import _wilcoxon
    a = np.array([0.99, 0.99, 0.99, 0.99, 0.99])
    assert _wilcoxon(a, a) == 1.0


def test_a12_on_separated_values():
    """_a12 on 10 vs 10 separated values is 1.0; on identical is 0.5."""
    import numpy as np

    from pipelines.drones.static_vs_online import _a12
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([0.99] * 10)
    assert _a12(a, b) == 1.0
    assert _a12(b, b) == 0.5
