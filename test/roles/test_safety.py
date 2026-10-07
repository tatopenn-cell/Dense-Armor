"""Tests for dense_armor/roles/safety.py."""
import doctest

import pytest

import dense_armor.roles.safety as safety_mod
from dense_armor.roles import Regressor
from dense_armor.roles.safety import (
    CHECKPOINT_VERSION,
    Health,
    HealthMonitor,
    SafeEstimator,
)
from dense_armor.utility.stats.moments import RunningMoments


class _LastValue(Regressor):
    """The simplest regressor: predict the last y seen."""

    def __init__(self) -> None:
        self.m_: RunningMoments = RunningMoments()
        self._last = 0.0

    def learn_one(self, x, y, t=None):
        self.m_.learn_one(x, t=t)
        self._last = float(y)
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self._last


def test_health_monitor_warming_up():
    hm = HealthMonitor(warmup=10)
    for _ in range(5):
        assert hm.update(0.1) is Health.WARMING_UP


def test_health_monitor_nominal_on_stable_residuals():
    hm = HealthMonitor(window=20, warmup=5, drift_mult=2.0, degraded_mult=5.0)
    state = None
    for _ in range(100):
        state = hm.update(0.1)
    assert state is Health.NOMINAL


def test_health_monitor_drifting_on_growing_residuals():
    hm = HealthMonitor(window=20, warmup=5, drift_mult=2.0, degraded_mult=5.0)
    for _ in range(50):
        hm.update(0.1)
    state = None
    for _ in range(20):
        state = hm.update(0.4)
    assert state is Health.DRIFTING


def test_health_monitor_degraded_on_large_residuals():
    hm = HealthMonitor(window=20, warmup=5, drift_mult=2.0, degraded_mult=5.0)
    for _ in range(50):
        hm.update(0.1)
    state = None
    for _ in range(20):
        state = hm.update(2.0)
    assert state is Health.DEGRADED


def test_safe_estimator_blocks_flagged_samples():
    est = SafeEstimator(
        _LastValue(),
        guard=lambda s: s["x"] > 100.0,
        fallback=-1.0,
    )
    for i in range(20):
        est.learn_one({"x": float(i)}, float(i))
    blocked_before = est.n_blocked_
    est.learn_one({"x": 1e6}, 0.0)
    assert est.n_blocked_ == blocked_before + 1
    assert est.model.m_.count == 20


def test_safe_estimator_returns_fallback_when_flagged():
    est = SafeEstimator(
        _LastValue(),
        guard=lambda s: s["x"] > 100.0,
        fallback=42.0,
    )
    for i in range(20):
        est.learn_one({"x": float(i)}, float(i))
    assert est.predict_one({"x": 1e6}) == 42.0


def test_safe_estimator_passes_through_when_not_flagged():
    est = SafeEstimator(
        _LastValue(),
        guard=lambda s: s["x"] > 100.0,
        fallback=42.0,
    )
    for i in range(20):
        est.learn_one({"x": float(i)}, float(i))
    out = est.predict_one({"x": 1.0})
    assert out == pytest.approx(19.0)


def test_safe_estimator_is_flagged_direct():
    est = SafeEstimator(
        _LastValue(), guard=lambda s: s["x"] > 100.0, fallback=0.0
    )
    assert est.is_flagged({"x": 1e6}) is True
    assert est.is_flagged({"x": 1.0}) is False


def test_safe_estimator_checkpoint_roundtrip(tmp_path):
    est = SafeEstimator(_LastValue(), fallback=0.0)
    for i in range(50):
        est.learn_one({"x": float(i)}, float(i))
    snapshot_mean = est.model.m_.mean
    p = est.save(tmp_path / "cp.pkl")
    fresh = SafeEstimator(_LastValue(), fallback=0.0)
    fresh.restore(p)
    assert fresh.model.m_.mean == pytest.approx(snapshot_mean)
    assert fresh.model.m_.count == est.model.m_.count


def test_safe_estimator_rejects_corrupted_checkpoint(tmp_path):
    import pickle

    est = SafeEstimator(_LastValue(), fallback=0.0)
    for i in range(50):
        est.learn_one({"x": float(i)}, float(i))
    p = est.save(tmp_path / "cp.pkl")
    payload = pickle.loads(p.read_bytes())
    payload["sha256"] = "0" * 64
    p.write_bytes(pickle.dumps(payload))
    fresh = SafeEstimator(_LastValue(), fallback=0.0)
    with pytest.raises(ValueError, match="corrupted"):
        fresh.restore(p)


def test_safe_estimator_rejects_wrong_version(tmp_path):
    import hashlib
    import pickle

    p = tmp_path / "cp.pkl"
    state = {
        "version": CHECKPOINT_VERSION + 99,
        "model_module": "x",
        "model_class": "_LastValue",
        "state_dict": {},
    }
    blob = pickle.dumps(state)
    payload = {"sha256": hashlib.sha256(blob).hexdigest(), "blob": blob}
    p.write_bytes(pickle.dumps(payload))
    est = SafeEstimator(_LastValue(), fallback=0.0)
    with pytest.raises(ValueError, match="version"):
        est.restore(p)


def test_safe_estimator_handles_model_without_predict_one():
    est = SafeEstimator(RunningMoments(), fallback=7.0)
    for i in range(20):
        est.learn_one({"x": float(i)}, float(i))
    assert est.predict_one({"x": 1.0}) is None
    assert est.model.count == 20


def test_doctests():
    assert doctest.testmod(safety_mod).failed == 0
