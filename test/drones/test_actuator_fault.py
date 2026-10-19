"""Tests for pipelines/drones/actuator_fault.py (task D1, item B1)."""

import numpy as np

from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from pipelines.drones.actuator_fault import (
    ActuatorFaultResult,
    _episode_starts,
    _residual,
    run,
)
from pipelines.drones.io import synthetic_flight

# ------------------------------------------------------------------
# _residual
# ------------------------------------------------------------------

def test_residual_returns_difference():
    r = _residual({
        "commanded": np.array([0.1, 0.2, 0.3]),
        "gyro": np.array([0.05, 0.15, 0.25]),
    })
    np.testing.assert_allclose(r, [0.05, 0.05, 0.05], atol=1e-12)


def test_residual_missing_commanded_returns_none():
    assert _residual({"gyro": np.array([0.1, 0.2, 0.3])}) is None


def test_residual_missing_gyro_returns_none():
    assert _residual({"commanded": np.array([0.1, 0.2, 0.3])}) is None


# ------------------------------------------------------------------
# _episode_starts
# ------------------------------------------------------------------

def test_episode_starts_empty():
    a = np.array([], dtype=bool)
    assert _episode_starts(a).size == 0


def test_episode_starts_all_false():
    a = np.zeros(5, dtype=bool)
    np.testing.assert_array_equal(_episode_starts(a), a)


def test_episode_starts_single_true():
    a = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(
        _episode_starts(a),
        [False, False, True, False, False],
    )


def test_episode_starts_two_episodes():
    a = np.array([False, True, True, False, True, True, False])
    np.testing.assert_array_equal(
        _episode_starts(a),
        [False, True, False, False, True, False, False],
    )


def test_episode_starts_start_at_zero():
    a = np.array([True, True, False, True])
    np.testing.assert_array_equal(
        _episode_starts(a),
        [True, False, False, True],
    )


# ------------------------------------------------------------------
# run: contract
# ------------------------------------------------------------------

def test_run_no_fault_no_episodes():
    stream = list(synthetic_flight(n=300, seed=0))
    r = run(stream)
    assert isinstance(r, ActuatorFaultResult)
    assert r.n_samples == 300
    assert r.fault_kind is None
    assert r.fault_index == -1
    assert r.delays_s == []
    assert r.max_delay_s is None
    assert r.mean_delay_s is None
    assert r.normal_seconds > 0


def test_run_with_actuator_fault_detects():
    stream = list(synthetic_flight(n=600, seed=1, fault="actuator_loss"))
    r = run(stream)
    assert r.fault_kind == "actuator_loss"
    assert r.fault_index > 0
    assert r.n_episodes >= 1
    assert len(r.delays_s) >= 1
    assert r.max_delay_s is not None
    assert r.max_delay_s >= 0.0
    assert r.max_delay_s <= 0.5
    assert r.mean_delay_s is not None
    assert 0.0 <= r.mean_delay_s <= 0.5
    assert r.false_alarms == 0
    assert r.false_alarms_per_hour < 1.0


def test_run_empty_stream():
    r = run([])
    assert r.n_samples == 0
    assert r.n_episodes == 0
    assert r.delays_s == []
    assert np.isnan(r.false_alarms_per_hour)


def test_run_samples_missing_commanded_are_skipped():
    stream = [
        {"t": 0.0, "gyro": np.zeros(3), "commanded": None},
        {"t": 0.01, "gyro": np.zeros(3), "commanded": None},
    ]
    r = run(stream)
    assert r.n_samples == 2
    assert np.all(np.isnan(r.mahalanobis_scores))


def test_run_custom_cusum_kwargs():
    stream = list(synthetic_flight(n=300, seed=0, fault="actuator_loss"))
    r = run(stream, cusum_kwargs={"h": 50.0})
    assert isinstance(r, ActuatorFaultResult)
    assert r.n_episodes <= 3


def test_run_custom_maha_kwargs():
    stream = list(synthetic_flight(n=300, seed=0, fault="actuator_loss"))
    r = run(stream, maha_kwargs={"threshold": 3.0})
    assert isinstance(r, ActuatorFaultResult)


def test_run_false_alarms_per_hour_computed_on_normal_part():
    stream = list(synthetic_flight(n=300, seed=5, fault="actuator_loss"))
    r = run(stream)
    assert r.normal_seconds > 0
    expected = r.false_alarms * 3600.0 / r.normal_seconds
    assert abs(r.false_alarms_per_hour - expected) < 1e-9


def test_run_no_normal_part_gives_nan():
    stream = [
        {"t": 0.0, "fault_started": True, "fault_kind": "actuator_loss",
         "gyro": np.zeros(3), "commanded": np.zeros(3)},
    ]
    r = run(stream)
    assert r.fault_index == 0
    assert np.isnan(r.false_alarms_per_hour)


# ------------------------------------------------------------------
# check_estimator on the estimators used
# ------------------------------------------------------------------

def test_cusum_passes_check_estimator():
    from dense_armor.checks import check_estimator
    det = CUSUMDriftDetector(reset_on_alarm=True, reference="fixed")
    out = check_estimator(det)
    assert out["not_applicable"] == []


def test_mahalanobis_passes_check_estimator():
    from dense_armor.checks import check_estimator
    est = OnlineRobustMahalanobis(
        feature_keys=["r0", "r1", "r2"], n_init=50,
    )
    out = check_estimator(est)
    assert out["not_applicable"] == []
