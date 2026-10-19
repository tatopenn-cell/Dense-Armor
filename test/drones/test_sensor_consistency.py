"""Tests for pipelines/drones/sensor_consistency.py (task D1, item B2)."""

import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from pipelines.drones.io import synthetic_flight
from pipelines.drones.sensor_consistency import (
    GRAVITY_NED,
    SensorConsistencyResult,
    _episode_starts,
    run,
)

# ------------------------------------------------------------------
# _episode_starts (same helper as B1, tested for B2 too)
# ------------------------------------------------------------------

def test_episode_starts_empty():
    a = np.array([], dtype=bool)
    assert _episode_starts(a).size == 0


def test_episode_starts_single():
    a = np.array([False, True, True, False])
    np.testing.assert_array_equal(
        _episode_starts(a), [False, True, False, False])


def test_episode_starts_multiple():
    a = np.array([False, True, True, False, True, False, True])
    np.testing.assert_array_equal(
        _episode_starts(a),
        [False, True, False, False, True, False, True])


def test_episode_starts_first_true():
    a = np.array([True, True, False])
    np.testing.assert_array_equal(
        _episode_starts(a), [True, False, False])


# ------------------------------------------------------------------
# run: contract
# ------------------------------------------------------------------

def test_run_no_fault_no_episodes():
    stream = list(synthetic_flight(n=400, seed=0))
    r = run(stream)
    assert isinstance(r, SensorConsistencyResult)
    assert r.n_samples == 400
    assert r.fault_kind is None
    assert r.fault_index == -1
    assert r.delays_s == []
    assert r.max_delay_s is None
    assert r.mean_delay_s is None
    assert r.false_alarms == 0
    assert np.all(np.isfinite(r.score))


def test_run_gps_ramp_detects_multiple_episodes():
    stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
    r = run(stream)
    assert r.fault_kind == "gps_ramp"
    assert r.fault_index > 0
    assert r.n_episodes >= 2
    assert len(r.delays_s) == r.n_episodes
    assert all(d > 0 for d in r.delays_s)
    assert r.max_delay_s is not None
    assert r.mean_delay_s is not None
    assert r.false_alarms == 0


def test_run_delays_increasing():
    stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
    r = run(stream)
    if len(r.delays_s) >= 2:
        assert r.delays_s[0] < r.delays_s[1]


def test_run_empty_stream():
    r = run([])
    assert r.n_samples == 0
    assert r.n_episodes == 0
    assert np.isnan(r.false_alarms_per_hour)


def test_run_samples_missing_acc_are_skipped():
    stream = [
        {"t": 0.0, "acc": None, "gps_v": np.zeros(3)},
        {"t": 0.01, "acc": None, "gps_v": np.zeros(3)},
    ]
    r = run(stream)
    assert r.n_samples == 2
    assert np.all(np.isnan(r.score))


def test_run_score_grows_with_ramp():
    stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
    r = run(stream)
    if r.fault_index >= 0:
        s_at_fault = r.score[r.fault_index]
        s_end = r.score[-1]
        assert s_end > s_at_fault


def test_run_normal_seconds_correct():
    stream = list(synthetic_flight(n=400, seed=0, fault="gps_ramp"))
    r = run(stream)
    if r.fault_index > 0:
        expected = r.n_samples and (r.fault_index - 1) * 0.01
        assert abs(r.normal_seconds - expected) < 0.02


def test_run_custom_cusum_kwargs():
    stream = list(synthetic_flight(n=600, seed=0, fault="gps_ramp"))
    r = run(stream, cusum_kwargs={"h": 40.0})
    assert isinstance(r, SensorConsistencyResult)
    assert r.n_episodes <= 5


def test_run_residual_shape():
    stream = list(synthetic_flight(n=300, seed=0, fault="gps_ramp"))
    r = run(stream)
    assert r.residual.shape == (300, 3)
    assert r.score.shape == (300,)
    assert r.alarms.shape == (300,)


# ------------------------------------------------------------------
# gravity constant matches the one of synthetic_flight
# ------------------------------------------------------------------

def test_gravity_constant():
    assert np.allclose(GRAVITY_NED, [0.0, 0.0, 9.81])


# ------------------------------------------------------------------
# check_estimator on the drift detector
# ------------------------------------------------------------------

def test_cusum_passes_check_estimator():
    det = CUSUMDriftDetector(reset_on_alarm=True, reference="fixed")
    out = check_estimator(det)
    assert out["not_applicable"] == []
