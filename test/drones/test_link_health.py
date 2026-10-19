"""Tests for pipelines/drones/link_health.py (task D1, item B7)."""

import numpy as np
import pytest

from dense_armor.utility.drift.detector import CUSUMDriftDetector
from pipelines.drones.link_health import (
    DEFAULT_CUSUM,
    FAULT_KINDS,
    LinkHealthResult,
    _episode_starts,
    run,
    synthetic_mqtt,
)


def _with_fault(packets, kind):
    fa = round(0.6 * len(packets))
    for i, p in enumerate(packets):
        if i >= fa:
            p["fault_started"] = True
            p["fault_kind"] = kind
    return packets


# ------------------------------------------------------------------
# synthetic_mqtt
# ------------------------------------------------------------------

def test_synthetic_mqtt_keys():
    s = synthetic_mqtt(n=5, seed=0)
    assert len(s) == 5
    for p in s:
        assert set(p.keys()) == {
            "t", "inter_arrival_s", "rssi_dbm", "payload_bytes",
            "fault_started", "fault_kind",
        }
        assert p["inter_arrival_s"] > 0
        assert p["t"] > 0


def test_synthetic_mqtt_nominal_values():
    s = synthetic_mqtt(n=500, seed=0)
    ia = np.array([p["inter_arrival_s"] for p in s])
    rssi = np.array([p["rssi_dbm"] for p in s])
    assert abs(ia.mean() - 0.1) < 0.01
    assert abs(rssi.mean() + 60.0) < 1.0


def test_synthetic_mqtt_zero():
    assert synthetic_mqtt(n=0, seed=0) == []


def test_synthetic_mqtt_negative_raises():
    with pytest.raises(ValueError):
        synthetic_mqtt(n=-1, seed=0)


def test_synthetic_mqtt_invalid_fault():
    with pytest.raises(ValueError):
        synthetic_mqtt(n=5, seed=0, fault="unknown")


def test_fault_kinds_constant():
    assert FAULT_KINDS == (None, "link_degradation", "link_dropout")


# ------------------------------------------------------------------
# _episode_starts
# ------------------------------------------------------------------

def test_episode_starts_empty():
    a = np.array([], dtype=bool)
    assert _episode_starts(a).size == 0


def test_episode_starts_single():
    a = np.array([False, True, True, False])
    np.testing.assert_array_equal(
        _episode_starts(a), [False, True, False, False])


def test_episode_starts_multiple():
    a = np.array([False, True, False, True, True, False])
    np.testing.assert_array_equal(
        _episode_starts(a), [False, True, False, True, False, False])


# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------

def test_run_no_fault_no_episodes():
    packets = _with_fault(synthetic_mqtt(n=1000, seed=0), None)
    r = run(packets)
    assert isinstance(r, LinkHealthResult)
    assert r.fault_kind is None
    assert r.n_episodes == 0
    assert r.false_alarms == 0


def test_run_dropout_detected():
    packets = _with_fault(synthetic_mqtt(n=1000, seed=0, fault="link_dropout"),
                          "link_dropout")
    r = run(packets)
    assert r.fault_kind == "link_dropout"
    assert r.n_episodes == 1
    assert r.max_delay_s is not None
    assert r.max_delay_s < 1.0
    assert r.false_alarms == 0


def test_run_degradation_detected():
    packets = _with_fault(synthetic_mqtt(n=1000, seed=0, fault="link_degradation"),
                          "link_degradation")
    r = run(packets)
    assert r.fault_kind == "link_degradation"
    assert r.n_episodes >= 1
    assert r.delays_s[0] < 10.0
    assert r.false_alarms == 0


def test_run_empty_packets():
    r = run([])
    assert r.n_packets == 0
    assert r.n_episodes == 0


def test_run_deterministic():
    packets = _with_fault(synthetic_mqtt(n=300, seed=0, fault="link_dropout"),
                          "link_dropout")
    r1 = run(packets)
    r2 = run(packets)
    np.testing.assert_array_equal(r1.alarm, r2.alarm)
    assert r1.n_episodes == r2.n_episodes


def test_cusum_passes_check_estimator():
    from dense_armor.checks import check_estimator
    det = CUSUMDriftDetector(**DEFAULT_CUSUM)
    out = check_estimator(det)
    assert out["not_applicable"] == []
