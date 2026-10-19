"""Tests for pipelines/drones/propeller_health.py (task D1, item B3)."""

import numpy as np
import pytest

from pipelines.drones.io import synthetic_flight
from pipelines.drones.propeller_health import (
    DEFAULT_BAND_HZ,
    DEFAULT_THRESHOLD,
    PropellerHealthResult,
    run,
)


def test_no_fault_no_episodes():
    stream = list(synthetic_flight(n=600, seed=0))
    r = run(stream)
    assert isinstance(r, PropellerHealthResult)
    assert r.n_samples == 600
    assert r.fault_kind is None
    assert r.fault_index == -1
    assert r.n_episodes == 0
    assert r.delays_s == []
    assert r.false_alarms == 0
    assert r.false_alarms_per_hour == 0.0


def test_propeller_fault_detected():
    stream = list(synthetic_flight(n=2000, seed=0, fault="propeller"))
    r = run(stream)
    assert r.fault_kind == "propeller"
    assert r.fault_index > 0
    assert r.n_episodes == 1
    assert len(r.delays_s) == 1
    assert r.max_delay_s is not None
    assert r.max_delay_s < 1.0
    assert r.false_alarms == 0
    assert r.false_alarms_per_hour == 0.0


def test_deterministic():
    stream = list(synthetic_flight(n=2000, seed=0, fault="propeller"))
    r1 = run(stream)
    r2 = run(stream)
    assert r1.n_episodes == r2.n_episodes
    np.testing.assert_array_equal(r1.score, r2.score)
    np.testing.assert_array_equal(r1.alarms, r2.alarms)


def test_empty_stream():
    r = run([])
    assert r.n_samples == 0
    assert r.n_episodes == 0
    assert np.isnan(r.false_alarms_per_hour)


def test_missing_gyro_skipped():
    stream = [
        {"t": 0.0, "gyro": None},
        {"t": 0.01, "gyro": None},
    ]
    r = run(stream)
    assert r.n_samples == 2
    assert np.all(np.isnan(r.score))


def test_invalid_band_raises():
    stream = list(synthetic_flight(n=100, seed=0))
    with pytest.raises(ValueError, match="band_hz"):
        run(stream, band_hz=(200.0, 250.0))


def test_custom_threshold():
    stream = list(synthetic_flight(n=2000, seed=0, fault="propeller"))
    r = run(stream, threshold=50.0)
    assert r.n_episodes <= 1
    r_low = run(stream, threshold=1.0)
    assert r_low.n_episodes >= r.n_episodes


def test_defaults_are_documented_constants():
    assert DEFAULT_THRESHOLD == 10.0
    assert DEFAULT_BAND_HZ == (40.0, 55.0)


def test_score_shape_and_finiteness():
    stream = list(synthetic_flight(n=300, seed=0, fault="propeller"))
    r = run(stream)
    assert r.score.shape == (300,)
    assert r.alarms.shape == (300,)
    valid = r.score[~np.isnan(r.score)]
    assert np.all(np.isfinite(valid))


def test_sdft_used():
    from dense_armor.utility.misc import SDFT as S
    s = S(n=8)
    assert hasattr(s, "spectrum")
    assert hasattr(s, "is_ready")
