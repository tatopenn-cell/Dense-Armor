"""Tests for pipelines/drones/safety_filter.py (task D1, item B5)."""

import numpy as np

import benchmarks.lidar_sim as ls
from pipelines.drones.safety_filter import (
    DEFAULT_SAFE_DIST_GEOFENCE,
    DEFAULT_SAFE_DIST_OBSTACLE,
    DEFAULT_V_MAX,
    SafetyFilterResult,
    run,
)


def _scans(n=200, seed=0, fault=None, onset=5000, start=0):
    rng = np.random.default_rng(seed)
    return [ls.scan(start + i, rng, fault, onset) for i in range(n)]


def test_clean_stream_low_speed_is_safe():
    r = run(_scans(fault=None), v_max=0.25)
    assert r.n_scans == 200
    assert r.n_modified_obs == 0
    assert r.n_modified_geo == 0
    assert r.n_modified_both == 0
    assert r.max_change < 1e-12


def test_clean_stream_full_speed_is_modified():
    r = run(_scans(fault=None), v_max=2.0)
    assert r.n_modified_both == 200
    assert r.max_change > 0.5


def test_dirty_stream_violated_even_at_low_speed():
    r = run(_scans(fault="dirt", start=20000), v_max=0.10)
    assert r.n_modified_both == 200
    assert r.max_change > 0.1


def test_v_safe_is_max_of_the_two():
    r = run(_scans(fault="dirt", start=20000, n=50), v_max=0.10)
    expected = np.maximum(r.u_obs, r.u_geo)
    np.testing.assert_array_equal(r.u_safe, expected)


def test_dataclass_fields():
    r = run(_scans(n=10))
    assert isinstance(r, SafetyFilterResult)
    assert r.safe_dist_obstacle == DEFAULT_SAFE_DIST_OBSTACLE
    assert r.safe_dist_geofence == DEFAULT_SAFE_DIST_GEOFENCE
    assert r.v_max == DEFAULT_V_MAX
    assert r.x_obs.shape == (10,)
    assert r.u_safe.shape == (10,)
    assert 0.0 <= r.fraction_modified <= 1.0


def test_min_distances_are_positive():
    r = run(_scans(n=50))
    assert r.min_x_obs > 0.0
    assert r.min_x_geo > 1.0


def test_deterministic():
    scans = _scans(n=50)
    r1 = run(scans)
    r2 = run(scans)
    np.testing.assert_array_equal(r1.u_safe, r2.u_safe)
    np.testing.assert_array_equal(r1.x_obs, r2.x_obs)


def test_empty_scans():
    r = run([])
    assert r.n_scans == 0
    assert np.isnan(r.fraction_modified)


def test_custom_safe_dist():
    r_small = run(_scans(fault="dirt", start=20000, n=50), safe_dist_obstacle=0.1)
    r_large = run(_scans(fault="dirt", start=20000, n=50), safe_dist_obstacle=2.0)
    assert r_large.max_change >= r_small.max_change


def test_u_des_is_negative_vmax():
    r = run(_scans(n=20), v_max=1.0)
    assert r.v_max == 1.0
    assert np.all(r.u_obs <= 1.0 + 1e-9)
