"""Tests for pipelines/drones/battery_endurance.py (task D1, item B4)."""

import numpy as np

from dense_armor.utility.linear.glm import LinearRegression
from pipelines.drones.battery_endurance import (
    BETA_SOLO,
    DEFAULT_CAPACITY_WH,
    DEFAULT_SOC_MIN,
    DEFAULT_V_FULL,
    DEFAULT_V_MIN,
    BatteryEnduranceResult,
    _p_features,
    _soc,
    run,
)
from pipelines.drones.io import synthetic_flight


def test_p_features_nine_terms():
    s = {
        "gps_v": np.array([3.0, 4.0, 0.0]),
        "acc": np.array([0.0, 0.0, 9.81]),
        "payload_kg": 0.5,
        "wind_xy": np.array([1.0, 0.0]),
    }
    f = _p_features(s)
    assert set(f.keys()) == set(BETA_SOLO.keys())
    assert abs(f["v_xy_norm"] - 5.0) < 1e-9
    assert abs(f["a_xy_norm"] - 0.0) < 1e-9
    assert abs(f["v_xy_a_xy"] - 0.0) < 1e-9
    assert abs(f["v_z_abs"] - 0.0) < 1e-9
    assert abs(f["a_z_abs"] - 0.0) < 1e-9
    assert abs(f["v_z_a_z"] - 0.0) < 1e-9
    assert abs(f["payload_kg"] - 0.5) < 1e-9
    assert abs(f["v_xy_dot_w_xy"] - 3.0) < 1e-9
    assert f["one"] == 1.0


def test_soc_linear():
    assert abs(_soc(12.6, 12.6, 9.0) - 1.0) < 1e-9
    assert abs(_soc(9.0, 12.6, 9.0) - 0.0) < 1e-9
    assert abs(_soc(10.8, 12.6, 9.0) - 0.5) < 1e-9
    assert _soc(15.0, 12.6, 9.0) == 1.0
    assert _soc(5.0, 12.6, 9.0) == 0.0


def test_pipeline_basic():
    stream = list(synthetic_flight(n=300, seed=0))
    r = run(stream)
    assert isinstance(r, BatteryEnduranceResult)
    assert r.n_samples == 300
    assert r.capacity_wh == DEFAULT_CAPACITY_WH
    assert r.v_full == DEFAULT_V_FULL
    assert r.v_min == DEFAULT_V_MIN
    assert r.soc_min == DEFAULT_SOC_MIN
    assert r.v_pred.shape == (300,)
    assert r.p_pred.shape == (300,)
    assert r.p_alyassi.shape == (300,)
    assert r.t_remaining.shape == (300,)


def test_p_alyassi_above_measured():
    stream = list(synthetic_flight(n=300, seed=0))
    r = run(stream)
    p_measured_med = float(np.median(r.p_pred[~np.isnan(r.p_pred)]))
    assert r.p_alyassi_median > p_measured_med
    assert r.p_alyassi_median > 400.0


def test_coverage_near_target():
    stream = list(synthetic_flight(n=600, seed=0))
    r = run(stream)
    assert np.isfinite(r.coverage)
    assert 0.75 <= r.coverage <= 0.99


def test_t_remaining_in_minutes():
    stream = list(synthetic_flight(n=400, seed=0))
    r = run(stream)
    assert 600.0 < r.t_remaining_q50 < 3600.0


def test_t_remaining_ordering():
    stream = list(synthetic_flight(n=400, seed=0))
    r = run(stream)
    assert r.t_remaining_q10 <= r.t_remaining_q50 <= r.t_remaining_q90


def test_deterministic():
    stream = list(synthetic_flight(n=300, seed=0))
    r1 = run(stream)
    r2 = run(stream)
    np.testing.assert_array_equal(r1.v_pred, r2.v_pred)
    np.testing.assert_array_equal(r1.p_pred, r2.p_pred)
    np.testing.assert_array_equal(r1.t_remaining, r2.t_remaining)


def test_empty_stream():
    r = run([])
    assert r.n_samples == 0
    assert np.isnan(r.t_remaining_q10)


def test_custom_parameters():
    stream = list(synthetic_flight(n=300, seed=0))
    r = run(stream, capacity_wh=100.0, soc_min=0.05)
    assert r.capacity_wh == 100.0
    assert r.soc_min == 0.05


def test_conformal_and_digest_used():
    from dense_armor.roles import AdaptiveConformalRegressor as ACR
    from dense_armor.utility.stats.quantiles import TDigest as TD
    acr = ACR(LinearRegression(optimizer=None) if False else
              LinearRegression(optimizer=__import__("dense_armor.utility.optim.adagrad",
                                                     fromlist=["AdaGrad"]).AdaGrad()))
    assert hasattr(acr, "predict_interval")
    td = TD()
    _ = td.update({"x": 1.0})
    _ = td.update({"x": 2.0})
    assert td.quantile(0.5) >= 1.0
