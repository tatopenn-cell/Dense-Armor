"""Tests for edge.power: energy, calibration, sysfs source, synthetic source."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from edge.power import (
    CALIBRATION_MW,
    PowerTrace,
    SyntheticPowerSource,
    SysfsPowerSource,
    _demo,
    calibrate_mw,
    energy_j,
    main,
    net_energy_j,
)


def _trace(mw, dt=1.0):
    mw = np.asarray(mw, dtype=float)
    t = np.arange(mw.size) * dt
    return PowerTrace(t_s=t, mw=mw)


def _sysfs_file(tmp_path, mw=5000):
    p = tmp_path / "power1_input"
    p.write_text(f"{mw}\n", encoding="utf-8")
    return p


def _run_sysfs(src, dt_s):
    trace = src.start()
    time.sleep(3.0 * dt_s)
    return trace.stop()


def test_constant_power_gives_pt_over_1000():
    trace = PowerTrace(t_s=np.array([0.0, 2.0]), mw=np.array([5000.0, 5000.0]))
    assert energy_j(trace) == pytest.approx(10.0, abs=1e-9)


def test_constant_power_three_samples_matches():
    trace = _trace([2500.0, 2500.0, 2500.0], dt=0.5)
    assert energy_j(trace) == pytest.approx(2500.0 * 1.0 / 1000.0, abs=1e-9)


def test_trace_equal_to_idle_gives_zero_net_energy():
    trace = _trace([3000.0, 3000.0, 3000.0], dt=1.0)
    assert net_energy_j(trace, idle_mw=3000.0) == 0.0


def test_trace_below_idle_gives_zero_net_energy():
    trace = _trace([2000.0, 2000.0, 2000.0], dt=1.0)
    assert net_energy_j(trace, idle_mw=3000.0) == 0.0


def test_net_energy_removes_only_the_part_above_idle():
    trace = _trace([4000.0, 4000.0], dt=1.0)
    assert net_energy_j(trace, idle_mw=3000.0) == pytest.approx(1.0, abs=1e-9)


def test_linear_ramp_gives_trapezoid_area():
    trace = PowerTrace(t_s=np.array([0.0, 2.0]), mw=np.array([1000.0, 3000.0]))
    assert energy_j(trace) == pytest.approx(
        (1000.0 + 3000.0) / 2.0 * 2.0 / 1000.0, abs=1e-9)


def test_three_point_ramp_matches_trapezoid():
    trace = PowerTrace(t_s=np.array([0.0, 1.0, 2.0]),
                       mw=np.array([1000.0, 2000.0, 3000.0]))
    assert energy_j(trace) == pytest.approx(4.0, abs=1e-9)


def test_trace_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        PowerTrace(t_s=np.array([0.0, 1.0]), mw=np.array([100.0]))


def test_trace_needs_at_least_two_samples():
    with pytest.raises(ValueError, match="at least two"):
        PowerTrace(t_s=np.array([0.0]), mw=np.array([100.0]))


def test_trace_needs_one_dimensional_arrays():
    with pytest.raises(ValueError, match="1-D"):
        PowerTrace(t_s=np.array([[0.0, 1.0]]), mw=np.array([[100.0, 200.0]]))


def test_calibrate_reproduces_table_i_at_1000():
    expected = {
        "AGX Orin": 1.02 * 1000 + 3115.39,
        "Xavier NX": 1.10 * 1000 + 3130.41,
        "TX2": 0.90 * 1000 + 1998.80,
        "Nano": 1.11 * 1000 + 232.60,
    }
    for device, y in expected.items():
        assert calibrate_mw(1000.0, device) == pytest.approx(y, abs=1e-9)


def test_calibration_table_has_four_devices():
    assert set(CALIBRATION_MW) == {"AGX Orin", "Xavier NX", "TX2", "Nano"}


def test_calibrate_unknown_device_raises():
    with pytest.raises(ValueError, match="unknown device"):
        calibrate_mw(1000.0, "Jetson Nano")


def test_calibrate_is_not_applied_by_default():
    trace = _trace([1000.0, 1000.0], dt=1.0)
    e_raw = energy_j(trace)
    e_calibrated = energy_j(PowerTrace(
        trace.t_s, np.array([calibrate_mw(v, "Nano") for v in trace.mw])))
    assert e_calibrated > e_raw


def test_sysfs_source_reads_the_written_values(tmp_path):
    p = _sysfs_file(tmp_path, 5000)
    src = SysfsPowerSource(str(p), dt_s=0.01)
    trace = _run_sysfs(src, 0.01)
    assert trace.mw.size >= 2
    assert np.all(trace.mw == 5000.0)
    assert trace.t_s.size == trace.mw.size


def test_sysfs_source_stop_joins_the_thread(tmp_path):
    p = _sysfs_file(tmp_path, 5000)
    before = {th.ident for th in threading.enumerate()}
    src = SysfsPowerSource(str(p), dt_s=0.01)
    trace = _run_sysfs(src, 0.01)
    after = {th.ident for th in threading.enumerate()}
    assert after == before
    assert trace.mw.size >= 2


def test_sysfs_source_start_twice_raises(tmp_path):
    p = _sysfs_file(tmp_path, 5000)
    src = SysfsPowerSource(str(p), dt_s=0.01)
    src.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            src.start()
    finally:
        time.sleep(0.03)
        src.stop()


def test_sysfs_source_stop_without_start_raises(tmp_path):
    p = _sysfs_file(tmp_path, 5000)
    src = SysfsPowerSource(str(p), dt_s=0.01)
    with pytest.raises(RuntimeError, match="not started"):
        src.stop()


def test_sysfs_source_one_sample_raises(tmp_path):
    p = _sysfs_file(tmp_path, 5000)
    src = SysfsPowerSource(str(p), dt_s=10.0)
    src.start()
    with pytest.raises(RuntimeError, match="at least two"):
        src.stop()


def test_sysfs_source_dt_must_be_positive():
    with pytest.raises(ValueError, match="dt_s must be positive"):
        SysfsPowerSource("x", dt_s=0.0)


def test_sysfs_source_missing_file_raises(tmp_path):
    src = SysfsPowerSource(str(tmp_path / "missing"), dt_s=0.01)
    src.start()
    with pytest.raises(RuntimeError, match="read failed"):
        src.stop()


def test_synthetic_source_constant_power():
    src = SyntheticPowerSource(mw=4000.0, dt_s=0.1, duration_s=1.0)
    trace = src.start().stop()
    assert trace.mw.size >= 2
    assert np.all(trace.mw == 4000.0)
    assert energy_j(trace) == pytest.approx(
        4000.0 * trace.t_s[-1] / 1000.0, abs=1e-9)


def test_synthetic_source_without_duration_uses_wall_time():
    src = SyntheticPowerSource(mw=2000.0, dt_s=0.05, duration_s=None)
    trace = src.start().stop()
    assert trace.t_s[-1] > 0.0
    assert np.all(trace.mw == 2000.0)


def test_synthetic_source_stop_without_start_raises():
    src = SyntheticPowerSource(mw=1000.0)
    with pytest.raises(RuntimeError, match="not started"):
        src.stop()


def test_synthetic_source_dt_must_be_positive():
    with pytest.raises(ValueError, match="dt_s must be positive"):
        SyntheticPowerSource(mw=1000.0, dt_s=-1.0)


def test_synthetic_source_duration_must_be_positive():
    with pytest.raises(ValueError, match="duration_s must be positive"):
        SyntheticPowerSource(mw=1000.0, duration_s=0.0)


def test_energy_of_synthetic_matches_pt():
    src = SyntheticPowerSource(mw=3000.0, dt_s=0.1, duration_s=2.0)
    trace = src.start().stop()
    assert energy_j(trace) == pytest.approx(6.0, abs=1e-3)


def test_demo_returns_zero_and_prints(capsys):
    rc = _demo()
    out = capsys.readouterr().out
    assert rc == 0
    assert "synthetic" in out
    assert "energy" in out
    assert "calibrated AGX Orin" in out


def test_main_with_demo_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["edge.power", "--demo"])
    rc = main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "energy" in out


def test_main_without_flags_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["edge.power"])
    rc = main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage" in out.lower()
