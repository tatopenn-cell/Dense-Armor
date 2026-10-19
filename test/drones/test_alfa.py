"""Tests for pipelines/drones/alfa.py (ALFA reader)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from pipelines.drones.alfa import (
    PWM_MIN,
    PWM_SPAN,
    alfa_fault_time,
    read_alfa,
)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _mini_flight(tmp_path: Path, fault_at_ns: int | None,
                 cmd: str = "10.0", meas: str = "5.0") -> Path:
    d = tmp_path / "carbonZ_test"
    d.mkdir()
    t0 = 1_531_943_810_000_000_000
    ts = [t0 + i * 100_000_000 for i in range(5)]
    imu = [[str(t), "0", "0", "", "0.1", "0.2", "0.3",
            "0.0", "0.0", "9.81"] for t in ts]
    _write(d / "carbonZ_test-mavros-imu-data_raw.csv",
           ["%time", "field.header.seq", "field.header.stamp",
            "field.header.frame_id", "field.angular_velocity.x",
            "field.angular_velocity.y", "field.angular_velocity.z",
            "field.linear_acceleration.x", "field.linear_acceleration.y",
            "field.linear_acceleration.z"], imu)
    gps = [[str(t), "0", "0", "map", "1.0", "2.0", "0.0",
            "0", "0", "0"] for t in ts]
    _write(d / "carbonZ_test-mavros-global_position-raw-gps_vel.csv",
           ["%time", "field.header.seq", "field.header.stamp",
            "field.header.frame_id", "field.twist.linear.x",
            "field.twist.linear.y", "field.twist.linear.z",
            "field.twist.angular.x", "field.twist.angular.y",
            "field.twist.angular.z"], gps)
    rc = [[str(t), "0", "0", "", "1500", "1600", "1260", "1520",
           "0", "0", "0", "0"] for t in ts]
    _write(d / "carbonZ_test-mavros-rc-out.csv",
           ["%time", "field.header.seq", "field.header.stamp",
            "field.header.frame_id", "field.channels0", "field.channels1",
            "field.channels2", "field.channels3", "field.channels4",
            "field.channels5", "field.channels6", "field.channels7"], rc)
    bat = [[str(t), "0", "0", "", "12.5", "1.0"] for t in ts]
    _write(d / "carbonZ_test-mavros-battery.csv",
           ["%time", "field.header.seq", "field.header.stamp",
            "field.header.frame_id", "field.voltage", "field.current"], bat)
    for axis in ("roll", "pitch", "yaw"):
        rows = [[str(t), "0", "0", "NED", cmd, meas] for t in ts]
        _write(d / f"carbonZ_test-mavros-nav_info-{axis}.csv",
               ["%time", "field.header.seq", "field.header.stamp",
                "field.header.frame_id", "field.commanded",
                "field.measured"], rows)
    if fault_at_ns is not None:
        _write(d / "carbonZ_test-failure_status-engines.csv",
               ["%time", "field.data"],
               [[str(fault_at_ns), "1"], [str(t0 + 5_000_000_000), "1"]])
    return d


def test_read_alfa_yields_one_dict_per_imu_sample(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    samples = list(read_alfa(d))
    assert len(samples) == 5
    s = samples[0]
    np.testing.assert_allclose(s["gyro"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(s["acc"], [0.0, 0.0, 9.81])
    np.testing.assert_allclose(s["gps_v"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(s["motor"],
                               [(1500 - PWM_MIN) / PWM_SPAN,
                                (1600 - PWM_MIN) / PWM_SPAN,
                                (1260 - PWM_MIN) / PWM_SPAN,
                                (1520 - PWM_MIN) / PWM_SPAN])
    assert s["battery_v"] == 12.5
    deg_to_rad = np.pi / 180.0
    assert "commanded" not in s
    np.testing.assert_allclose(s["attitude_cmd"], [10.0 * deg_to_rad] * 3)
    np.testing.assert_allclose(s["attitude"], [5.0 * deg_to_rad] * 3)
    np.testing.assert_allclose(s["attitude_error"], [5.0 * deg_to_rad] * 3)
    assert s["fault_started"] is False
    assert s["fault_kind"] is None


def test_attitude_error_wraps_to_plus_minus_pi(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None, cmd="350.0", meas="10.0")
    s = next(iter(read_alfa(d)))
    np.testing.assert_allclose(s["attitude_error"],
                               [-20.0 * np.pi / 180.0] * 3)


def test_attitude_files_of_different_length(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    t0 = 1_531_943_810_000_000_000
    short = [[str(t0 + i * 100_000_000), "0", "0", "NED", "20.0", "15.0"]
             for i in range(3)]
    _write(d / "carbonZ_test-mavros-nav_info-yaw.csv",
           ["%time", "field.header.seq", "field.header.stamp",
            "field.header.frame_id", "field.commanded", "field.measured"],
           short)
    samples = list(read_alfa(d))
    assert len(samples) == 5
    np.testing.assert_allclose(samples[4]["attitude_error"][2],
                               5.0 * np.pi / 180.0)


def test_t_is_seconds_from_first_imu_sample(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    samples = list(read_alfa(d))
    np.testing.assert_allclose([s["t"] for s in samples],
                               [0.0, 0.1, 0.2, 0.3, 0.4])


def test_alfa_fault_time_with_fault(tmp_path):
    t0 = 1_531_943_810_000_000_000
    fault_ns = t0 + 200_000_000
    d = _mini_flight(tmp_path, fault_at_ns=fault_ns)
    t_fault, kind = alfa_fault_time(d)
    assert kind == "engines"
    assert abs(t_fault - 0.2) < 1e-9


def test_alfa_fault_time_without_fault(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    assert alfa_fault_time(d) == (None, None)


def test_fault_started_flips_at_fault_time(tmp_path):
    t0 = 1_531_943_810_000_000_000
    d = _mini_flight(tmp_path, fault_at_ns=t0 + 200_000_000)
    samples = list(read_alfa(d))
    flags = [s["fault_started"] for s in samples]
    assert flags == [False, False, True, True, True]
    kinds = [s["fault_kind"] for s in samples]
    assert kinds == [None, None, "engines", "engines", "engines"]


def test_missing_csv_raises(tmp_path):
    d = tmp_path / "carbonZ_broken"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        list(read_alfa(d))


def test_align_takes_last_value_at_or_before(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    bat = d / "carbonZ_test-mavros-battery.csv"
    with open(bat, newline="") as fh:
        rows = list(csv.reader(fh))
    t0 = int(rows[1][0])
    rows = [rows[0], [str(t0 + 150_000_000), "0", "0", "", "9.0", "1.0"]]
    _write(bat, rows[0], rows[1:])
    samples = list(read_alfa(d))
    assert np.isnan(samples[0]["battery_v"])
    assert np.isnan(samples[1]["battery_v"])
    assert samples[2]["battery_v"] == 9.0
    assert samples[3]["battery_v"] == 9.0


def test_failure_status_with_one_row_returns_none(tmp_path):
    d = _mini_flight(tmp_path, fault_at_ns=None)
    _write(d / "carbonZ_test-failure_status-engines.csv",
           ["%time", "field.data"], [])
    assert alfa_fault_time(d) == (None, None)
