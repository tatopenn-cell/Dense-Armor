"""Tests for pipelines/drones/io.py (task D1, item A)."""

import struct
import sys
from typing import ClassVar

import numpy as np
import pytest

from pipelines.drones.io import (
    _align,
    _align_records,
    _mav_time,
    _ulog_series,
    read_mavlink_log,
    read_ulog,
    synthetic_flight,
)

# ------------------------------------------------------------------
# Helpers to build fixtures
# ------------------------------------------------------------------

def _require_pyulog():
    pytest.importorskip("pyulog")


def _require_pymavlink():
    pytest.importorskip("pymavlink")
    pytest.importorskip("serial")


def _build_ulog(messages, start_ts_us=1_000_000, path=None):
    """Build a valid ULog file; return the path."""
    import pyulog
    HEADER = b"ULog\x01\x125"
    TYPE_FMT = {k: v[0] for k, v in pyulog.ULog._UNPACK_TYPES.items()}
    out = bytearray(HEADER)
    out.extend(struct.pack("<B", 1))
    out.extend(struct.pack("<Q", start_ts_us))
    for name, fields, _ in messages:
        data = bytearray((name + ":").encode())
        for type_str, n_arr, fname in fields:
            if n_arr > 0:
                data.extend(f"{type_str}[{n_arr}] {fname};".encode())
            else:
                data.extend(f"{type_str} {fname};".encode())
        out.extend(struct.pack("<HB", len(data),
                               pyulog.ULog.MSG_TYPE_FORMAT))
        out.extend(data)
    for msg_id, (name, _fields, _records) in enumerate(messages, 1):
        data = bytearray()
        data.extend(struct.pack("<BH", 0, msg_id))
        data.extend(name.encode())
        out.extend(struct.pack("<HB", len(data),
                               pyulog.ULog.MSG_TYPE_ADD_LOGGED_MSG))
        out.extend(data)
    all_data = []
    for msg_id, (name, fields, records) in enumerate(messages, 1):
        for rec in records:
            data = bytearray(struct.pack("<H", msg_id))
            for type_str, n_arr, fname in fields:
                fmt = "<" + TYPE_FMT[type_str]
                v = rec[fname]
                if n_arr > 0:
                    for x in v:
                        data.extend(struct.pack(fmt, x))
                else:
                    data.extend(struct.pack(fmt, v))
            all_data.append((rec.get("timestamp", 0), bytes(data)))
    all_data.sort(key=lambda x: x[0])
    for _t, payload in all_data:
        out.extend(struct.pack("<HB", len(payload),
                               pyulog.ULog.MSG_TYPE_DATA))
        out.extend(payload)
    p = path
    p.write_bytes(bytes(out))
    return p


def _build_tlog(messages, path):
    """Build a valid tlog; messages is a list of ('type', kwargs) tuples."""
    from pymavlink import mavutil
    with open(path, "wb") as fh:
        mav = mavutil.mavlink.MAVLink(fh, srcSystem=1, srcComponent=1)
        for kind, kwargs in messages:
            getattr(mav, kind + "_send")(**kwargs)
    return path


def _hb():
    from pymavlink import mavutil
    return ("heartbeat", {
        "type": mavutil.mavlink.MAV_TYPE_QUADROTOR,
        "autopilot": mavutil.mavlink.MAV_AUTOPILOT_PX4,
        "base_mode": 0, "custom_mode": 0,
        "system_status": mavutil.mavlink.MAV_STATE_ACTIVE,
    })


def _att(t_ms, roll=0.0):
    return ("attitude", {
        "time_boot_ms": t_ms,
        "roll": roll, "pitch": 0.0, "yaw": 0.0,
        "rollspeed": 0.01, "pitchspeed": 0.02, "yawspeed": 0.03,
    })


def _hr_imu(t_us):
    return ("highres_imu", {
        "time_usec": t_us, "xacc": 0.1, "yacc": 0.2, "zacc": 9.81,
        "xgyro": 0.01, "ygyro": 0.02, "zgyro": 0.03,
        "xmag": 0.0, "ymag": 0.0, "zmag": 0.0,
        "abs_pressure": 1000.0, "diff_pressure": 0.0,
        "pressure_alt": 0.0, "temperature": 2500,
        "fields_updated": 0xFFFF,
    })


def _gps(t_us, vel=100, cog=0):
    return ("gps_raw_int", {
        "time_usec": t_us, "fix_type": 3, "lat": 0, "lon": 0, "alt": 0,
        "eph": 0, "epv": 0, "vel": vel, "cog": cog,
        "satellites_visible": 10,
    })


def _servo(t_us):
    return ("servo_output_raw", {
        "time_usec": t_us, "port": 0,
        "servo1_raw": 500, "servo2_raw": 600,
        "servo3_raw": 700, "servo4_raw": 800,
        "servo5_raw": 0, "servo6_raw": 0, "servo7_raw": 0, "servo8_raw": 0,
    })


def _batt():
    return ("battery_status", {
        "id": 0, "battery_function": 0, "type": 0, "temperature": 2500,
        "voltages": [12600, 12500, 0, 0, 0, 0, 0, 0, 0, 0],
        "current_battery": 500, "current_consumed": 10,
        "energy_consumed": 0, "battery_remaining": 80,
    })


# ------------------------------------------------------------------
# read_ulog
# ------------------------------------------------------------------

def test_read_ulog_full(tmp_path):
    _require_pyulog()
    msgs = [
        ("sensor_combined", [
            ("uint64_t", 0, "timestamp"),
            ("float", 3, "gyro_rad"),
            ("float", 3, "accelerometer_m_s2"),
        ], [
            {"timestamp": 1_000_000, "gyro_rad": [0.1, 0.2, 0.3],
             "accelerometer_m_s2": [0.0, 0.0, 9.81]},
            {"timestamp": 1_100_000, "gyro_rad": [0.1, 0.2, 0.3],
             "accelerometer_m_s2": [0.0, 0.0, 9.81]},
        ]),
        ("vehicle_gps_position", [
            ("uint64_t", 0, "timestamp"),
            ("float", 0, "vel_n_m_s"),
            ("float", 0, "vel_e_m_s"),
            ("float", 0, "vel_d_m_s"),
        ], [
            {"timestamp": 1_000_000, "vel_n_m_s": 1.0, "vel_e_m_s": 0.0,
             "vel_d_m_s": 0.0},
        ]),
        ("actuator_outputs", [
            ("uint64_t", 0, "timestamp"),
            ("float", 4, "output"),
        ], [
            {"timestamp": 1_000_000, "output": [0.5, 0.5, 0.5, 0.5]},
        ]),
        ("battery_status", [
            ("uint64_t", 0, "timestamp"),
            ("float", 0, "voltage_v"),
        ], [
            {"timestamp": 1_000_000, "voltage_v": 12.6},
        ]),
    ]
    p = _build_ulog(msgs, path=tmp_path / "full.ulg")
    out = list(read_ulog(p))
    assert len(out) >= 1
    assert out[0]["t"] == 0.0
    np.testing.assert_allclose(out[0]["gyro"], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(out[0]["acc"], [0.0, 0.0, 9.81], atol=1e-6)
    np.testing.assert_allclose(out[0]["gps_v"], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[0]["motor"], [0.5] * 4, atol=1e-6)
    assert abs(out[0]["battery_v"] - 12.6) < 1e-4


def test_read_ulog_no_sensor_combined_uses_attitude(tmp_path):
    _require_pyulog()
    msgs = [
        ("vehicle_attitude", [
            ("uint64_t", 0, "timestamp"),
            ("float", 0, "rollspeed"),
            ("float", 0, "pitchspeed"),
            ("float", 0, "yawspeed"),
        ], [
            {"timestamp": 1_000_000, "rollspeed": 0.5, "pitchspeed": 0.6,
             "yawspeed": 0.7},
        ]),
    ]
    p = _build_ulog(msgs, path=tmp_path / "att_only.ulg")
    out = list(read_ulog(p))
    assert out[0]["acc"] is None
    np.testing.assert_allclose(out[0]["gyro"], [0.5, 0.6, 0.7], atol=1e-6)


def test_read_ulog_partial_fields(tmp_path):
    _require_pyulog()
    msgs = [
        ("sensor_combined", [
            ("uint64_t", 0, "timestamp"),
            ("float", 3, "gyro_rad"),
            ("float", 3, "accelerometer_m_s2"),
        ], [
            {"timestamp": 1_000_000, "gyro_rad": [0.1, 0.2, 0.3],
             "accelerometer_m_s2": [0.0, 0.0, 9.81]},
        ]),
    ]
    p = _build_ulog(msgs, path=tmp_path / "partial.ulg")
    out = list(read_ulog(p))
    assert out[0]["gps_v"] is None
    assert out[0]["motor"] is None
    assert np.isnan(out[0]["battery_v"])


def test_read_ulog_irrelevant_topic_yields_empty(tmp_path):
    _require_pyulog()
    msgs = [
        ("junk_topic", [
            ("uint64_t", 0, "timestamp"),
            ("float", 0, "junk"),
        ], [
            {"timestamp": 1_000_000, "junk": 1.0},
        ]),
    ]
    p = _build_ulog(msgs, path=tmp_path / "junk.ulg")
    assert list(read_ulog(p)) == []


def test_read_ulog_missing_timestamp_returns_none(tmp_path):
    _require_pyulog()
    import pyulog
    msgs = [
        ("sensor_combined", [
            ("uint64_t", 0, "timestamp"),
            ("float", 3, "gyro_rad"),
            ("float", 3, "accelerometer_m_s2"),
        ], [
            {"timestamp": 1_000_000, "gyro_rad": [0.1, 0.2, 0.3],
             "accelerometer_m_s2": [0.0, 0.0, 9.81]},
        ]),
    ]
    p = _build_ulog(msgs, path=tmp_path / "no_ts.ulg")
    u = pyulog.ULog(str(p))
    del u.data_list[0].data["timestamp"]
    assert _ulog_series(u, "sensor_combined",
                        [("gyro_rad", 3)]) is None


def test_read_ulog_import_error(monkeypatch, tmp_path):
    _require_pyulog()
    msgs = [("t", [("uint64_t", 0, "timestamp"), ("float", 0, "v")],
             [{"timestamp": 1, "v": 0.0}])]
    p = _build_ulog(msgs, path=tmp_path / "x.ulg")
    monkeypatch.setitem(sys.modules, "pyulog", None)
    with pytest.raises(ImportError, match="pyulog"):
        list(read_ulog(p))


# ------------------------------------------------------------------
# read_mavlink_log
# ------------------------------------------------------------------

def test_read_mavlink_log_full(tmp_path):
    _require_pymavlink()
    p = _build_tlog([
        _hb(),
        _hr_imu(1_000_000),
        _gps(1_000_000),
        _servo(1_000_000),
        _batt(),
    ], tmp_path / "full.tlog")
    out = list(read_mavlink_log(p))
    assert out, "no samples"
    first = out[0]
    assert first["gyro"] is not None
    assert first["acc"] is not None
    assert first["gps_v"] is not None
    assert first["motor"] is not None
    np.testing.assert_allclose(first["motor"], [0.5, 0.6, 0.7, 0.8],
                               atol=1e-6)


def test_read_mavlink_log_attitude_only(tmp_path):
    _require_pymavlink()
    p = _build_tlog([_hb(), _att(1000, roll=0.0)], tmp_path / "att.tlog")
    out = list(read_mavlink_log(p))
    assert out
    np.testing.assert_allclose(out[0]["gyro"], [0.01, 0.02, 0.03],
                               atol=1e-6)


def test_read_mavlink_log_highres_wins_over_attitude(tmp_path):
    _require_pymavlink()
    p = _build_tlog([
        _hb(), _hr_imu(1_000_000), _att(1000),
    ], tmp_path / "both.tlog")
    out = list(read_mavlink_log(p))
    g = next(s for s in out if s["gyro"] is not None)
    np.testing.assert_allclose(g["gyro"], [0.01, 0.02, 0.03], atol=1e-6)


def test_read_mavlink_log_battery_before_time_is_skipped(tmp_path):
    _require_pymavlink()
    p = _build_tlog([_batt(), _hr_imu(1_000_000)], tmp_path / "b.tlog")
    out = list(read_mavlink_log(p))
    assert out
    assert all(np.isnan(s["battery_v"]) for s in out)


def test_read_mavlink_log_battery_after_time_uses_last(tmp_path):
    _require_pymavlink()
    p = _build_tlog([_hr_imu(1_000_000), _batt()], tmp_path / "bb.tlog")
    out = list(read_mavlink_log(p))
    assert any(abs(s["battery_v"] - 12.6) < 1e-4 for s in out)


def test_read_mavlink_log_empty_file(tmp_path):
    _require_pymavlink()
    p = _build_tlog([], tmp_path / "empty.tlog")
    assert list(read_mavlink_log(p)) == []


def test_read_mavlink_log_import_error(monkeypatch, tmp_path):
    _require_pymavlink()
    p = _build_tlog([_hb()], tmp_path / "x.tlog")
    monkeypatch.setitem(sys.modules, "pymavlink", None)
    with pytest.raises(ImportError, match="pymavlink"):
        list(read_mavlink_log(p))


# ------------------------------------------------------------------
# _mav_time branches
# ------------------------------------------------------------------

def test_mav_time_branches():
    class M:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    assert _mav_time(M(time_usec=1_000_000)) == 1.0
    assert _mav_time(M(time_usec=0, time_boot_ms=1000)) == 1.0
    assert _mav_time(M(time_usec=0, time_boot_ms=0)) is None
    assert _mav_time(M()) is None


# ------------------------------------------------------------------
# _align and _align_records direct branches
# ------------------------------------------------------------------

def test_align_empty():
    assert list(_align({})) == []


def test_align_records_empty():
    assert list(_align_records([])) == []


def test_align_carry_forward():
    ts_g = np.array([0.0, 1.0])
    vg = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ts_m = np.array([0.5])
    vm = np.array([[10.0, 10.0, 10.0, 10.0]])
    out = list(_align({"gyro": (ts_g, vg), "motor": (ts_m, vm)}))
    assert len(out) == 3
    np.testing.assert_allclose(out[0]["gyro"], [1.0, 0.0, 0.0])
    assert out[0]["motor"] is None
    np.testing.assert_allclose(out[1]["motor"], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(out[2]["gyro"], [2.0, 0.0, 0.0])


# ------------------------------------------------------------------
# synthetic_flight
# ------------------------------------------------------------------

def test_synthetic_flight_no_fault():
    stream = list(synthetic_flight(n=10, seed=0))
    assert len(stream) == 10
    assert all(not s["fault_started"] for s in stream)
    assert stream[0]["motor"].shape == (4,)
    assert stream[0]["gyro"].shape == (3,)
    assert stream[0]["gps_v"].shape == (3,)
    assert stream[0]["acc"].shape == (3,)
    assert stream[0]["t"] == 0.0
    assert stream[-1]["t"] == pytest.approx(0.09, abs=1e-9)


def test_synthetic_flight_reproducible():
    a = list(synthetic_flight(n=5, seed=7))
    b = list(synthetic_flight(n=5, seed=7))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x["gyro"], y["gyro"])


def test_synthetic_flight_zero_samples():
    assert list(synthetic_flight(n=0, seed=0)) == []


def test_synthetic_flight_negative_n():
    with pytest.raises(ValueError):
        list(synthetic_flight(n=-1, seed=0))


def test_synthetic_flight_invalid_fault():
    with pytest.raises(ValueError):
        list(synthetic_flight(n=3, seed=0, fault="unknown"))


def test_synthetic_flight_actuator_fault():
    stream = list(synthetic_flight(n=20, seed=1, fault="actuator"))
    fault_at = round(0.6 * 20)
    assert not stream[fault_at - 1]["fault_started"]
    assert stream[fault_at]["fault_started"]
    assert stream[fault_at]["motor"][0] == 0.2


def test_synthetic_flight_gps_step():
    stream = list(synthetic_flight(n=20, seed=1, fault="gps_step"))
    fault_at = round(0.6 * 20)
    before = stream[fault_at - 1]["gps_v"][0]
    after = stream[fault_at]["gps_v"][0]
    assert abs(after - before) > 1.5


def test_synthetic_flight_propeller():
    stream = list(synthetic_flight(n=400, seed=2, fault="propeller"))
    fault_at = round(0.6 * 400)
    gyro_x = [s["gyro"][0] for s in stream[fault_at:]]
    assert max(abs(g) for g in gyro_x) > 0.2


def test_synthetic_flight_battery_sag():
    stream = list(synthetic_flight(n=20, seed=3, fault="battery_sag"))
    fault_at = round(0.6 * 20)
    v_before = stream[fault_at - 1]["battery_v"]
    v_after = stream[fault_at]["battery_v"]
    assert v_before - v_after > 1.5

# ------------------------------------------------------------------
# Extra: _ulog_series rami scoperti
# ------------------------------------------------------------------

def test_ulog_series_topic_absent_returns_none():
    class FakeULog:
        def __init__(self):
            self.data_list: list = []
    assert _ulog_series(FakeULog(), "sensor_combined",
                        [("gyro_rad", 3)]) is None


def test_ulog_series_no_timestamp_returns_none():
    class DS:
        name = "sensor_combined"
        data: ClassVar[dict] = {"gyro_rad[0]": [1.0]}

    class FakeULog:
        def __init__(self):
            self.data_list = [DS()]
    assert _ulog_series(FakeULog(), "sensor_combined",
                        [("gyro_rad", 3)]) is None


def test_ulog_series_no_columns_returns_none():
    class DS:
        name = "sensor_combined"
        data: ClassVar[dict] = {"timestamp": [1]}

    class FakeULog:
        def __init__(self):
            self.data_list = [DS()]
    assert _ulog_series(FakeULog(), "sensor_combined",
                        [("gyro_rad", 3)]) is None


def test_ulog_series_scalar_and_array_mixed():
    class DS:
        name = "vehicle_gps_position"
        data: ClassVar[dict] = {
            "timestamp": [1_000_000],
            "vel_n_m_s": [1.0],
            "vel_e_m_s": [0.5],
            "vel_d_m_s": [-0.1],
        }

    class FakeULog:
        def __init__(self):
            self.data_list = [DS()]
    out = _ulog_series(FakeULog(), "vehicle_gps_position",
                       [("vel_n_m_s", 0), ("vel_e_m_s", 0),
                        ("vel_d_m_s", 0)])
    assert out is not None
    np.testing.assert_allclose(out[1], [[1.0, 0.5, -0.1]])

