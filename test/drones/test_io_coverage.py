"""Coverage tests for pipelines/drones/io.py without optional deps."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipelines.drones.io import (
    _import_pymavlink,
    read_mavlink_log,
)


def test_import_pymavlink_raises_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymavlink", None)
    with pytest.raises(ImportError, match="pymavlink"):
        _import_pymavlink()


class _FakeMsg:
    """One message with the fields read_mavlink_log looks at."""

    def __init__(self, kind, **fields):
        self._kind = kind
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self):
        return self._kind


def test_read_mavlink_log_with_fake_mavutil(monkeypatch):
    """The body of read_mavlink_log runs on a fake mavutil.mavlogfile."""
    msgs = [
        _FakeMsg("HIGHRES_IMU", time_usec=1_000_000,
                 xgyro=0.1, ygyro=0.2, zgyro=0.3,
                 xacc=0.0, yacc=0.0, zacc=9.81),
        _FakeMsg("GPS_RAW_INT", time_usec=1_000_000,
                 vel=100, cog=0),
        _FakeMsg("SERVO_OUTPUT_RAW", time_usec=1_000_000,
                 servo1_raw=500, servo2_raw=600,
                 servo3_raw=700, servo4_raw=800),
        _FakeMsg("BATTERY_STATUS", voltages=[12600]),
    ]
    mav = MagicMock()
    mav.recv_match.side_effect = list(msgs) + [None]
    fake_mavutil = MagicMock()
    fake_mavutil.mavlogfile.return_value = mav

    monkeypatch.setattr(
        "pipelines.drones.io._import_pymavlink",
        lambda: fake_mavutil,
    )
    out = list(read_mavlink_log("/tmp/fake.tlog"))
    assert len(out) >= 1
    first = out[0]
    assert first["gyro"] is not None
    np.testing.assert_allclose(first["gyro"], [0.1, 0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(first["motor"], [0.5, 0.6, 0.7, 0.8],
                               atol=1e-9)
    assert abs(first["battery_v"] - 12.6) < 1e-9


def test_read_mavlink_log_attitude_when_no_highres(monkeypatch):
    msgs = [
        _FakeMsg("ATTITUDE", time_boot_ms=1000,
                 rollspeed=0.5, pitchspeed=0.6, yawspeed=0.7),
    ]
    mav = MagicMock()
    mav.recv_match.side_effect = list(msgs) + [None]
    fake_mavutil = MagicMock()
    fake_mavutil.mavlogfile.return_value = mav
    monkeypatch.setattr(
        "pipelines.drones.io._import_pymavlink",
        lambda: fake_mavutil,
    )
    out = list(read_mavlink_log("/tmp/fake.tlog"))
    assert len(out) == 1
    np.testing.assert_allclose(out[0]["gyro"], [0.5, 0.6, 0.7], atol=1e-9)


def test_read_mavlink_log_attitude_target(monkeypatch):
    msgs = [
        _FakeMsg("HIGHRES_IMU", time_usec=1_000_000,
                 xgyro=0.0, ygyro=0.0, zgyro=0.0,
                 xacc=0.0, yacc=0.0, zacc=9.81),
        _FakeMsg("ATTITUDE_TARGET", time_boot_ms=1000,
                 body_roll_rate=0.1, body_pitch_rate=0.2,
                 body_yaw_rate=0.3),
    ]
    mav = MagicMock()
    mav.recv_match.side_effect = list(msgs) + [None]
    fake_mavutil = MagicMock()
    fake_mavutil.mavlogfile.return_value = mav
    monkeypatch.setattr(
        "pipelines.drones.io._import_pymavlink",
        lambda: fake_mavutil,
    )
    out = list(read_mavlink_log("/tmp/fake.tlog"))
    assert out[0]["commanded"] is not None


def test_read_mavlink_log_no_time_skipped(monkeypatch):
    """A message without time_usec or time_boot_ms is skipped."""
    msgs = [
        _FakeMsg("HIGHRES_IMU", xgyro=0.0, ygyro=0.0, zgyro=0.0,
                 xacc=0.0, yacc=0.0, zacc=9.81),
    ]
    mav = MagicMock()
    mav.recv_match.side_effect = list(msgs) + [None]
    fake_mavutil = MagicMock()
    fake_mavutil.mavlogfile.return_value = mav
    monkeypatch.setattr(
        "pipelines.drones.io._import_pymavlink",
        lambda: fake_mavutil,
    )
    assert list(read_mavlink_log("/tmp/fake.tlog")) == []
