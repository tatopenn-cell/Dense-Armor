"""Reader for the ALFA dataset (Keipour, Mousaei & Scherer 2020, IJRR).

ALFA is a dataset of 47 processed flights of a Carbon-Z T-28 fixed-wing
with ArduPilot (paper section 2.2, page 2), with the time and the type
of the fault in the ground truth (section 3.3, page 4). The processed
flights are folders of one CSV per ROS topic (section 3.1, page 3).

``read_alfa(flight_dir)`` yields one dict per sample with the fields of
:func:`pipelines.drones.io.read_mavlink_log` -- ``t``, ``gyro``,
``commanded``, ``acc``, ``gps_v``, ``motor``, ``battery_v`` -- plus
``fault_started`` and ``fault_kind``. The samples are the rows of
``mavros-imu-data_raw.csv``; every other field is aligned to the most
recent value at or before the IMU timestamp, as in
``pipelines/drones/io.py``.

``alfa_fault_time(flight_dir)`` returns ``(t_fault, kind)`` from the
``failure_status-*.csv`` file of the flight, or ``(None, None)`` for a
flight with no fault. The time is in seconds from the first IMU sample,
the same origin as ``t`` in ``read_alfa``.

The mapping from the CSV to the field is my choice: the paper names the
topic families (section 3.3, page 4: ``mavros/nav_info/roll`` and so on)
but not the exact column for each field of this reader.

- ``gyro``: ``mavros-imu-data_raw.csv``, ``field.angular_velocity.*``,
  rad/s.
- ``acc``: ``mavros-imu-data_raw.csv``, ``field.linear_acceleration.*``,
  m/s^2.
- ``commanded``: ``mavros-nav_info-{roll,pitch,yaw}.csv``,
  ``field.commanded``, deg/s, converted to rad/s.
- ``gps_v``: ``mavros-global_position-raw-gps_vel.csv``,
  ``field.twist.linear.*``, m/s.
- ``motor``: ``mavros-rc-out.csv``, ``field.channels0..3``, PWM in
  [1000, 2000], normalised to [0, 1] as ``(pwm - 1000) / 1000``.
- ``battery_v``: ``mavros-battery.csv``, ``field.voltage``, V.

Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> from pipelines.drones.alfa import alfa_fault_time
    >>> d = Path(tempfile.mkdtemp())
    >>> h = ["%time,field.data", "1050000000,1"]
    >>> _ = (d / "x-failure_status-engines.csv").write_text(
    ...     chr(10).join(h))
    >>> g = ["%time,x", "1000000000,0"]
    >>> _ = (d / "x-mavros-imu-data_raw.csv").write_text(
    ...     chr(10).join(g))
    >>> alfa_fault_time(d)
    (0.05, 'engines')
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

PWM_MIN = 1000.0
PWM_SPAN = 1000.0
DEG_TO_RAD = np.pi / 180.0
NS_PER_S = 1e9

_IMU = "mavros-imu-data_raw"
_GPS_VEL = "mavros-global_position-raw-gps_vel"
_RC_OUT = "mavros-rc-out"
_BATTERY = "mavros-battery"
_NAV = ("mavros-nav_info-roll", "mavros-nav_info-pitch",
        "mavros-nav_info-yaw")
_FAILURE = "failure_status-"


def _find(flight: Path, stem: str) -> Path:
    matches = sorted(flight.glob(f"*{stem}.csv"))
    if not matches:
        raise FileNotFoundError(f"{stem}.csv not in {flight}")
    return matches[0]


def _read(flight: Path, stem: str, columns: list[str]
          ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = _find(flight, stem)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: header.index(c) for c in columns}
        t: list[int] = []
        vals: dict[str, list[float]] = {c: [] for c in columns}
        for row in reader:
            t.append(int(row[0]))
            for c in columns:
                vals[c].append(float(row[idx[c]]))
    return (np.array(t, dtype=np.int64),
            {c: np.array(v, dtype=float) for c, v in vals.items()})


def _at(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray
        ) -> np.ndarray:
    pos: np.ndarray = np.asarray(
        np.searchsorted(t_src, t_dst, side="right"), dtype=np.int64) - 1
    out = np.full(t_dst.shape, np.nan, dtype=float)
    ok = pos >= 0
    out[ok] = v_src[pos[ok]]
    return out


def _origin(flight: Path) -> int:
    t, _ = _read(flight, _IMU, [])
    return int(t[0])


def alfa_fault_time(flight_dir) -> tuple[float | None, str | None]:
    """Return ``(t_fault, kind)`` from the ground truth of a flight.

    The ground truth is one ``failure_status-<surface>.csv`` file in the
    flight folder (paper section 3.3, page 4). Its first row is the
    time of the fault, in nanoseconds from the epoch; ``kind`` is the
    surface in the file name (``engines``, ``aileron``, ``rudder``,
    ``elevator``). The time is in seconds from the first IMU sample.

    Args:
        flight_dir: folder of one processed flight.

    Returns:
        ``(t_fault, kind)``, or ``(None, None)`` when the flight has no
        fault.
    """
    flight = Path(flight_dir)
    files = sorted(flight.glob(f"*{_FAILURE}*.csv"))
    if not files:
        return None, None
    with open(files[0], newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None, None
    kind = files[0].stem.split(_FAILURE, 1)[1]
    t_fault = (int(rows[1][0]) - _origin(flight)) / NS_PER_S
    return float(t_fault), kind


def read_alfa(flight_dir) -> Iterator[dict[str, Any]]:
    """Yield one dict per IMU sample of a processed ALFA flight.

    Args:
        flight_dir: folder of one processed flight.

    Yields:
        One dict per sample, with the fields of
        :func:`pipelines.drones.io.read_mavlink_log` plus
        ``fault_started`` and ``fault_kind``.

    Raises:
        FileNotFoundError: if a required CSV is missing.
    """
    flight = Path(flight_dir)
    t_imu, imu = _read(flight, _IMU, [
        "field.angular_velocity.x", "field.angular_velocity.y",
        "field.angular_velocity.z", "field.linear_acceleration.x",
        "field.linear_acceleration.y", "field.linear_acceleration.z",
    ])
    t_gps, gps = _read(flight, _GPS_VEL, [
        "field.twist.linear.x", "field.twist.linear.y",
        "field.twist.linear.z",
    ])
    t_rc, rc = _read(flight, _RC_OUT, [
        "field.channels0", "field.channels1",
        "field.channels2", "field.channels3",
    ])
    t_bat, bat = _read(flight, _BATTERY, ["field.voltage"])
    nav_t, nav = [], {}
    for stem in _NAV:
        t_n, v = _read(flight, stem, ["field.commanded"])
        nav_t.append(t_n)
        nav[stem] = v["field.commanded"]
    t_nav = np.stack(nav_t)

    t0 = int(t_imu[0])
    secs = (t_imu - t0) / NS_PER_S
    gyro = np.stack([imu["field.angular_velocity.x"],
                     imu["field.angular_velocity.y"],
                     imu["field.angular_velocity.z"]], axis=1)
    acc = np.stack([imu["field.linear_acceleration.x"],
                    imu["field.linear_acceleration.y"],
                    imu["field.linear_acceleration.z"]], axis=1)
    gps_v = np.stack([_at(t_gps, gps["field.twist.linear.x"], t_imu),
                      _at(t_gps, gps["field.twist.linear.y"], t_imu),
                      _at(t_gps, gps["field.twist.linear.z"], t_imu)], axis=1)
    motor = np.stack([(_at(t_rc, rc["field.channels0"], t_imu) - PWM_MIN)
                      / PWM_SPAN,
                      (_at(t_rc, rc["field.channels1"], t_imu) - PWM_MIN)
                      / PWM_SPAN,
                      (_at(t_rc, rc["field.channels2"], t_imu) - PWM_MIN)
                      / PWM_SPAN,
                      (_at(t_rc, rc["field.channels3"], t_imu) - PWM_MIN)
                      / PWM_SPAN], axis=1)
    battery_v = _at(t_bat, bat["field.voltage"], t_imu)
    cmd = []
    for i, stem in enumerate(_NAV):
        cmd.append(_at(t_nav[i], nav[stem], t_imu) * DEG_TO_RAD)
    commanded = np.stack(cmd, axis=1)

    t_fault, kind = alfa_fault_time(flight)
    for i in range(len(secs)):
        started = (t_fault is not None and float(secs[i]) >= t_fault)
        yield {
            "t": float(secs[i]),
            "gyro": gyro[i].copy(),
            "commanded": commanded[i].copy(),
            "acc": acc[i].copy(),
            "gps_v": gps_v[i].copy(),
            "motor": motor[i].copy(),
            "battery_v": float(battery_v[i]),
            "fault_started": bool(started),
            "fault_kind": kind if started else None,
        }
