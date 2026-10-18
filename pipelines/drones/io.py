"""Data adapters for the drones section.

Three entry points, all of which produce the same dict of fields:

- ``read_ulog(path)``: a PX4 ULog file, read with ``pyulog``.
- ``read_mavlink_log(path)``: a MAVLink tlog file, read with ``pymavlink``.
- ``synthetic_flight(n, seed, fault=None)``: a labelled synthetic stream.

The fields, in every dict:

    t           time in seconds since the first sample of the log.
    gyro        measured angular velocity, rad/s, (x, y, z).
    commanded   reference angular velocity, rad/s, (x, y, z), or None
                when the log has no setpoint topic.
    acc         linear acceleration, m/s^2, (x, y, z).
    gps_v       GPS velocity, m/s, (north, east, down).
    motor       motor commands, normalised in [0, 1], (n_motors,).
    battery_v   battery voltage, V, scalar.

The two readers align every field by timestamp: at each time in the
union of the timestamps seen, a field takes the most recent value at or
before that time. A field that has never been seen is None.

On ALFA (Keipour, Mousaei & Scherer 2020, IJRR). The platform is a
Carbon-Z T-28 fixed-wing with a modified ArduPilot firmware (paper
section 2.1-2.2, page 2), not a PX4 quadrotor. The processed files are
``.mat``, ``.bag`` and one ``.csv`` per topic (paper section 3.1, page
3); the telemetry is MAVLink ``.tlog``, the dataflash is ``.bin``. To
read ALFA use ``read_mavlink_log`` on the telemetry ``.tlog``;
``read_ulog`` is for PX4 ULog, which ALFA does not provide. The two
readers are both present for this reason.

Optional dependencies (extra ``drone``):

- ``read_ulog`` needs ``pyulog``.
- ``read_mavlink_log`` needs ``pymavlink`` and ``pyserial``.

Both are imported lazily; a missing package gives an ImportError that
names the package and the extra to install.

The synthetic fault models (all "my choice" unless stated):

- ``actuator_loss`` follows the input-multiplicative model of Garg and
  Fan 2023, section II, equation (4), page 2: a faulty actuator scales
  the commanded input by ``Theta_i in [0, 1)``. The paper's equation is
  on the thrust and the moments; the synthetic stream applies it to the
  commanded rate, which is an approximation.
- ``gps_ramp`` follows the description of SemperFi (Sathaye et al.
  2021) section 2.2, page 3: the attacker "gradually introduces
  offsets". The ramp rate is my choice.
- ``propeller`` follows Zhang et al. 2023, section V, page 5: only one
  propeller is damaged and the fault is visible on the onboard IMU. The
  vibration frequency and amplitude are my choice.
- ``battery_drop`` has no paper: a constant voltage drop. Alyassi et al.
  2017 is a nine-term regression on power (section IV, pages 4-5); it is
  used in the battery pipeline, not here.

Example:
    >>> from pipelines.drones.io import synthetic_flight
    >>> stream = list(synthetic_flight(n=3, seed=0))
    >>> len(stream)
    3
    >>> len(stream[0])
    9
    >>> stream[0]["fault_kind"] is None
    True
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

# ---------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------

def _import_pyulog():
    try:
        from pyulog import ULog
    except ImportError as exc:
        raise ImportError(
            "read_ulog needs pyulog. Install the 'drone' extra: "
            "pip install 'dense-armor[drone]'"
        ) from exc
    return ULog


def _import_pymavlink():
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError(
            "read_mavlink_log needs pymavlink (and pyserial). "
            "Install the 'drone' extra: pip install 'dense-armor[drone]'"
        ) from exc
    return mavutil


# ---------------------------------------------------------------
# ULog reader
# ---------------------------------------------------------------

def _ulog_series(
    ulog, topic_name: str, field_spec: list[tuple[str, int]]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(timestamps_s, values)`` for ``topic_name`` or None."""
    by_name = {ds.name: ds for ds in ulog.data_list}
    if topic_name not in by_name:
        return None
    ds = by_name[topic_name]
    if "timestamp" not in ds.data:
        return None
    ts = np.asarray(ds.data["timestamp"], dtype=np.float64) * 1e-6
    columns = []
    for field_name, n in field_spec:
        if n > 0:
            for i in range(n):
                key = f"{field_name}[{i}]"
                if key in ds.data:
                    columns.append(np.asarray(ds.data[key], dtype=np.float64))
        elif field_name in ds.data:
            columns.append(np.asarray(ds.data[field_name], dtype=np.float64))
    if not columns:
        return None
    return ts, np.stack(columns, axis=-1)


def read_ulog(path) -> Iterator[dict[str, Any]]:
    """Yield one dict per timestamp from a PX4 ULog file.

    The PX4 topics read, and the field each one contributes:

    ============================ ==================================
    topic                        field
    ============================ ==================================
    ``sensor_combined``          ``gyro`` from ``gyro_rad[0..2]``,
                                 ``acc`` from
                                 ``accelerometer_m_s2[0..2]``
    ``vehicle_attitude``         ``gyro`` from ``rollspeed``,
                                 ``pitchspeed``, ``yawspeed`` (used
                                 only when ``sensor_combined`` is
                                 missing)
    ``vehicle_rates_setpoint``   ``commanded`` from ``roll``,
                                 ``pitch``, ``yaw``
    ``vehicle_gps_position``     ``gps_v`` from ``vel_n_m_s``,
                                 ``vel_e_m_s``, ``vel_d_m_s``
    ``actuator_outputs``         ``motor`` from ``output[0..N]``
    ``battery_status``           ``battery_v`` from ``voltage_v``
    ============================ ==================================

    The topic names are PX4 topic names, not from ALFA: ALFA is
    ArduPilot (paper section 2.2, page 2), and to read it use
    :func:`read_mavlink_log` on the telemetry ``.tlog``.

    Args:
        path: filesystem path to the ``.ulg`` file.

    Yields:
        One dict per unique timestamp, with the standard fields.

    Raises:
        ImportError: if ``pyulog`` is not installed.
    """
    ULog = _import_pyulog()
    ulog = ULog(str(path))

    per_field: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    series = _ulog_series(
        ulog, "sensor_combined",
        [("gyro_rad", 3), ("accelerometer_m_s2", 3)],
    )
    if series is not None:
        ts, vals = series
        per_field["gyro"] = (ts, vals[:, :3])
        per_field["acc"] = (ts, vals[:, 3:6])
    else:
        series = _ulog_series(
            ulog, "vehicle_attitude",
            [("rollspeed", 0), ("pitchspeed", 0), ("yawspeed", 0)],
        )
        if series is not None:
            per_field["gyro"] = series

    series = _ulog_series(
        ulog, "vehicle_rates_setpoint",
        [("roll", 0), ("pitch", 0), ("yaw", 0)],
    )
    if series is not None:
        per_field["commanded"] = series

    series = _ulog_series(
        ulog, "vehicle_gps_position",
        [("vel_n_m_s", 0), ("vel_e_m_s", 0), ("vel_d_m_s", 0)],
    )
    if series is not None:
        per_field["gps_v"] = series

    series = _ulog_series(ulog, "actuator_outputs", [("output", 4)])
    if series is not None:
        per_field["motor"] = series

    series = _ulog_series(ulog, "battery_status", [("voltage_v", 0)])
    if series is not None:
        per_field["battery_v"] = series

    yield from _align(per_field)


# ---------------------------------------------------------------
# MAVLink tlog reader
# ---------------------------------------------------------------

def _mav_time(msg) -> float | None:
    """Timestamp in seconds for a MAVLink message, or None.

    Uses ``time_usec`` (microseconds) if positive, else
    ``time_boot_ms`` (milliseconds) if positive, else returns None.
    """
    for attr, scale in (("time_usec", 1e-6), ("time_boot_ms", 1e-3)):
        v = getattr(msg, attr, None)
        if v is not None and v > 0:
            return float(v) * scale
    return None


def read_mavlink_log(path) -> Iterator[dict[str, Any]]:
    """Yield one dict per timestamp from a MAVLink tlog file.

    This is the reader for ALFA telemetry (Keipour, Mousaei & Scherer
    2020, section 3.1, page 3: the telemetry is a MAVLink ``.tlog``).

    The MAVLink messages read, and the field each one contributes:

    ============================ ==================================
    message                      field
    ============================ ==================================
    ``HIGHRES_IMU``              ``gyro`` from ``xgyro``,
                                 ``ygyro``, ``zgyro``; ``acc`` from
                                 ``xacc``, ``yacc``, ``zacc``
    ``ATTITUDE``                 ``gyro`` from ``rollspeed``,
                                 ``pitchspeed``, ``yawspeed`` (used
                                 only when ``HIGHRES_IMU`` is
                                 missing)
    ``ATTITUDE_TARGET``          ``commanded`` from
                                 ``body_roll_rate``,
                                 ``body_pitch_rate``,
                                 ``body_yaw_rate``
    ``GPS_RAW_INT``              ``gps_v`` from ``vel`` (cm/s) and
                                 ``cog`` (cdeg), rotated to NED
    ``SERVO_OUTPUT_RAW``         ``motor`` from ``servo1_raw`` ...
                                 ``servo4_raw``, divided by 1000
    ``BATTERY_STATUS``           ``battery_v`` from ``voltages[0]``
                                 in mV, divided by 1000
    ============================ ==================================

    The message ``ATTITUDE_TARGET`` provides the reference rate; when
    it is absent (some logs do not record the setpoint), ``commanded``
    is None.

    Args:
        path: filesystem path to the ``.tlog`` file.

    Yields:
        One dict per timestamp, with the standard fields.

    Raises:
        ImportError: if ``pymavlink`` (or its ``pyserial`` dependency)
            is not installed.
    """
    mavutil = _import_pymavlink()
    mav = mavutil.mavlogfile(str(path), notimestamps=True)

    msgs = []
    while True:
        msg = mav.recv_match(blocking=False)
        if msg is None:
            break
        msgs.append(msg)

    has_highres = any(m.get_type() == "HIGHRES_IMU" for m in msgs)
    records: list[tuple[float, str, Any]] = []
    last_t: float | None = None
    for msg in msgs:
        t = _mav_time(msg)
        if t is None:
            if last_t is None:
                continue
            t = last_t
        last_t = t
        kind = msg.get_type()
        if kind == "HIGHRES_IMU":
            records.append((t, "gyro", np.array(
                [msg.xgyro, msg.ygyro, msg.zgyro], dtype=np.float64)))
            records.append((t, "acc", np.array(
                [msg.xacc, msg.yacc, msg.zacc], dtype=np.float64)))
        elif kind == "ATTITUDE" and not has_highres:
            records.append((t, "gyro", np.array(
                [msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                dtype=np.float64)))
        elif kind == "ATTITUDE_TARGET":
            records.append((t, "commanded", np.array(
                [msg.body_roll_rate, msg.body_pitch_rate,
                 msg.body_yaw_rate], dtype=np.float64)))
        elif kind == "GPS_RAW_INT":
            vel = float(msg.vel) * 1e-2
            cog = math.radians(float(msg.cog) * 1e-2)
            records.append((t, "gps_v", np.array(
                [vel * math.cos(cog), vel * math.sin(cog), 0.0],
                dtype=np.float64)))
        elif kind == "SERVO_OUTPUT_RAW":
            raw = np.array([msg.servo1_raw, msg.servo2_raw,
                            msg.servo3_raw, msg.servo4_raw],
                           dtype=np.float64)
            records.append((t, "motor", raw / 1000.0))
        elif kind == "BATTERY_STATUS":
            mv = float(msg.voltages[0])
            records.append((t, "battery_v", mv / 1000.0))

    yield from _align_records(records)


# ---------------------------------------------------------------
# Shared alignment
# ---------------------------------------------------------------

def _align(
    per_field: dict[str, tuple[np.ndarray, np.ndarray]]
) -> Iterator[dict[str, Any]]:
    """Merge per-field arrays on the union of their timestamps."""
    records: list[tuple[float, str, Any]] = []
    for field, (ts, vals) in per_field.items():
        for i, t in enumerate(ts):
            records.append((float(t), field, vals[i]))
    yield from _align_records(records)


def _align_records(
    records: list[tuple[float, str, Any]]
) -> Iterator[dict[str, Any]]:
    """Yield one dict per unique timestamp, carrying forward last-seen values."""
    if not records:
        return
    records.sort(key=lambda r: r[0])
    t0 = records[0][0]
    state: dict[str, Any] = {
        "gyro": None, "commanded": None, "acc": None,
        "gps_v": None, "motor": None, "battery_v": float("nan"),
    }
    current_t: float | None = None
    for t, field, value in records:
        if current_t is not None and t != current_t:
            out = {"t": current_t - t0}
            out.update(state)
            yield out
        state[field] = value
        current_t = t
    if current_t is not None:
        out = {"t": current_t - t0}
        out.update(state)
        yield out


# ---------------------------------------------------------------
# Synthetic flight
# ---------------------------------------------------------------

FAULT_AT_FRACTION = 0.6
FAULT_KINDS = (None, "actuator_loss", "gps_ramp", "propeller", "battery_drop")


def synthetic_flight(
    n: int, seed: int, fault: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield ``n`` labelled synthetic flight samples at 100 Hz.

    The trajectory is a hover plus a slow circle in the north-east
    plane: constant yaw rate, tangential velocity, centripetal
    acceleration, all in NED. Small roll and pitch oscillations are
    added to the commanded rate so that the actuator fault has a
    measurable effect on the yaw residual. Gaussian noise is added to
    every field.

    The stream is labelled synthetic: it is generated by this
    function, it is not real flight data. The four fault kinds and
    their papers are listed in the module docstring.

    Args:
        n: number of samples (fixed rate 100 Hz, so ``n`` samples
            span ``n / 100`` seconds). ``n`` must be non-negative.
        seed: RNG seed.
        fault: one of ``None``, ``"actuator_loss"``, ``"gps_ramp"``,
            ``"propeller"``, ``"battery_drop"``. The fault starts at
            sample ``round(FAULT_AT_FRACTION * n)``.

    Yields:
        One dict per sample, with the standard fields plus
        ``fault_started`` (bool) and ``fault_kind`` (the fault name
        from the fault sample onwards, else None).

    Raises:
        ValueError: on a negative ``n`` or an unknown ``fault``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if fault not in FAULT_KINDS:
        raise ValueError(
            f"fault must be one of {FAULT_KINDS}, got {fault!r}"
        )
    rng = np.random.default_rng(seed)
    dt = 0.01
    omega = 0.2
    radius = 5.0
    fault_at = (
        round(FAULT_AT_FRACTION * n) if fault is not None else n + 1
    )
    t_fault = fault_at * dt
    for k in range(n):
        t = k * dt
        v_n = -radius * omega * math.sin(omega * t)
        v_e = radius * omega * math.cos(omega * t)
        a_n = -radius * omega * omega * math.cos(omega * t)
        a_e = -radius * omega * omega * math.sin(omega * t)
        cmd = np.array(
            [
                0.10 * math.sin(2.0 * math.pi * 0.5 * t),
                0.05 * math.sin(2.0 * math.pi * 0.3 * t),
                omega,
            ],
            dtype=np.float64,
        )
        gyro = cmd + rng.normal(0.0, 0.02, 3)
        acc = np.array([a_n, a_e, 9.81]) + rng.normal(0.0, 0.05, 3)
        gps_v = np.array([v_n, v_e, 0.0]) + rng.normal(0.0, 0.05, 3)
        motor = 0.5 + rng.normal(0.0, 0.01, 4)
        battery_v = 12.6 - 0.2 * (t / 60.0) + rng.normal(0.0, 0.005)
        started = k >= fault_at
        if started and fault is not None:
            if fault == "actuator_loss":
                theta = np.array([1.0, 1.0, 0.2])
                gyro = theta * cmd + rng.normal(0.0, 0.02, 3)
            elif fault == "gps_ramp":
                gps_v = gps_v.copy()
                gps_v[0] += 1.0 * (t - t_fault)
            elif fault == "propeller":
                gyro = gyro.copy()
                gyro[0] += 0.5 * math.sin(2.0 * math.pi * 47.0 * (t - t_fault))
            elif fault == "battery_drop":
                battery_v -= 2.0
        yield {
            "t": t,
            "gyro": gyro,
            "commanded": cmd.copy(),
            "acc": acc,
            "gps_v": gps_v,
            "motor": motor,
            "battery_v": float(battery_v),
            "fault_started": bool(started),
            "fault_kind": fault if started else None,
        }
