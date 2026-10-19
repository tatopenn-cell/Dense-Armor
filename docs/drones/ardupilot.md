# ArduPilot against PX4

My drone runs ArduPilot, not PX4: what changes?

The drone pipelines read a MAVLink log with `read_mavlink_log`, and the seven fields it yields are the same on both autopilots. What changes is the message the autopilot emits for each field, and sometimes whether it emits one at all. This page lists, for each field, the MAVLink message and the field it comes from on ArduPilot and on PX4, and shows one runnable example per step.

The names below were checked in the two dialects of `pymavlink`, `pymavlink.dialects.v20.ardupilotmega` and `pymavlink.dialects.v20.common`, by importing them and printing the field names. Koubaa et al. 2019, the MAVLink survey, names only `HEARTBEAT` explicitly (page 3); the other names come from the dialect files, which are the authoritative source.

ALFA is ArduPilot (Keipour, Mousaei and Scherer 2020, section 2, page 2), so the ALFA reader of `pipelines/drones/alfa.py` is the one to reuse, not the ULog one.

## 1. The seven fields

| Field | ArduPilot message and field | PX4 message and field |
|---|---|---|
| `t` | `time_usec` or `time_boot_ms` of any message | `time_usec` or `time_boot_ms` of any message |
| `gyro` | `HIGHRES_IMU.xgyro, ygyro, zgyro`; `ATTITUDE.rollspeed, pitchspeed, yawspeed` when `HIGHRES_IMU` is absent | `HIGHRES_IMU.xgyro, ygyro, zgyro` |
| `acc` | `HIGHRES_IMU.xacc, yacc, zacc` | `HIGHRES_IMU.xacc, yacc, zacc` |
| `gps_v` | `GPS_RAW_INT.vel, cog` | `GPS_RAW_INT.vel, cog` |
| `motor` | `SERVO_OUTPUT_RAW.servo1_raw ... servo4_raw` | `SERVO_OUTPUT_RAW.servo1_raw ... servo4_raw` |
| `battery_v` | `BATTERY_STATUS.voltages[0]` | `BATTERY_STATUS.voltages[0]` |
| `commanded` | `ATTITUDE_TARGET.body_roll_rate, body_pitch_rate, body_yaw_rate` | `ATTITUDE_TARGET.body_roll_rate, body_pitch_rate, body_yaw_rate` |

The two messages that differ are `HIGHRES_IMU` and `ATTITUDE`: ArduPilot logs both (the `HIGHRES_IMU` only when the high-resolution stream is enabled), PX4 logs `HIGHRES_IMU`. When only `ATTITUDE` is present, the gyro comes from it and `acc` stays `None`.

## 2. The gyro and the acceleration

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/imu.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_highres_imu_message(
        1000000, 0.0, 0.0, 9.81, 0.1, 0.2, 0.3,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/imu.tlog"))
print([round(float(v), 3) for v in s["gyro"]],
      [round(float(v), 3) for v in s["acc"]])
```

```
[0.1, 0.2, 0.3] [0.0, 0.0, 9.81]
```

The message writes the gyro in rad/s and the acceleration in m/s^2, and `read_mavlink_log` copies them unchanged. This is the PX4 case: `HIGHRES_IMU` is the message both autopilots can log.

## 3. The gyro from ATTITUDE when HIGHRES_IMU is absent

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/att.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_attitude_message(
        1000, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/att.tlog"))
print([round(float(v), 3) for v in s["gyro"]], s["acc"])
```

```
[0.7, 0.8, 0.9] None
```

ArduPilot logs `ATTITUDE` even when the high-resolution stream is off. `read_mavlink_log` uses it only when `HIGHRES_IMU` is absent, and the acceleration stays `None`: on that log the field cannot be read with the current reader. That is a proposal for `io.py`, not a change: `SCALED_IMU2.xacc, yacc, zacc` is the message ArduPilot logs, and adding it to the reader would fill `acc` without touching the other fields.

## 4. The GPS velocity

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/gps.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_gps_raw_int_message(
        1000000, 3, 0, 0, 0, 0, 0, 300, 9000, 10,
        0, 0, 0, 0, 0, 0))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/gps.tlog"))
print([round(float(v), 3) for v in s["gps_v"]])
```

```
[0.0, 3.0, 0.0]
```

`vel` is in cm/s and `cog` in centidegrees. The reader converts both and rotates the velocity to NED: 300 cm/s with a course of 90 degrees is 3 m/s east.

## 5. The motors

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/servo.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_servo_output_raw_message(
        1000000, 0, 1500, 1600, 1260, 1520, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/servo.tlog"))
print([round(float(v), 3) for v in s["motor"]])
```

```
[1.5, 1.6, 1.26, 1.52]
```

The four `servo*_raw` channels are divided by 1000, so 1500 is 1.5 and 1260 is 1.26. The scale is the one of the synthetic stream, not a normalisation to [0, 1]: a real servo output in [1000, 2000] maps to [1.0, 2.0].

## 6. The battery

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/bat.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_attitude_message(
        1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    mav.send(mavlink.MAVLink_battery_status_message(
        0, 0, 0, 0, [12600, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        0, 0, 0, 0, 0, 0, [0, 0, 0, 0], 0, 0))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/bat.tlog"))
print(round(float(s["battery_v"]), 3))
```

```
12.6
```

`BATTERY_STATUS.voltages[0]` is in mV, divided by 1000. The `ATTITUDE` message comes first because `BATTERY_STATUS` has no timestamp of its own: the reader gives it the last timestamp seen.

## 7. The commanded rate

```python
from pymavlink.dialects.v20 import ardupilotmega as mavlink

with open("/tmp/cmd.tlog", "wb") as f:
    mav = mavlink.MAVLink(f)
    mav.srcSystem = 1
    mav.srcComponent = 1
    mav.send(mavlink.MAVLink_attitude_target_message(
        1000, 0, [1.0, 0.0, 0.0, 0.0], 0.1, 0.2, 0.3, 0.0))

from pipelines.drones.io import read_mavlink_log

s = next(read_mavlink_log("/tmp/cmd.tlog"))
print([round(float(v), 3) for v in s["commanded"]])
```

```
[0.1, 0.2, 0.3]
```

`ATTITUDE_TARGET` carries the reference rate that the autopilot is tracking. On a log without it, `commanded` is `None` and the actuator-fault pipeline cannot compute the residual.

## Details

### References

- Keipour, A., Mousaei, M., Scherer, S. (2020). *ALFA: A Dataset for UAV Fault and Anomaly Detection.* The International Journal of Robotics Research. arXiv:1907.06268v2. Section 2, page 2: ALFA is ArduPilot. Section 3.1, page 3: the processed data are one CSV per topic.
- Koubaa, A. et al. (2019). *Micro Air Vehicle Link (MAVLink) in a Nutshell: A Survey.* IEEE Access. arXiv:1906.10641. Page 3: `HEARTBEAT` is the only message the survey names explicitly.
- The dialect files `pymavlink.dialects.v20.ardupilotmega` and `pymavlink.dialects.v20.common` are the source of every message and field name in the table.

### Choices stated explicitly

- The mapping from field to message is the one of `pipelines/drones/io.py`, not a new one. The table states it, it does not propose a different reader.
- The eight code blocks write their own tlog with `pymavlink` and read it back with `read_mavlink_log`: no file on disk is needed, and each block is self-contained.
- `motor` is divided by 1000, as in `io.py`, so the values are in [1.0, 2.0] on a real log and not in [0, 1].
- `SCALED_IMU2` is named as the proposal for `acc` when `HIGHRES_IMU` is absent, because ArduPilot logs it. It is not added to `io.py` here: B says no change to `io.py`.

### Development diary

The first version of the page read a real ALFA tlog and showed the first samples. The tlog contains `RAW_IMU` and `SCALED_IMU2`, not `HIGHRES_IMU`, and `read_mavlink_log` reads only the latter: the gyro comes from `ATTITUDE`, the acceleration is `None`. The page now says that, and the examples write a tlog with the message each field needs, so the reader is shown on the case it handles.
