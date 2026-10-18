# Sensor consistency: is the GPS telling the truth?

A GPS receiver can be spoofed. An attacker transmits fake signals that look like the real ones and slowly pushes the reported position and velocity away from the truth. SemperFi (Sathaye et al. 2021, section 2.2, page 3) describes it as "the attacker-defined trajectory or static position" and "gradually introduces offsets" into the victim's signals. A drone that trusts its GPS follows the spoofed position and flies off course. This page shows the software detector: it compares the GPS velocity to the one a pure inertial integration produces, and raises an alarm when the two diverge.

## The scene

A quadrotor flies a slow circle. The flight controller uses the GPS for velocity and position; the accelerometer integrates the same motion. Under normal conditions the two velocities agree, to within the noise of the accelerometer. When a spoofer pushes the GPS velocity away, the accelerometer does not know about it: it integrates the true acceleration, not the spoofed one. The difference between the GPS velocity and the IMU-integrated one is therefore a signature of the attack.

A single-sample threshold would fire on every noisy sample. A drift detector fires once, when the change is statistically clear. Under a gradual attack, the difference grows continuously, so the detector fires again every time the difference passes a new threshold: the attack is a *sequence* of episodes, not a single one.

## What is in the box

| Tool | Its one job | Example |
|---|---|---|
| Integrated IMU | Estimate the velocity from the accelerometer | `imu_v` after 1 s of integration |
| Residual | `gps_v - imu_v`, three components | `[0.89, 1.02, -0.05]` m/s at the attack |
| CUSUMDriftDetector | Fire when the residual norm changes | 3 episodes, 0.44 / 1.61 / 2.81 s after the attack |

![Three components of gps_v minus imu_v, synthetic stream, with the attack marked](../assets/drones/sensor_consistency_residual.png)

*Synthetic stream, 800 samples at 100 Hz. Black dashed: the attack starts at sample 480 (second 4.80). The north component (the one the attacker pushes) diverges; the east and down components stay near zero.*

## 1. The IMU velocity

The accelerometer measures the linear acceleration in the body frame, which the library's `synthetic_flight` gives in the NED frame. Removing gravity and integrating in time gives the velocity that the drone would have without the GPS.

```python
import numpy as np
from pipelines.drones.io import synthetic_flight
from pipelines.drones.sensor_consistency import GRAVITY_NED

stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
imu_v = np.zeros(3)
prev_t = None
for s in stream:
    t = s["t"]
    if prev_t is not None and t > prev_t:
        imu_v = imu_v + (np.asarray(s["acc"]) - GRAVITY_NED) * (t - prev_t)
    prev_t = t
print(np.round(imu_v, 2))
