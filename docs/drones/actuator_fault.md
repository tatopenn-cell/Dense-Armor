# Actuator fault: which part of the drone broke, and how many seconds after?

A quadrotor flies a hover plus a slow circle. At some moment one actuator loses part of its authority: the motor still spins, but the commanded rate and the measured rate stop agreeing. The pilot on the ground sees nothing until the drone drifts; the onboard computer sees the residual at once. This page shows the residual, the detector that turns it into a single alarm episode, and the second detector that gives a score on the same three numbers.

## The scene

A quadrotor with four motors, each one commanded by the flight controller. The controller asks the body to rotate at a certain rate on each axis, the gyroscope measures the rate the body actually has, and the difference is the residual. When one motor loses 80% of its thrust, the body still turns, but it turns slower than commanded on the axis that motor controls. The residual on that axis is no longer noise: it has a bias.

A single-sample threshold would fire on every sample after the fault. A drift detector fires once, when the change becomes statistically clear, and then stays quiet. What an operator wants is not a score, it is a count: how many episodes, and how long after the fault the first one arrived. That is what this pipeline produces, with the metric names that the ALFA dataset uses.

## What is in the box

| Tool | Its one job | Example |
|---|---|---|
| Residual | commanded minus gyro, three components | [0.001, 0.002, 0.040] rad/s after the fault |
| CUSUMDriftDetector | Fire once, when the mean of the residual norm changes | One episode, 20 ms after the fault |
| OnlineRobustMahalanobis | Score the three residuals together | Score 11.3, above the threshold 7 |

![Residual of the commanded minus the measured rate, synthetic stream, with the fault and the detected episode marked](../assets/drones/actuator_fault_residual.png)

*Synthetic stream, 1000 samples at 100 Hz. Black dashed: the fault. Red cross: the single episode the detector raised, 20 ms after the fault.*

## 1. The residual

The flight controller sends a rate setpoint on each body axis; the gyroscope measures the rate the body has. The residual is their difference, one three-vector per sample.

```python
from pipelines.drones.io import synthetic_flight
from pipelines.drones.actuator_fault import _residual

stream = list(synthetic_flight(n=1000, seed=0, fault="actuator_loss"))
r = _residual(stream[600])
print(round(float(r[2]), 3))
