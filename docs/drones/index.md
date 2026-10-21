# Drones

A drone is a flying robot with a small computer onboard, and every page of this
section is a guide to one question about the drone. The pages are independent:
each one takes a synthetic stream, runs one pipeline on it, and says what the
pipeline predicts and how well. The software that runs on the small computer
onboard is at the bottom, with the two autopilots of the ecosystem.

The first page is [Start here](start_here.md), which walks through one mission
with a drone and a board.

## Pipelines on the drone

Each pipeline reads one kind of signal and answers one question about the drone,
one sample at a time.

- **[Actuator fault](actuator_fault.md)** — a motor that loses part of its
  authority, from the residual between the commanded and the measured body rate
  on the gyro.
- **[Sensor consistency](sensor_consistency.md)** — whether the GPS is telling
  the truth, by comparing the velocity it reports with the one the accelerometer
  integrates, under a spoofing attack.
- **[Propeller health](propeller_health.md)** — a damaged propeller, from the
  vibration the gyro picks up in the band set by the motor speed.
- **[Battery endurance](battery_endurance.md)** — the remaining flight time,
  with a conformal interval that covers the uncertainty of the estimate.
- **[Safety filter](safety_filter.md)** — whether the next move is safe, from
  the distance to an obstacle measured by a lidar and the speed of the drone.
- **[Link health](link_health.md)** — the trend of an MQTT radio link, from the
  inter-arrival time of the packets and the RSSI measured by the radio.
- **[Visual landing](visual_landing.md)** — where the landing pad is in the
  camera frame, and whether the camera itself is moving.

## The models, batch against online

- **[Static vs online](static_vs_online.md)** — a CART tree fitted on the
  training half and an online Hoeffding tree, on the same fault stream, with
  the accuracy, the time per sample and the memory of each.

## The edge kit and the board

The software that runs on the small computer onboard.

- **[Jetson power](jetson_power.md)** — the power modes of the Orin modules,
  the commands to read and set the mode, and how to measure the time per
  sample in each mode.
- **[ArduPilot against PX4](ardupilot.md)** — what changes on the two
  autopilots of the drone ecosystem, the messages each one emits and the
  fields the pipelines read.
