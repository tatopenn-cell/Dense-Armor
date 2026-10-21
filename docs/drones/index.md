# Drones

A drone is a flying robot with a small computer onboard, and every page of this
section is a piece of the software that lets it fly and keep learning while it
flies. The pages follow the flow of a mission: a synthetic stream of sensor
samples enters a pipeline, the pipeline predicts a fault or a health state, and
a computer on the edge (a Jetson, a Raspberry Pi) measures the cost of the run.

The first page is [Start here](start_here.md), which walks through one mission
with a drone and a board.

## Pipelines on the drone

Each pipeline reads the sensors of a synthetic flight and predicts one thing
about the drone, one sample at a time.

- **[Actuator fault](actuator_fault.md)** — a stuck or drifting actuator, from
  the residual between the commanded and the measured torque.
- **[Sensor consistency](sensor_consistency.md)** — which sensor of a redundant
  set disagrees with the others.
- **[Propeller health](propeller_health.md)** — a damaged propeller, from the
  vibration on the gyro.
- **[Battery endurance](battery_endurance.md)** — the remaining flight time,
  with the uncertainty of the estimate.
- **[Safety filter](safety_filter.md)** — whether the next commanded move is
  inside the physical limits of the drone.
- **[Link health](link_health.md)** — whether the radio link to the ground
  station is about to fail.
- **[Visual landing](visual_landing.md)** — where the landing pad is, and
  whether the camera is moving.

## The models, batch against online

- **[Static vs online](static_vs_online.md)** — a CART tree fitted on the
  training half and an online Hoeffding tree, on the same fault stream, with
  the accuracy, the time per sample and the memory of each.

## The edge kit and the board

The software that runs on the small computer onboard.

- **[Jetson power](jetson_power.md)** — the power modes of the Orin modules,
  the commands to read and set the mode, and how to measure the time per
  sample in each mode.
- **[ArduPilot against PX4](ardupilot.md)** — the two autopilots of the drone
  ecosystem, the companion computers that fly them, and the libraries of the
  edge kit.
