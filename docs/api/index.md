# API Reference

Dense-Armor has two kinds of entry point. A **shield** cleans a whole series or a whole
model in one call. A **single-purpose module** does one thing (score a sample, detect a
drift, bound a command) and composes with the others.

Start from what you have in front of you:

- You have a **1D series** (a loss curve, one sensor channel, a token stream) → `Armatura`.
- You have a **model and a corrupted input** → `Orca`.
- You need **one score per sample** → the anomaly detectors.
- You need **one alarm for a slow drift** → CUSUM.
- You need **a safe command for a real robot** → the control modules.
- You need **the physical model of a robot** → the dynamics modules.

## The two shields

- **[`Armatura`](../shields/armatura.md)** — wearable shield for a single 1D series. Built
  on the [hybrid engine](../shields/hybrid_engine.md).
- **[`Orca`](../shields/orca.md)** — full input + output shield for an entire model. Built
  on the [adaptive engine](../shields/engine.md) (`AdaptiveSignalStabilizer`, Stage 1)
  plus a Collatz-based gate (Stage 2). Optional `use_arbiter=True` adds per-point routing
  (see [Arbiter](../protect/arbiter.md)) instead of one gate for the whole signal.

## Per-sample scoring and routing

- **[Arbiter](../protect/arbiter.md)** — classifies each point as clean / spike / regime
  against a wide causal reference window, then routes it. Also callable on its own
  (`classify_segments` / `route_and_correct`), not only through `Orca(use_arbiter=True)`.
- **[Streaming](../anomaly/streaming.md)** — a zero-latency, one-point-at-a-time port of
  Arbiter's causal deviation check (not the spike/regime label, which needs lookahead and
  stays batch-only), plus native multi-channel wrappers.
- **[Robust filters](../anomaly/robust_filters.md)** — the same four classic detectors
  (Chauvenet, Tukey, Hampel, sigma clipping) on whole series, and `pressure_valve`.
- **[Curvature](../anomaly/curvature.md)** — a bounded proximity-to-reference score in
  `[0, 1)`, used internally by `Orca`'s input shield.
- **[CUSUM and ARL theory](../drift/cusum.md)** — accumulates small, sustained deviations
  Arbiter's instantaneous threshold is structurally blind to, plus a closed-form
  pre-flight estimate (`detectability_report`) of expected detection / false-alarm
  latency.

## Control

Each page bounds a different aspect of a command, on the same single-integrator joint
model, and they compose in this order:

- **[Trajectory](../control/trajectory.md)** — generates a smooth, minimum-jerk
  point-to-point path (the reference to follow).
- **[Kinematic controller](../control/kinematic_controller.md)** — turns that reference
  into a joint velocity command with an exact exponential convergence guarantee.
- **[Rate limiter](../control/rate_limiter.md)** — bounds how fast the command can
  physically change (WHEN / HOW FAST).
- **[CBF filter](../control/cbf_filter.md)** — bounds where the command can go (WHERE).

## Robot dynamics

- **[Rigid-body dynamics](../dynamics/urdf_dynamics.md)** — mass matrix, gravity, forward
  dynamics, and kinematics of any link, from the robot's URDF file.
- **[Passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md)** —
  task-space control that stays away from kinematic singularities.
- **[Full 6-DoF controller](../dynamics/six_dof_pbc_cbf_controller.md)** — the same, for
  position *and* orientation.
- **[Xacro files](../dynamics/xacro_support.md)** and **[mimic joints](../dynamics/mimic_joints.md)** —
  loading robots described with macros, and modelling coupled joints.

## Online learning

- **[Calibration](../learn/calibration.md)** — online Platt scaling: honest probabilities
  from any classifier.
- **[Metric learning](../learn/metric_learning.md)** — OASIS, LEGO and POLA, and a k-NN
  classifier that learns its distance online.
- **[Online classifiers](../learn/online_classifiers.md)** — robot-state classifiers that
  adapt when the data drifts.
- **[Online dynamics](../learn/online_dynamics.md)** — learns, sample by sample, the
  torque the URDF model misses.

## The library by section

The same pages, grouped the way the package is organised (`dense_armor.<section>`):

| Section | What it is for | Pages |
|---|---|---|
| [Shields](../shields/index.md) | one object that cleans a whole series or a whole model | Armatura, Orca, their engines |
| [Anomaly detection](../anomaly/index.md) | a score per sample: high means it does not look like the recent past | Streaming, Robust filters, Curvature |
| [Drift detection](../drift/index.md) | one alarm when a slow, sustained change is real | CUSUM and ARL theory |
| [Protection](../protect/index.md) | what to do with a wrong-looking sample: pass, repair, or route | Arbiter (batch and streaming) |
| [Online learning](../learn/index.md) | models that learn one sample at a time while the robot runs | Calibration, Metric learning, Classifiers, Dynamics |
| [Control](../control/index.md) | commands: where a robot may go, how fast, along which path | Rate limiter, CBF filter, Trajectory, Kinematic controller |
| [Robot dynamics](../dynamics/index.md) | the physical model of a real robot from its URDF | Rigid-body dynamics, Controllers, Xacro, Mimic joints |
| [Benchmarks](../benchmarks/index.md) | the library on recordings from real robots | Real robot arm (CASPER, UR3e) |
| [Toolkit](../toolkit/index.md) | generic tools the shields use | Toolkit |
