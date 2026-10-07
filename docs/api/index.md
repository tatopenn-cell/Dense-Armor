# API Reference

Two entry points, depending on where you sit in the pipeline:

- **[`Armatura`](../shields/armatura.md)** -- wearable shield for a single 1D series (loss, sensor
  telemetry, token stream). Built on the [hybrid engine](../shields/hybrid_engine.md).
- **[`Orca`](../shields/orca.md)** -- full input+output shield for an entire model. Built on the
  [adaptive engine](../shields/engine.md) (`AdaptiveSignalStabilizer`, Stage 1) plus a Collatz-based
  gate (Stage 2). Optional `use_arbiter=True` adds per-point routing (see
  [Arbiter](../protect/arbiter.md)) instead of one gate for the whole signal.

Plus a standalone toolkit, independent of both:

- **[Arbiter](../protect/arbiter.md)** -- classifies each point as clean/spike/regime against a wide
  causal reference window, then routes it to the right corrector. Also callable on its own
  (`classify_segments`/`route_and_correct`), not only through `Orca(use_arbiter=True)`.
- **[Streaming](../anomaly/streaming.md)** -- a zero-latency, one-point-at-a-time port of Arbiter's
  causal deviation check (not the spike/regime label, which needs lookahead and stays
  batch-only), plus native multi-channel wrappers for real-time and multi-sensor use.
- **[CUSUM + ARL theory](../drift/cusum.md)** -- accumulates small, sustained deviations Arbiter's
  instantaneous threshold is structurally blind to, plus a closed-form pre-flight estimate
  (`detectability_report`) of expected detection/false-alarm latency, validated on two real
  physical domains.
- **[Rate limiter](../control/rate_limiter.md)** -- bounds how fast a command can physically change instead
  of classifying whether a deviation is real; a safety bound for real-time command damping (e.g.
  LLM-to-motor), not a signal cleaner. Validated on two real physical domains (SO-101, ALOHA).
- **[CBF filter](../control/cbf_filter.md)** -- bounds WHERE a command can go (never enter a forbidden
  region), complementing the rate limiter's WHEN/HOW FAST guarantee. Same underlying theory
  as SAFER-Splat, without its GPU-bound perception requirement. Validated on two real
  physical domains (SO-101, ALOHA).
- **[Trajectory](../control/trajectory.md)** -- generates the reference the rate limiter and CBF filter
  keep safe in the first place: a closed-form, minimum-jerk-continuous point-to-point path
  for any number of joints. Validated on two real physical domains (SO-101, ALOHA).
- **[Kinematic controller](../control/kinematic_controller.md)** -- turns a reference from Trajectory
  into an actual velocity command: closed-form feedforward-plus-proportional tracking with
  an exact exponential convergence guarantee. Validated on two real physical domains
  (SO-101, ALOHA), chained with Trajectory.
- **[Robust filters](../anomaly/robust_filters.md)** -- four classic anomaly detectors (Chauvenet,
  Tukey, Hampel, sigma-clipping) and `pressure_valve`, a Lagrange-multiplier minimum-variance
  orchestrator with a Jensen-Shannon-modulated dynamic threshold.
- **[Curvature](../anomaly/curvature.md)** -- a bounded, saturating proximity-to-reference score in
  `[0,1)`, used internally by `Orca`'s input shield. Takes a `scale` parameter (default 1.0,
  backward compatible) tying its saturation point to real physical units -- see its own docs
  for a real finding on why the unscaled default is a near-binary indicator, not a graded one.
- **[Toolkit](../toolkit/toolkit.md)** -- generic JAX/NumPy pipeline tools that don't participate in the
  anomaly shield: an op-compiler, a memory guard, hardware profiling, logging/provenance
  export, and audio/HDF5/NetCDF I/O helpers.

## The library by section

The same pages, grouped the way the package is organised (`dense_armor.<section>`):

| Section | What it is for | Pages |
|---|---|---|
| [Shields](../shields/index.md) | one object that cleans a whole series or a whole model | Armatura, Orca, their engines |
| [Anomaly detection](../anomaly/index.md) | a score per sample: high means it does not look like the recent past | Streaming, Robust filters, Curvature |
| [Drift detection](../drift/index.md) | one alarm when a slow, sustained change is real | CUSUM and ARL theory |
| [Protection](../protect/index.md) | what to do with a wrong-looking sample: pass, repair, or route it | Arbiter (batch and streaming), healing |
| [Online learning](../learn/index.md) | models that learn one sample at a time while the robot runs | Calibration, Metric learning, Online classifiers, Online dynamics |
| [Control](../control/index.md) | commands: where a robot may go, how fast, along which path | Rate limiter, CBF filter, Trajectory, Kinematic controller |
| [Robot dynamics](../dynamics/index.md) | the physical model of a real robot from its URDF, and controllers on it | Rigid-body dynamics, Passivity + CBF controllers, Xacro, Mimic joints |
| [Benchmarks](../benchmarks/index.md) | the library on recordings from real robots | Real robot arm (CASPER, UR3e) |
| [Toolkit](../toolkit/index.md) | generic tools the shields use | Toolkit |

Pages added since this list was first written, not described above:

- **[Calibration](../learn/calibration.md)** -- online Platt scaling: corrects the probabilities
  of any classifier one sample at a time.
- **[Metric learning](../learn/metric_learning.md)** -- OASIS, LEGO and POLA, and a k-NN
  classifier that learns its distance online.
- **[Online classifiers](../learn/online_classifiers.md)** -- robot-state classifiers that adapt
  when the data drift.
- **[Online dynamics](../learn/online_dynamics.md)** -- learns, sample by sample, the torque the
  URDF model misses (payload, friction).
- **[Rigid-body dynamics](../dynamics/urdf_dynamics.md)** and the
  **[passivity + singularity-CBF controllers](../dynamics/passivity_cbf_controller.md)** -- the
  real robot model from its URDF, and task-space controllers built on it.
- **[Real robot benchmark](../benchmarks/real_robot_benchmark.md)** -- the anomaly stack on 24.5 h
  of a real Universal Robots UR3e arm.

