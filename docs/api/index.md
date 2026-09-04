# API Reference

Two entry points, depending on where you sit in the pipeline:

- **[`Armatura`](armatura.md)** -- wearable shield for a single 1D series (loss, sensor
  telemetry, token stream). Built on the [hybrid engine](hybrid_engine.md).
- **[`Orca`](orca.md)** -- full input+output shield for an entire model. Built on the
  [adaptive engine](engine.md) (`AdaptiveSignalStabilizer`, Stage 1) plus a Collatz-based
  gate (Stage 2). Optional `use_arbiter=True` adds per-point routing (see
  [Arbiter](arbiter.md)) instead of one gate for the whole signal.

Plus a standalone toolkit, independent of both:

- **[Arbiter](arbiter.md)** -- classifies each point as clean/spike/regime against a wide
  causal reference window, then routes it to the right corrector. Also callable on its own
  (`classify_segments`/`route_and_correct`), not only through `Orca(use_arbiter=True)`.
- **[Streaming](streaming.md)** -- a zero-latency, one-point-at-a-time port of Arbiter's
  causal deviation check (not the spike/regime label, which needs lookahead and stays
  batch-only), plus native multi-channel wrappers for real-time and multi-sensor use.
- **[CUSUM + ARL theory](cusum.md)** -- accumulates small, sustained deviations Arbiter's
  instantaneous threshold is structurally blind to, plus a closed-form pre-flight estimate
  (`detectability_report`) of expected detection/false-alarm latency, validated on two real
  physical domains.
- **[Rate limiter](rate_limiter.md)** -- bounds how fast a command can physically change instead
  of classifying whether a deviation is real; a safety bound for real-time command damping (e.g.
  LLM-to-motor), not a signal cleaner. Validated on two real physical domains (SO-101, ALOHA).
- **[CBF filter](cbf_filter.md)** -- bounds WHERE a command can go (never enter a forbidden
  region), complementing the rate limiter's WHEN/HOW FAST guarantee. Same underlying theory
  as SAFER-Splat, without its GPU-bound perception requirement. Validated on two real
  physical domains (SO-101, ALOHA).
- **[Trajectory](trajectory.md)** -- generates the reference the rate limiter and CBF filter
  keep safe in the first place: a closed-form, minimum-jerk-continuous point-to-point path
  for any number of joints. Validated on two real physical domains (SO-101, ALOHA).
- **[Robust filters](robust_filters.md)** -- four classic anomaly detectors (Chauvenet,
  Tukey, Hampel, sigma-clipping) and `pressure_valve`, a Lagrange-multiplier minimum-variance
  orchestrator with a Jensen-Shannon-modulated dynamic threshold.
- **[Curvature](curvature.md)** -- a bounded, saturating proximity-to-reference score in
  `[0,1)`, used internally by `Orca`'s input shield. Takes a `scale` parameter (default 1.0,
  backward compatible) tying its saturation point to real physical units -- see its own docs
  for a real finding on why the unscaled default is a near-binary indicator, not a graded one.
- **[Toolkit](toolkit.md)** -- generic JAX/NumPy pipeline tools that don't participate in the
  anomaly shield: an op-compiler, a memory guard, hardware profiling, logging/provenance
  export, and audio/HDF5/NetCDF I/O helpers.
