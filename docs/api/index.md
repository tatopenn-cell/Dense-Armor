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
- **[Robust filters](robust_filters.md)** -- four classic anomaly detectors (Chauvenet,
  Tukey, Hampel, sigma-clipping) and `pressure_valve`, a Lagrange-multiplier minimum-variance
  orchestrator with a Jensen-Shannon-modulated dynamic threshold.
- **[Toolkit](toolkit.md)** -- generic JAX/NumPy pipeline tools that don't participate in the
  anomaly shield: an op-compiler, a memory guard, hardware profiling, logging/provenance
  export, and audio/HDF5/NetCDF I/O helpers.
