# API Reference

Two entry points, depending on where you sit in the pipeline:

- **[`Armatura`](armatura.md)** -- wearable shield for a single 1D series (loss, sensor
  telemetry, token stream). Built on the [hybrid engine](hybrid_engine.md).
- **[`Orca`](orca.md)** -- full input+output shield for an entire model. Built on the
  [adaptive engine](engine.md) (`AdaptiveSignalStabilizer`, Stage 1) plus a Collatz-based
  gate (Stage 2).

Plus a standalone toolkit, independent of both:

- **[Robust filters](robust_filters.md)** -- four classic anomaly detectors (Chauvenet,
  Tukey, Hampel, sigma-clipping) and `pressure_valve`, a Lagrange-multiplier minimum-variance
  orchestrator with a Jensen-Shannon-modulated dynamic threshold.
