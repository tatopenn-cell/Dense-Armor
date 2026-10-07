# Anomaly detection

An anomaly detector gives every sample a score: high means the sample does not look like the recent past. These detectors work one sample at a time, so they run inside a control loop.

- **[Streaming](streaming.md)** — zero-latency deviation scores, single and multi-channel, plus causal Hampel, Tukey, Chauvenet, sigma clipping and online robust Mahalanobis.
- **[Robust filters](robust_filters.md)** — the four classic detectors on whole series, and the pressure-valve orchestrator.
- **[Curvature](curvature.md)** — a bounded proximity-to-reference score in [0, 1).
