# Robust filters (standalone detectors)

Four classic, low-cost anomaly detectors -- no dynamic model, no state, plain arithmetic on a
centered local window (offline/batch cleanup, not the causal real-time loop
[`core/hybrid_engine.py`](hybrid_engine.md) uses). `pressure_valve` combines all four via a
minimum-variance estimator (derived with a Lagrange multiplier) instead of a vote, plus a
Jensen-Shannon-modulated dynamic threshold.

::: dense_armor.utility.robust_filters
