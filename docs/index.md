# Dense-Armor

**The wearable AI-safety shield: runtime anomaly damping for any AI input/output.**

[![tests](https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/dense-armor.svg)](https://pypi.org/project/dense-armor/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/backend-JAX-orange.svg)](https://github.com/google/jax)

A sensor that drops readings (`NaN`) or fires an absurd value (`1e6` instead of `1.2`) silently
breaks any downstream pipeline. Dense-Armor sits between the raw data and the model that
consumes it:

```
  corrupted data ──► [ INPUT SHIELD ] ──► AI model ──► [ OUTPUT SHIELD ] ──► clean output
                      purifies vs               │           purifies vs
                      reference                  │           response-to-reference
                      (or robust blind estimate)│           (or self-consistency)
```

No retraining, no weight changes. Runs at inference time on any JAX/NumPy tensor.

## What's in here

- **[`Armatura`](api/armatura.md)** — the wearable shield for 1D series (loss, sensor
  telemetry, token streams): `Armatura.analizza()` decides, point by point, without an
  intermediate state -- a value is either a genuine change (passes) or noise/an isolated spike
  (replaced with the local baseline).
- **[`Orca`](api/orca.md)** — the full input+output shield for an entire model:
  `Orca.protect_and_forward()` purifies the input, runs the model, and checks the response
  isn't itself corrupted.
- **[Hybrid engine](api/hybrid_engine.md)** — the binary-trigger engine behind `Armatura`,
  ported and adapted from [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution)'s
  own verified `healing.py` primitives.
- **[Adaptive engine](api/engine.md)** — `AdaptiveSignalStabilizer`, Orca's Stage 1: a causal,
  `jax.lax.scan`-based recursive filter with a sigmoid damping curve.
- **[Robust filters](api/robust_filters.md)** — four classic, low-cost anomaly detectors
  (Chauvenet's criterion, Tukey's fences, Hampel filter, iterative sigma-clipping) plus
  `pressure_valve`, an orchestrator combining all four via a Lagrange-multiplier-derived
  minimum-variance estimator, with a Jensen-Shannon-modulated dynamic threshold.
- **[Toolkit](api/toolkit.md)** — a second, independent part of the package: an op-compiler,
  memory guard, hardware profiler, logging/provenance export, and audio/HDF5/NetCDF I/O
  helpers. None of it participates in the anomaly shield.

## Real, tested numbers

Adversarial robustness (see [`test/test_boundA-E.py`](https://github.com/tatopenn-cell/Dense-Armor/tree/master/test)
for the actual attack code, not just the reported numbers):

| attack | type | defense |
|---|---|---|
| PGD / BIM / MI-FGSM | gradient, 1000 steps | mitigated, V_max 0.013-0.078 |
| Affine / elastic | geometric, 50k iter | contained, V_inf 0.05-0.14 |
| Fourier broadband | frequency domain, 50k FFT iter | **99.77%+** |
| Carlini-Wagner (L2) | optimization | 78.96% |
| Carlini-Wagner (L∞) | optimization | **64.39%** -- the weakest point found so far |
| DeepFool | optimization | 78.79% |

Honest: **C&W in L∞ norm is the attack that breaks through the most.** Root cause understood,
not yet fixed: it builds a spatially-smooth perturbation across the whole grid in one shot, and
no purely local coherence check (comparing a point to its immediate neighbors) can distinguish
genuinely-smooth structure from adversarially-smooth structure without an external reference.
See the [README](https://github.com/tatopenn-cell/Dense-Armor#-limiti---known) for the full
"known limits" list.
