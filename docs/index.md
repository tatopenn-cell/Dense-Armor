# Dense-Armor

**The wearable AI-safety shield: runtime anomaly damping for any AI input/output.**

A sensor that drops readings, or fires an absurd value, silently breaks any pipeline
that consumes it. Dense-Armor sits between the raw data and the model: it purifies the
input, the model runs on clean data, and the output is checked before it is used.

```
  corrupted data ──► [ INPUT SHIELD ] ──► AI model ──► [ OUTPUT SHIELD ] ──► clean output
                      purifies vs                │          purifies vs
                      reference                  │          response-to-reference
                      (or robust blind estimate) │          (or self-consistency)
```

No retraining, no weight changes. It runs at inference time on any JAX / NumPy tensor.

## The first protected signal, in five lines

```python
import jax
jax.config.update("jax_enable_x64", True)
from dense_armor import Armatura

a = Armatura(livello_ia=0.0)
clean, K, anomalies = a.analizza([1.2, 1.3, 9999, 1.25, float("nan"), 1.3])
```

`clean` is the same six-value series with the two broken samples replaced by the local
baseline; `anomalies` is the list of indices that were replaced.

## What is in here

- **[Armatura](shields/armatura.md)** — the wearable shield for a single 1D series
  (loss curve, sensor telemetry, token stream). `Armatura.analizza()` decides, point by
  point, without an intermediate state: a value is either a genuine change (passes) or
  noise / an isolated spike (replaced with the local baseline).
- **[Orca](shields/orca.md)** — the full input+output shield for an entire model.
  `Orca.protect_and_forward()` purifies the input, runs the model, and checks the
  response isn't itself corrupted. Optional `use_arbiter=True` routes each point to the
  right corrector (see [Arbiter](protect/arbiter.md)).
- **[Arbiter](protect/arbiter.md)** — classifies each point as clean / spike / regime
  against a wide causal reference window, then routes it.
- **[Hybrid engine](shields/hybrid_engine.md)** — the binary-trigger engine behind
  `Armatura`.
- **[Adaptive engine](shields/engine.md)** — `AdaptiveSignalStabilizer`, Orca's Stage 1:
  a causal, `jax.lax.scan`-based recursive filter with a sigmoid damping curve.
- **[Robust filters](anomaly/robust_filters.md)** — four classic anomaly detectors
  (Chauvenet, Tukey, Hampel, sigma clipping) plus `pressure_valve`, an orchestrator
  combining all four via a Lagrange-multiplier minimum-variance estimator, with a
  Jensen–Shannon-modulated dynamic threshold.
- **[Toolkit](toolkit/toolkit.md)** — an op-compiler, memory guard, hardware profiler,
  logging / provenance export, and audio / HDF5 / NetCDF I/O helpers. None of it
  participates in the anomaly shield.

## Honest results

Adversarial robustness (see `test/test_boundA-E.py` for the actual attack code, not just
the reported numbers):

| attack | type | defense |
|---|---|---|
| PGD / BIM / MI-FGSM | gradient, 1000 steps | mitigated, V_max 0.013–0.078 |
| Affine / elastic | geometric, 50k iter | contained, V_inf 0.05–0.14 |
| Fourier broadband | frequency domain, 50k FFT iter | **99.77 %+** |
| Carlini-Wagner (L2) | optimization | 78.96 % |
| Carlini-Wagner (L∞) | optimization | **64.39 %** — weakest point found so far |
| DeepFool | optimization | 78.79 % |

**C&W in L∞ norm is the attack that breaks through the most.** Root cause understood, not
yet fixed: it builds a spatially-smooth perturbation across the whole grid in one shot,
and no purely local coherence check (comparing a point to its immediate neighbours) can
distinguish genuinely-smooth structure from adversarially-smooth structure without an
external reference. See the
[README](https://github.com/tatopenn-cell/Dense-Armor#-limiti---known) for the full
"known limits" list.
