# Dense-Armor

**Robot-AI Online Learning with Shielding: estimators that learn one sample at a time from
robot and AI data, and a shield that keeps faulty samples out of what they learn.**

Dense-Armor is an online learning library with a shielding layer on top. Everything that
learns does it one sample at a time, at control-loop speed, and everything that is
estimating can be protected from corrupted data without retraining or weight changes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          APPLICATIONS                                       │
│                                                                             │
│   Robot                  Vision                 LLM                AI       │
│   control                frames                 agents             models   │
│   dynamics               patches                tools              + shields │
│   URDF                   vocabulary             JSON                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    SHIELDING (optional, on top)                             │
│                                                                             │
│   Armatura (1D series)   Orca (input+output)   Arbiter (per-point routing)  │
│   Robust filters         Curvature              Streaming check              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ONLINE LEARNING (the core)                               │
│                                                                             │
│   Classifiers            Regressors             Transformers                │
│   Anomaly detectors      Drift detectors        Clustering                  │
│   Trees                  Calibration            Metric learning             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    SIGNALS & STATISTICS (the layer below)                   │
│                                                                             │
│   Signal                 Streams                Datasets                    │
│   Moments                Quantiles              Sketches                    │
│   Preprocessing          Metrics                Evaluation                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ROLES (the foundation)                                   │
│                                                                             │
│   Root → Estimator → AnomalyDetector, Classifier, Regressor, Transformer,   │
│                      DriftDetector                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The learning is the core. The shield sits on top and is optional. Robot, vision, LLM and
the AI models are the applications this core was built for.

## When you already have data, and where the shield goes

Four ways to use the library, from the simplest to the most protected.

**1 — Learn from a clean stream.** You have a model and the data is trustworthy.

```
  sample ──► ESTIMATOR.learn_one(x, y) ──► updated
  sample ──► ESTIMATOR.predict_one(x)  ──► prediction
```

**2 — Learn from a stream that can be corrupted.** Add a shield. Same estimator, one
wrapper.

```
  sample ──► [ SHIELD ] ──► ESTIMATOR.learn_one   (skipped if the sample is flagged)
                  │
                  └──► ESTIMATOR.predict_one     (always runs)
```

**3 — Score each sample without learning.** You only need a number: high means the sample
does not look like the recent past.

```
  sample ──► [ DETECTOR.score_one ] ──► score ──► threshold ──► flag
```

**4 — Score the joint vector, not the single channel.** When several channels move
together and a per-channel detector is blind to the combination.

```
  residual of every joint ──► [ MULTIVARIATE DETECTOR ] ──► score
                                looks at the joint vector
                                as one point in d dimensions
```

Recipe 1 is the core. Recipe 2 adds the shield. Recipes 3 and 4 are the shield used on
its own, for when you only need a detector.

## Install

```bash
pip install dense-armor
```

## The first learned model, in five lines

```python
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares

model = RecursiveLeastSquares(lam=1.0, feature_keys=["one", "q", "qd"])
for t, (x, y) in enumerate(stream):
    model.learn_one({"one": 1.0, "q": q[t], "qd": qd[t]}, tau[t])
```

Each call to `learn_one` updates the model with the new sample. No epochs, no batches.

## The first protected model, in five lines

```python
from dense_armor.roles import AnomalyGate, Protected
from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares

detector = AnomalyGate(HalfSpaceTrees(n_trees=25, depth=8, window=200))
model = Protected(
    model=RecursiveLeastSquares(lam=1.0, feature_keys=["one"]),
    detector=detector,
    fallback=0.0,
)
for t, (x, y) in enumerate(stream):
    model.learn_one({"one": 1.0}, tau[t])
```

Same call as before, but when the detector flags the sample as anomalous, `learn_one` on
the inner model is skipped and the prediction falls back to the last clean value. A
payload fault no longer poisons the model's state.

## What is in here

**Online learning** — the core. Models that learn one sample at a time.

- **[Calibration](learn/calibration.md)** — online Platt scaling: honest probabilities
  from any classifier.
- **[Metric learning](learn/metric_learning.md)** — OASIS, LEGO, POLA, and a k-NN
  classifier that learns its distance.
- **[Online classifiers](learn/online_classifiers.md)** — robot-state classifiers that
  adapt when the data drifts.
- **[Online dynamics](learn/online_dynamics.md)** — learns, sample by sample, the torque
  the URDF model misses.
- **[Clustering](cluster/index.md)** — k-means online, DenStream.
- **[Trees](tree/index.md)** — Hoeffding, Hoeffding Anytime, Hoeffding Adaptive with
  per-node ADWIN, Mondrian forests with uncertainty, stochastic gradient trees.

**Shielding** — the protection that sits between the data and the learning.

- **[Armatura](shields/armatura.md)** — wearable shield for a single 1D series.
- **[Orca](shields/orca.md)** — full input + output shield for an entire model.
- **[Hybrid engine](shields/hybrid_engine.md)** and
  **[adaptive engine](shields/engine.md)** — the two causal engines the shields sit on.

**Detection** — one number per sample, high means the sample does not look like the recent
past.

- **[Arbiter](protect/arbiter.md)** — classifies each point as clean / spike / regime
  against a wide causal reference window, then routes it.
- **[Streaming](anomaly/streaming.md)** — a zero-latency, one-point-at-a-time port of
  Arbiter's causal deviation check, plus native multi-channel wrappers.
- **[Robust filters](anomaly/robust_filters.md)** — Chauvenet, Tukey, Hampel, sigma
  clipping on whole series, plus `pressure_valve`.
- **[Curvature](anomaly/curvature.md)** — a bounded proximity-to-reference score in
  [0, 1).
- **[CUSUM and ARL theory](drift/cusum.md)** — accumulates small, sustained deviations,
  plus a closed-form pre-flight estimate of detection and false-alarm latency.
- **[Drift detectors](drift/detectors.md)** — ADWIN, KSWIN, Page–Hinkley, and the CUSUM
  family.

**Multivariate anomaly detectors** — when several channels move together.

- **[Online Robust Mahalanobis](anomaly/mahalanobis.md)** — distance from a robust median
  under the median covariation matrix.
- **[LODA](anomaly/loda.md)** — rare bins in many random 1D projections.
- **[Half-Space Trees](anomaly/hst.md)** — sparse regions of the joint space.
- **[Online Isolation Forest](anomaly/oiforest.md)** — depth needed to isolate a point.
- **[One-Class SGD (SONAR)](anomaly/ocsvm.md)** — signed distance from a learned kernel
  boundary with a formal error bound.

**Signals and statistics** — the layer the estimators live on.

- **[Streams and datasets](stream/index.md)** — CSV, ROS bag, array, merge by time,
  shuffle; the Casper, DriftStream and SyntheticArm datasets.
- **[Statistics](stats/index.md)** — moments, robust statistics, streaming quantiles
  (t-digest, DDSketch), dependence, sketches.
- **[Preprocessing](preprocessing/index.md)** — scalers per feature and per joint,
  text tokenisation, selection, resampling.
- **[Metrics and evaluation](metrics/index.md)** — classification, regression, intervals,
  rolling, ROC-AUC, prequential, event-based (Tatbul PRF, NAB, detection delay).

**Robot** — the physical side of the library.

- **[Control](control/index.md)** — rate limiter, CBF filter, trajectory generation,
  kinematic controller.
- **[Dynamics](dynamics/index.md)** — rigid-body dynamics from a URDF, passivity-CBF
  controller, full 6-DoF controller, xacro and mimic joints.
- **[Benchmark](benchmarks/real_robot_benchmark.md)** — CASPER on the UR3e.

**Vision** — native image features and patch memory.

- **[Vision](vision/index.md)** — HOG, moments, Lucas–Kanade optical flow, colour
  features, patch descriptors, visual vocabulary.
- **[Patches](vision/patches.md)** — PatchFeatures and PatchMemory (native PatchCore),
  reservoir and coreset.

**Tools** — the parts that do not participate in the anomaly shield.

- **[Toolkit](toolkit/toolkit.md)** — op compiler, memory guard, profiler, logging and
  provenance export, audio / HDF5 / NetCDF I/O.
- **[MCP server](mcp.md)** — estimators exposed as LLM tools.
- **[API reference](api/index.md)** — the base classes every estimator builds on.

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

## Citation

If you use Dense-Armor in a paper, cite the Zenodo release:

```
@software{dense_armor,
  author  = {tatopenn-cell},
  title   = {Dense-Armor: Robot-AI Online Learning with Shielding},
  year    = {2026},
  doi     = {10.5281/zenodo.23265098},
  url     = {https://github.com/tatopenn-cell/Dense-Armor}
}
```
