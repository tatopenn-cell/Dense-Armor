<img width="1024" height="1024" alt="d20" src="https://github.com/user-attachments/assets/6aa8e84c-809b-4dee-9f56-76605f98014d" />

# Dense-Armor

Dense-Armor is a library that learns from streams of numbers (and from images and text) in three ways: one sample at a time, on a whole table at once, and with deep networks. It can spot strange readings, notice slow changes, protect against bad data and keep robot commands safe.

**A library for three kinds of learning: online learning on robot, vision, and AI streams; machine learning on a whole dataset; and deep learning.**

## The three kinds of learning

| | Online learning | Machine learning | Deep learning |
|---|---|---|---|
| It reads | one sample at a time, at control-loop speed | a whole table, with `fit` and `predict` | batches or a stream, through layers |
| Module | `dense_armor.core.online` | `dense_armor.ml` | `dense_armor.core.deep.neural` |
| It has | classifiers, regressors, trees, forests, bandits, anomaly and drift detectors, clustering, time series | statistics, preprocessing, linear models, trees, forests, ensembles, anomaly and change detection | layers, attention, convolutional and recurrent blocks, training, continual learning, decoder language models |
| Start with | [Online learning](learn/index.md) | [Machine learning](ml/index.md) | [Deep learning](neural/index.md) |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 APPLICATIONS                                │
│                                                                             │
│ Robot control, dynamics, URDF    Vision    LLM agents, tools    AI models   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           THREE KINDS OF LEARNING                           │
│                                                                             │
│ ONLINE                   MACHINE LEARNING         DEEP LEARNING             │
│ one sample at a time     fit / predict on a       layers, attention,        │
│ classifiers, regressors  whole table              training, language        │
│ trees, anomaly, drift   statistics, linear,      models, continual         │
│ detectors, clustering    trees, ensembles         learning                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                PROTECTION (optional, around any online model)               │
│                                                                             │
│ Armor (1D series)    Orca (input + output)    Arbiter (per-point routing)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                    SIGNALS & STATISTICS (the layer below)                   │
│                                                                             │
│ Signal   Streams   Datasets   Moments   Quantiles   Sketches   Metrics      │
├─────────────────────────────────────────────────────────────────────────────┤
│                            ROLES (the foundation)                           │
│                                                                             │
│ Root > Estimator > AnomalyDetector, Classifier, Regressor, Transformer,     │
│                    DriftDetector                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

The three kinds of learning are the core. The shield (Armor, Orca, Arbiter) is optional protection around an online model. Robot, vision, LLM and the AI models are the applications this core was built for.

## Online learning in practice: four ways to use it, and where the shield goes

Four ways to use the online part, from the simplest to the most protected.

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

Recipe 1 is the online core. Recipe 2 adds the shield. Recipes 3 and 4 are the shield used on
its own, for when you only need a detector.

## Install

```bash
pip install dense-armor
```

## The first learned model, in five lines

```python
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

model = RecursiveLeastSquares(lam=1.0, feature_keys=["one", "q", "qd"])
for t, (x, y) in enumerate(stream):
    model.learn_one({"one": 1.0, "q": q[t], "qd": qd[t]}, tau[t])
```

Each call to `learn_one` updates the model with the new sample. No epochs, no batches.

## The first protected model, in five lines

```python
from dense_armor.roles import AnomalyGate, Protected
from dense_armor.core.online.anomaly.hst import HalfSpaceTrees
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

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

**Deep learning** — networks built from layers, with a gradient that can be checked.

- **[Foundations](neural/index.md)**, **[layers](neural/layers.md)** and
  **[chains](neural/chains.md)** — the smallest pieces of a network, and how to join them.
- **[Convolution and pooling](neural/vision_layers.md)**,
  **[recurrent layers](neural/sequence_layers.md)** and
  **[attention](neural/attention.md)** — patterns in an image, memory of the past samples,
  and a step that looks at every other step.
- **[Training](neural/training.md)** — schedules, mini-batches and the trainer with weight
  decay and early stopping.
- **[Continuous training](neural/continuous.md)** — online estimators, Hedge
  Backpropagation, a background learner.
- **[Encoders and decoders](neural/encoder_decoder.md)** — compress a stream into a few
  numbers or integers and expand it back or into a forecast.
- **[Deep](neural/deep.md)** — the pieces a deep network needs beyond the layers.
- **[Composable attention](neural/composable_attention.md)** — cross-attention and the
  fusion of two sensors.
- **[Continual learning](neural/continual_learning.md)** — how to keep a network from
  forgetting the past, and how to measure what it remembers.
- **[Mixture of experts and test-time compute](neural/moe_testtime.md)** — use only part of
  the parameters on each sample, spend more compute at inference.
- **[Decoder language models](neural/decoder_lm.md)** — the decoder block, the next-token
  loss and `generate`.
- **[Language-model training](neural/lm_training.md)** — a 14.2-million-parameter decoder
  trained on a GPU, with its baselines.
- **[Rotary decoder](neural/rotary_decoder.md)** — RMSNorm, rotary attention and SwiGLU, and
  generation against a static key/value cache.
- **[Stable variance in float32](stats/stable_variance.md)** — a variance that is exactly
  zero on identical values.

**Machine learning** — the classes of `dense_armor.ml`, which read a whole table at once
with `fit(X, y)`. The [overview](ml/index.md) has one page per family:
[anomaly detection](ml/anomaly.md), [Naive Bayes](ml/bayes.md),
[change detection](ml/drift.md), [ensembles](ml/ensemble.md), [forests](ml/forest.md),
[linear models](ml/linear.md), [multiclass](ml/multiclass.md),
[multi-output](ml/multioutput.md), [nearest neighbours](ml/neighbors.md),
[preprocessing](ml/preprocessing.md), [distributions](ml/proba.md),
[statistics and sketches](ml/stats.md), [decision trees](ml/tree.md) and
[vision](ml/vision.md). The pages in
[machine learning against online learning](compare/anomaly.md) compare the two on the
same data.

**Online learning** — the core. Models that learn one sample at a time.

- **[Optimizers](optim/index.md)** — SGD, AdaGrad, Adam, FTRLProximal,
  OnlineNewtonStep, each in object and pure JAX form.
- **[Linear models](linear/glm.md)** — LinearRegression, LogisticRegression,
  BayesianLinearRegression, PayloadEstimator.
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

- **[Armor](shields/armor.md)** — wearable shield for a single 1D series.
- **[Orca](shields/orca.md)** — full input + output shield for an entire model.
- **[Hybrid engine](shields/hybrid_engine.md)** and
  **[adaptive engine](shields/engine.md)** — the two causal engines the shields sit on.

**Detection** — one number per sample, high means the sample does not look like the recent
past.

- **[Arbiter](protect/arbiter.md)** — classifies each point as clean / spike / regime
  against a wide causal reference window, then routes it.
- **[Streaming](anomaly/streaming.md)** — a per-sample port of
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
- **[Shared memory for big datasets](api/data_shared.md)** — let worker processes read a very
  large list without every worker copying it.
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

**Vision, video and tracking** — from a frame with a timestamp to objects, joints and events.

- **[Vision](vision/index.md)** — HOG, moments, Lucas–Kanade optical flow, colour
  features, patch descriptors, visual vocabulary.
- **[Frames](vision/frames.md)** — the frame with a timestamp, the currency of every video.
- **[Features](vision/features.md)** — what a frame carries: edges and their direction.
- **[Optical flow](vision/flow.md)** — how far things moved between two frames.
- **[Reduce](vision/reduce.md)** — fewer numbers, same meaning: random projection, PCA,
  the whole chain.
- **[Motion in time](vision/motion.md)** — energy, global shift, scene changes.
- **[Tracking](vision/tracking.md)** — objects without a detector, signals, joints.
- **[Kinematics](vision/kinematics.md)** — landmarks, kinematics, registration.
- **[Quality](vision/quality.md)** — frame quality, sync, event cameras, visualisation,
  benchmark.
- **[Patches](vision/patches.md)** — PatchFeatures and PatchMemory (native PatchCore),
  reservoir and coreset.

**Applications** — pipelines that chain the blocks above on a synthetic stream.

- **[Finance](finance/index.md)** — streaming pipelines for risk, execution and fraud
  detection.
- **[Drones](drones/index.md)** — one page per question about a flying robot: actuator
  fault, sensor consistency, battery endurance, link health, visual landing and others.

**Tools** — the parts that do not participate in the anomaly shield.

- **[Toolkit](toolkit/toolkit.md)** — op compiler, memory guard, profiler, logging and
  provenance export, audio / HDF5 / NetCDF I/O.
- **[MCP server](mcp.md)** — estimators exposed as LLM tools.
- **[API reference](api/index.md)** — the base classes every estimator builds on.

## Honest results

Adversarial robustness (see `test/experiments/test_boundA.py` to `test_boundE.py` for the actual attack code, not just
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
[known limits](limits.md) page for the full list.

## Citation

If you use Dense-Armor in a paper, cite the Zenodo release:

```
@software{dense_armor,
  author  = {Pennacchio, Salvatore},
  title   = {Dense-Armor: Low-Level Learning toolkit},
  year    = {2026},
  doi     = {10.5281/zenodo.24030156},
  url     = {https://github.com/tatopenn-cell/Dense-Armor}
}
```
