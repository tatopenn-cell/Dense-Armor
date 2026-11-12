# API Reference

A starting point for finding the right tool. Say what you have in front of you (an order to check, a stream of returns, a table, a network to train) and this page tells you which part of the library to use. The examples are about finance first, then neural networks.

## The three kinds of learning

| | Online learning | Machine learning | Deep learning |
|---|---|---|---|
| It reads | one sample at a time, at control-loop speed | a whole table, with `fit` and `predict` | batches or a stream, through layers |
| Module | `dense_armor.core.online` | `dense_armor.ml` | `dense_armor.core.deep.neural` |
| It has | classifiers, regressors, trees, forests, bandits, anomaly and drift detectors, clustering, time series | statistics, preprocessing, linear models, trees, forests, ensembles, anomaly and change detection | layers, attention, convolutional and recurrent blocks, training, continual learning, decoder language models |
| Start with | [Online learning](../learn/index.md) | [Machine learning](../ml/index.md) | [Deep learning](../neural/index.md) |

The three kinds of learning are the core. Finance is the application the examples follow, and deep learning is where the library is going. The shields, the robot and the drones are attributes of the same blocks and are listed at the end of this page.

## Start from what you have

- **An order to approve before it leaves** → the [pre-trade checker](../finance/pretrade.md) and the [decision log](../finance/decision_log.md).
- **A stream of returns or prices** → the [drift detectors](../drift/detectors.md) and the [regime page](../finance/regime.md).
- **Transactions to screen for fraud, or a stream with strange points** → the [anomaly detectors](../anomaly/index.md).
- **A whole table** → [machine learning](../ml/index.md), with `fit` and `predict`.
- **A model that must learn one sample at a time** → [online learning](../learn/index.md).
- **A network to train, on a dataset or on a stream** → [deep learning](../neural/index.md).
- **To pick one option at a time and learn from the reward** → the [bandit policies](../bandit/index.md).
- **To ask for a label only on the samples that matter** → the [active-learning strategies](../active/index.md).

## Two finance examples

The two calls that make the first example, from the quickstart of the README:

```python
from dense_armor.core.finance import PreTradeChecker, DecisionLog

checker = PreTradeChecker(
    max_notional=50_000.0, max_size=0.2, price_band=0.001,
    max_orders=3, window_s=10.0, duplicate_window_s=1.0,
)
log = DecisionLog("decisions.jsonl")
d = checker.check("MSFT", "buy", 5.0, 100.0, 100.0, t=0.0)
log.append(t=0.0, payload={"account": "MSFT", "approved": d.approved, "reason": d.reason})
print(log.verify())
```

```
True
```

A change in the mean of daily returns (in percent, synthetic), caught by ADWIN while the stream runs:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

rng = np.random.default_rng(0)
r = np.r_[rng.normal(0.04, 0.6, 400), rng.normal(2.0, 0.6, 400)]
d = ADWIN()
for t, x in enumerate(r):
    d.update(x)
    if d.drift_detected:
        print(t)
        break
```

```
441
```

## What is in here

**Finance** — the first application, and the one the examples follow. Each page opens with the question it answers.

- **[Pre-trade limits](../finance/pretrade.md)** and **[decision log](../finance/decision_log.md)** — the five controls of SEC Rule 15c3-5(c)(1) and a hash-chained log that anyone can verify.
- **[Regime](../finance/regime.md)**, **[drift versus noise](../finance/drift_vs_noise.md)** and **[stress](../finance/stress.md)** — a change in returns, told apart from noise.
- **[Risk](../finance/risk.md)**, **[portfolio](../finance/portfolio.md)**, **[order book](../finance/order_book.md)**, **[execution](../finance/execution.md)** and **[schedules](../finance/schedule.md)** — streaming value-at-risk, weights, order flow, TWAP and VWAP.
- **[Surveillance](../finance/surveillance.md)**, **[fraud](../finance/fraud.md)**, **[intrusion](../finance/intrusion.md)** and **[credit scoring](../finance/credit_scoring.md)** — pipelines on a stream.
- **[All the finance pages](../finance/index.md)**, the **[dashboard](../finance/dashboard.md)** and the **[emergency stop](../finance/emergency.md)**.

**Deep learning** — networks built from layers, with a gradient that can be checked.

- **[Foundations](../neural/index.md)**, **[layers](../neural/layers.md)** and
  **[chains](../neural/chains.md)** — the smallest pieces of a network, and how to join them.
- **[Convolution and pooling](../neural/vision_layers.md)**,
  **[recurrent layers](../neural/sequence_layers.md)** and
  **[attention](../neural/attention.md)** — patterns in an image, memory of the past samples,
  and a step that looks at every other step.
- **[Training](../neural/training.md)** — schedules, mini-batches and the trainer with weight
  decay and early stopping.
- **[Continuous training](../neural/continuous.md)** — online estimators, Hedge
  Backpropagation, a background learner.
- **[Encoders and decoders](../neural/encoder_decoder.md)** — compress a stream into a few
  numbers or integers and expand it back or into a forecast.
- **[Deep](../neural/deep.md)** — the pieces a deep network needs beyond the layers.
- **[Composable attention](../neural/composable_attention.md)** — cross-attention and the
  fusion of two sensors.
- **[Continual learning](../neural/continual_learning.md)** — how to keep a network from
  forgetting the past, and how to measure what it remembers.
- **[Mixture of experts and test-time compute](../neural/moe_testtime.md)** — use only part of
  the parameters on each sample, spend more compute at inference.
- **[Decoder language models](../neural/decoder_lm.md)** — the decoder block, the next-token
  loss and `generate`.
- **[Language-model training](../neural/lm_training.md)** — a 14.2-million-parameter decoder
  trained on a GPU, with its baselines.
- **[Rotary decoder](../neural/rotary_decoder.md)** — RMSNorm, rotary attention and SwiGLU, and
  generation against a static key/value cache.
- **[Stable variance in float32](../stats/stable_variance.md)** — a variance that is exactly
  zero on identical values.

**Machine learning** — the classes of `dense_armor.ml`, which read a whole table at once
with `fit(X, y)`. The [overview](../ml/index.md) has one page per family:
[anomaly detection](../ml/anomaly.md), [Naive Bayes](../ml/bayes.md),
[change detection](../ml/drift.md), [ensembles](../ml/ensemble.md), [forests](../ml/forest.md),
[linear models](../ml/linear.md), [multiclass](../ml/multiclass.md),
[multi-output](../ml/multioutput.md), [nearest neighbours](../ml/neighbors.md),
[preprocessing](../ml/preprocessing.md), [distributions](../ml/proba.md),
[statistics and sketches](../ml/stats.md), [decision trees](../ml/tree.md) and
[vision](../ml/vision.md). The pages in
[machine learning against online learning](../compare/anomaly.md) compare the two on the
same data.

**Online learning** — the core. Models that learn one sample at a time.

- **[Optimizers](../optim/index.md)** — SGD, AdaGrad, Adam, FTRLProximal,
  OnlineNewtonStep, each in object and pure JAX form.
- **[Linear models](../linear/glm.md)** — LinearRegression, LogisticRegression,
  BayesianLinearRegression.
- **[Calibration](../learn/calibration.md)** — online Platt scaling: honest probabilities
  from any classifier.
- **[Metric learning](../learn/metric_learning.md)** — OASIS, LEGO, POLA, and a k-NN
  classifier that learns its distance.
- **[Fraud](../finance/fraud.md)** and **[credit scoring](../finance/credit_scoring.md)** — online trees and ensembles on a stream of transactions and customers.
- **[Clustering](../cluster/index.md)** — k-means online, DenStream.
- **[Trees](../tree/index.md)** — Hoeffding, Hoeffding Anytime, Hoeffding Adaptive with
  per-node ADWIN, Mondrian forests with uncertainty, stochastic gradient trees.

**Detection** — one number per sample, high means the sample does not look like the recent
past.

- **[Arbiter](../protect/arbiter.md)** — classifies each point as clean / spike / regime
  against a wide causal reference window, then routes it.
- **[Streaming](../anomaly/streaming.md)** — a per-sample port of
  Arbiter's causal deviation check, plus native multi-channel wrappers.
- **[Robust filters](../anomaly/robust_filters.md)** — Chauvenet, Tukey, Hampel, sigma
  clipping on whole series, plus `pressure_valve`.
- **[Curvature](../anomaly/curvature.md)** — a bounded proximity-to-reference score in
  [0, 1).
- **[CUSUM and ARL theory](../drift/cusum.md)** — accumulates small, sustained deviations,
  plus a closed-form pre-flight estimate of detection and false-alarm latency.
- **[Drift detectors](../drift/detectors.md)** — ADWIN, KSWIN, Page–Hinkley, and the CUSUM
  family.

**Multivariate anomaly detectors** — when several channels move together.

- **[Online Robust Mahalanobis](../anomaly/mahalanobis.md)** — distance from a robust median
  under the median covariation matrix.
- **[LODA](../anomaly/loda.md)** — rare bins in many random 1D projections.
- **[Half-Space Trees](../anomaly/hst.md)** — sparse regions of the joint space.
- **[Online Isolation Forest](../anomaly/oiforest.md)** — depth needed to isolate a point.
- **[One-Class SGD (SONAR)](../anomaly/ocsvm.md)** — signed distance from a learned kernel
  boundary with a formal error bound.

**Signals and statistics** — the layer the estimators live on.

- **[Streams and datasets](../stream/index.md)** — CSV, ROS bag, array, merge by time,
  shuffle; the Casper, DriftStream and SyntheticArm datasets.
- **[Shared memory for big datasets](../api/data_shared.md)** — let worker processes read a very
  large list without every worker copying it.
- **[Statistics](../stats/index.md)** — moments, robust statistics, streaming quantiles
  (t-digest, DDSketch), dependence, sketches.
- **[Preprocessing](../preprocessing/index.md)** — scalers per feature and per joint,
  text tokenisation, selection, resampling.
- **[Metrics and evaluation](../metrics/index.md)** — classification, regression, intervals,
  rolling, ROC-AUC, prequential, event-based (Tatbul PRF, NAB, detection delay).

**Vision, video and tracking** — from a frame with a timestamp to objects, joints and events.

- **[Vision](../vision/index.md)** — HOG, moments, Lucas–Kanade optical flow, colour
  features, patch descriptors, visual vocabulary.
- **[Frames](../vision/frames.md)** — the frame with a timestamp, the currency of every video.
- **[Features](../vision/features.md)** — what a frame carries: edges and their direction.
- **[Optical flow](../vision/flow.md)** — how far things moved between two frames.
- **[Reduce](../vision/reduce.md)** — fewer numbers, same meaning: random projection, PCA,
  the whole chain.
- **[Motion in time](../vision/motion.md)** — energy, global shift, scene changes.
- **[Tracking](../vision/tracking.md)** — objects without a detector, signals, joints.
- **[Kinematics](../vision/kinematics.md)** — landmarks, kinematics, registration.
- **[Quality](../vision/quality.md)** — frame quality, sync, event cameras, visualisation,
  benchmark.
- **[Patches](../vision/patches.md)** — PatchFeatures and PatchMemory (native PatchCore),
  reservoir and coreset.

**Tools** — the parts that do not participate in the anomaly shield.

- **[Toolkit](../toolkit/toolkit.md)** — op compiler, memory guard, profiler, logging and
  provenance export, audio / HDF5 / NetCDF I/O.
- **[MCP server](../mcp.md)** — estimators exposed as LLM tools.
- **[API reference](../api/index.md)** — the base classes every estimator builds on.

<small>

**Also in the library**, as attributes of the same blocks and not as the main subject:

**Shielding** — the protection that sits between the data and the learning.

- **[Armor](../shields/armor.md)** — wearable shield for a single 1D series.
- **[Orca](../shields/orca.md)** — full input + output shield for an entire model.
- **[Hybrid engine](../shields/hybrid_engine.md)** and
  **[adaptive engine](../shields/engine.md)** — the two causal engines the shields sit on.

**Robot** — the physical side of the library.

- **[Control](../control/index.md)** — rate limiter, CBF filter, trajectory generation,
  kinematic controller.
- **[Dynamics](../dynamics/index.md)** — rigid-body dynamics from a URDF, passivity-CBF
  controller, full 6-DoF controller, xacro and mimic joints.
- **[Benchmark](../benchmarks/real_robot_benchmark.md)** — CASPER on the UR3e.

- **[Online classifiers](../learn/online_classifiers.md)** and **[online dynamics](../learn/online_dynamics.md)** — online learning on a robot: its own state from the joint signals, and the torque a rigid-body model misses.
- **[Drones](../drones/index.md)** — one page per question about a flying robot.

</small>
