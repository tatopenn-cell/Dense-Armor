<p align="center">
<img width="1800" height="400" alt="banner_nuovo" src="https://github.com/user-attachments/assets/9557ee09-fc5d-4ae9-8e1a-684017013f6e" />
</p>

<p align="center">
  <img alt="tests" src="https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/tatopenn-cell/Dense-Armor"><img alt="codecov" src="https://codecov.io/gh/tatopenn-cell/Dense-Armor/branch/master/graph/badge.svg"></a>
  <a href="https://pypi.org/project/dense-armor"><img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.23584384"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23584384.svg"></a>
  <img alt="license" src="https://img.shields.io/badge/license-AGPL_3.0-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="vision" src="https://img.shields.io/badge/vision-native-9cf.svg">
  <img alt="robot control" src="https://img.shields.io/badge/robot%20control-rate%20limit%20%2B%20CBF-9cf.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<h3 align="center">Robot-AI Online Shielding &amp; Learning</h3>

<p align="center"><strong>A library for online learning on robot, vision, and AI streams, one sample at a time: streaming statistics and protection, drift and anomaly detection, safe robot commands, deep learning with continual learning and memory consolidation, and robust time-series models.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

---

## Who is this for

Python developers working on systems where data arrives one sample at a time and the model has to learn or stay stable *while it runs*: robot control loops, sensor pipelines, agent runtimes, any streaming workload. A fixed dataset works too: the [Machine learning](https://tatopenn-cell.github.io/Dense-Armor/ml/) classes of `dense_armor.ml` read the whole table at once.

## Use cases

The primitives are domain-agnostic: anything that ingests data one sample at a time and has to react when the stream changes. It has been measured on a robot arm (CASPER on a UR3e), on stock and crypto prices, card transactions, network flows and a telemarketing log (the [finance pages](https://tatopenn-cell.github.io/Dense-Armor/finance/)), and on synthetic streams with known drift; the layers above and below the model are the same shape everywhere.

| domain | what Dense-Armor brings |
|---|---|
| Robot control loops | bounded commands (rate limits, CBF), drift and anomaly on joint streams, calibrated intervals |
| Deployed ML models | input drift detection, runtime shielding of `NaN` and spikes, no retraining required |
| Antivirus / security | event stream statistics, behavioural drift, anomaly scoring one event at a time |
| Fraud detection | per-transaction scoring, drift on the user profile, streaming statistics |
| Search / ranking | monitoring layer around the ranker: drift on user behaviour, streaming metrics |
| Live monitoring | Prometheus-style metrics with drift detection, sketches, quantiles in constant memory |
| IoT / sensor fusion | the same shape as a robot joint: noisy multichannel input, drift, anomalies, shields |

## What it expects from you

No specific model. Any estimator that follows the same contract: `learn_one(x, y)` updates the model from one sample, `predict_one(x)` returns a prediction, `transform_one(x)` returns a transform or a residual, `score_one(x)` returns an anomaly score (high = anomalous), `update(x)` advances a drift detector. Inputs and outputs are dicts of numpy-compatible scalars or arrays. Runs on CPU or GPU through JAX. A shield can wrap any callable from tensor to tensor.

## What changes when you use it

- The model keeps learning even when the stream has `NaN`, spikes, or missing samples — they are cleaned before the model sees them, and every correction reports how far it moved.
- Concept drift is caught by an explicit detector instead of silently degrading accuracy.
- Anomalies come back as a score, one sample at a time.
- Robot commands are bounded (rate limits, control barrier functions) before they reach the actuator.
- Predictions can carry a calibrated uncertainty interval.

In this library, *shielding* means cleaning the data a model sees (NaN, spikes, anomalous samples) and bounding the commands a robot receives, around any online learner that follows the `learn_one` / `score_one` contract. It is not the formal shield of safe reinforcement learning.

## What it does

Dense-Armor is a Python machine learning library with an online part. Every online estimator learns one sample at a time, at control-loop speed, on JAX, and the machine learning classes of `dense_armor.ml` read a whole dataset with `fit`. The same `learn_one` / `predict_one` / `transform_one` contract runs on a table, a time series, a robot joint stream, a video frame, a token stream, and a shield can sit in front of any of them. On top of that, `dense_armor.utility.neural` has the deep-learning pieces (layers, attention, convolutional and recurrent blocks, batch and streaming training, continual learning with EWC/GEM/A-GEM, memory consolidation with vector quantization, product quantization and a residual tree), and `dense_armor.utility.stats` and `.timeseries` have the robust models (Student-t, Gaussian mixture, GARCH, AR/VAR/Kalman).

---

## Quickstart

As a quick example, we'll train an online Gaussian naive Bayes classifier to classify a drift stream with 5 features and one sudden concept change. Here's a look at the first observation in the dataset.

```python
>>> from dense_armor.utility.datasets import DriftStream

>>> dataset = DriftStream(n_samples=2000, n_features=5, n_drifts=1, kind="sudden", seed=0)

>>> for x, y in dataset.stream():
...     print(x.to_dict())
...     print(y)
...     break
{'x0': 0.00889176732883129, 'x1': 0.7559062648510446, 'x2': -2.788269695138754, 'x3': 0.0168051027297742, 'x4': 1.9630566360240487}
0
```

Now let's run the model on the dataset in a streaming fashion. We sequentially interleave predictions and model updates. Meanwhile, we update a performance metric to see how well the model is doing.

```python
>>> from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
>>> from dense_armor.utility.metrics import Accuracy

>>> model = OnlineGaussianNB()

>>> metric = Accuracy()

>>> for x, y in dataset.stream():
...     y_pred = model.predict_one(x)
...     if y_pred is not None:
...         metric.update(y, y_pred)
...     model.learn_one(x, y)

>>> metric
Accuracy(n=1999, value=0.9989994997498749)
```

Of course, this is just a contrived example. Check the [getting started](https://tatopenn-cell.github.io/Dense-Armor/getting-started/) section of the documentation for a more thorough tutorial.

---

## On a robot

A joint torque stream from `SyntheticArm` (real Panda URDF, real rigid-body dynamics) carries a `1e6` spike at sample 50. A rate-limited follower caps the applied jump at `max_vel` per step, so a downstream controller never sees the spike:

```python
from pathlib import Path
import numpy as np
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.control.rate_limiter import rate_limited_follower

ds = SyntheticArm(
    Path("test/fixtures/urdf/panda.urdf"),
    period_s=1.0, rate_hz=50.0, n_cycles=2,
    fault_at_s=None, fault=None, seed=0,
)
tau = np.array([float(sig["tau_1"]) for sig, _ in ds.stream()])
tau[50] = 1e6

safe = rate_limited_follower(tau, max_vel=2.0, max_accel=1.0)
print(round(float(np.max(np.abs(tau))), 1), round(float(np.max(np.abs(safe))), 1))
```

```
1000000.0 272.2
```

`RigidBodyModel` loads dynamics from the same URDF (`.xacro` and `<mimic>` joints included), `quintic_trajectory` generates a minimum-jerk path, and `cbf_safety_filter` enforces a spatial barrier around an obstacle.

---

## Install

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[vision]"       # + camera (OpenCV), image folder (Pillow), in-memory
pip install "dense-armor[ros]"          # + rosbag readers
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
pip install "dense-armor[mcp]"          # + MCP server for LLM agents
pip install "dense-armor[quantum]"      # + Dense-Evolution (NISQ simulator)
```

Works on Python 3.10+. JAX is the backend for every estimator.

---

## Features

Dense-Armor provides online implementations of the following families of algorithms:

- Linear models, with a wide array of optimizers
- Decision trees (including FIMT-DD regression model trees), online forests (adaptive random forest, aggregated Mondrian forests, online extra trees) and rule sets (adaptive model rules)
- Naive Bayes, nearest neighbours, multiclass and multi-output wrappers, conjugate distributions
- [Ensembles](https://tatopenn-cell.github.io/Dense-Armor/ensemble/) (online bagging and boosting, exponentially weighted average, voting, stacking)
- [Bandits](https://tatopenn-cell.github.io/Dense-Armor/bandit/) (epsilon-greedy, UCB, Thompson sampling, Bayes-UCB, Exp3, LinUCB, offline replay)
- [Active learning](https://tatopenn-cell.github.io/Dense-Armor/active/) (random, fixed, variable and randomised uncertainty with a labelling budget)
- Factorization machines ([plain, field-aware, field-weighted, higher-order](https://tatopenn-cell.github.io/Dense-Armor/facto/)) and [recommenders](https://tatopenn-cell.github.io/Dense-Armor/reco/) (baseline, biased and plain matrix factorization, sliding DFT, skyline)
- Clustering
- Anomaly detection (including the local outlier factor)
- Drift detection
- Time series (AR, VAR, Kalman, adaptive Kalman, cycle residual)
- Online classifiers
- Online dynamics
- Calibration and metric learning
- Imbalanced learning
- [Vision](https://tatopenn-cell.github.io/Dense-Armor/vision/) (video files, frame features, patches, vocabulary, online reduction, motion, tracking, registration, dense flow)
- [Deep learning](https://tatopenn-cell.github.io/Dense-Armor/neural/) (layers, attention, convolutional and recurrent blocks, batch and streaming training, continual learning with EWC, GEM, A-GEM, memory consolidation with VQ, PQ, residual tree, dual and archival memory)
- Robot control (rate limits, control barrier functions, trajectories, kinematic tracking)
- Robot dynamics (URDF, passivity + CBF controllers)
- Shielding (1D series cleanup, input/output shield, per-point routing)
- [Finance pipelines](https://tatopenn-cell.github.io/Dense-Armor/finance/): nine pipelines for markets, fraud, intrusion and credit scoring, each page opening with the question it answers
- [Finance: Regime](https://tatopenn-cell.github.io/Dense-Armor/finance/regime/): the regime page.
- [Finance: Drift vs noise](https://tatopenn-cell.github.io/Dense-Armor/finance/drift_vs_noise/): the drift vs noise page.
- [Finance: Portfolio](https://tatopenn-cell.github.io/Dense-Armor/finance/portfolio/): the portfolio page.
- [Finance: Order book](https://tatopenn-cell.github.io/Dense-Armor/finance/order_book/): the order book page.
- [Finance: Stress](https://tatopenn-cell.github.io/Dense-Armor/finance/stress/): the stress page.
- [Finance: Streaming](https://tatopenn-cell.github.io/Dense-Armor/finance/streaming/): the streaming page.
- [Finance: Timing](https://tatopenn-cell.github.io/Dense-Armor/finance/timing/): the timing page.
- [Machine learning](https://tatopenn-cell.github.io/Dense-Armor/ml/): the classes of `dense_armor.ml` read the whole table in `fit`: statistics, preprocessing, trees, forests, ensembles, anomaly and change detection, vision

Dense-Armor also provides other online utilities:

- Feature extraction and selection
- Online statistics and sketches, including [stable float32 variance](https://tatopenn-cell.github.io/Dense-Armor/stats/stable_variance/) (`stable_var`, `stable_moments`): exactly `0.0` on identical values up to `1e38`, a compensated tree that runs on CPU and GPU, and an optional `dtype` on the 11 statistics estimators
- Metrics for streams and robots
- Online models: [autoregressive](docs/timeseries/ar.md), [Kalman](docs/timeseries/kalman.md), [cycle residual](docs/timeseries/cycle.md), [Student-t](docs/stats/student_t.md), [Gaussian mixture](docs/stats/mixture.md), [GARCH(1,1)](docs/timeseries/garch.md)
- Preprocessing, with a [text vectorizer](docs/preprocessing/text_vectorizer.md) that ranks documents by TF-IDF and cosine similarity
- Streams (CSV, arrays, rosbag, frame folders, merge by time, shuffle)
- Built-in datasets
- Progressive model validation
- Model pipelines
- LLM bridge (embeddings as features, estimators as JSON tools)
- Checks (`check_estimator`, p99 budget, memory growth, clone, pickle, state dict, roc_auc)
- Core engine (`DynamicAICodegen` JIT compiler, `ImageChunker`, `UniversalMemoryGuard`, `AdaptiveSignalStabilizer`, `AIHardwareProfiler`, `StochasticAdversarialNoise`)

Check out [the API](https://tatopenn-cell.github.io/Dense-Armor/api/) for a comprehensive overview.

---

## Should I be using Dense-Armor?

Learning on the whole dataset does the job for many problems, and Dense-Armor covers it with the machine learning classes of `dense_armor.ml`. An online approach fits when:

- you want a model that learns from new data without revisiting past data
- you want a model that is robust to concept drift
- the production context is event-based, not batch-based, and you want the code to look like it
- you want the model to be protected from corrupted input, NaN, spikes, or slow drift without retraining it
- you work with a real robot and need command-side safety in the same library that learns the model

Some specifics:

- It focuses on clarity and user experience, more than on raw speed.
- It runs on JAX, so the same code runs on CPU, GPU or TPU.
- It plays nicely with the rest of Python's ecosystem: numpy in, numpy out.

---

## Useful links

- [Documentation](https://tatopenn-cell.github.io/Dense-Armor/)
- [Getting started](https://tatopenn-cell.github.io/Dense-Armor/getting-started/)
- [API reference](https://tatopenn-cell.github.io/Dense-Armor/api/)
- [Changelog](https://tatopenn-cell.github.io/Dense-Armor/changelog/)
- [Package releases](https://pypi.org/project/dense-armor/#history)
- [Sister project (Dense-Evolution)](https://github.com/tatopenn-cell/Dense-Evolution)
- [DOI](https://doi.org/10.5281/zenodo.23584384)

---

## Contributing

Contributions are welcome. The library is built to be extended one estimator at a time, following the same interface (`learn_one` / `predict_one` / `transform_one`) and the same checks (`dense_armor.checks`). Code style is checked with `ruff check dense_armor`, configured in `pyproject.toml`.

---

## Citation

```bibtex
@software{pennacchio_dense_armor,
  author    = {Pennacchio, Salvatore},
  title     = {Dense-Armor: Robot-AI Online Shielding \\& Learning},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23584384}
}
```

---

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
Releases up to 1.1.32 were published under the Business Source License 1.1:
[LICENSE-BUSL-1.1-until-1.1.32.md](LICENSE-BUSL-1.1-until-1.1.32.md).
A commercial licence is available on request.

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Twin project of [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution)
(NISQ quantum circuit simulator).


## Speed

The per-sample cost of the main estimators, measured by
`benchmarks/speed.py` on an Intel Xeon 2.20 GHz with 2 cores, 20 seeds,
2000 samples per seed, first 200 discarded. Full table and machine in
`benchmarks/speed_log.md`.

| component | median µs | p99 µs |
|---|---|---|
| Hoeffding tree | 9.1 | 18.1 |
| Gaussian naive Bayes | 44.2 | 85.8 |
| Adaptive random forest (10 models) | 754.1 | 1374.0 |
| Half-Space Trees (25 trees) | 1285.2 | 2412.3 |
| ADWIN (`update`) | 9.5 | 16.2 |

`stable_var` is the compensated float32 variance. On `[4096, 4096]`, `axis=1`, it costs 3.59x the plain two-term formula on CPU (range [3.55, 4.60] over three sessions) and 2.24x on a T4 GPU (range [2.20, 2.27]): the compensated tree is `log2(n)` vectorised levels, so it runs on both backends. The maximum relative error against `np.var(float64)` is below the baseline formula on three of the four datasets of the task in each combination (CPU eager, CPU jit, GPU eager, GPU jit). The two JSON files are `benchmarks/stable_variance_float32.json` (CPU) and `benchmarks/stable_variance_float32_gpu.json` (GPU). The shift by the first element, on which the kernel is built, is the shifted-data formula of Chan, Golub and LeVeque (1983) with K = x[0]; the combination of the steps is by Salvatore Pennacchio.

The Hoeffding Anytime tree (`HoeffdingAnytimeTreeClassifier`) is 26-58x faster than the adaptive random forest on the drift benchmark of `benchmarks/neural/efdt_vs_arf.py` (21-51 µs per sample against 723-1563 µs): at parity on sudden and gradual drift, 30.0 pp behind after the change on incremental drift, no native drift detection. An opt-in speed option, not a replacement.

The robot-side components stay inside a 1 kHz control-loop budget
(1000 µs) at the median: 36.5 µs for the rate-limited follower, 0.9 µs
for the control barrier function filter, 8.1 µs for the kinematic
tracking controller, 66 µs for the rigid-body mass matrix and 55 µs for
the gravity forces (the last two through `jax.jit`).

## Acknowledgements

Dense-Armor takes inspiration from [River](https://riverml.xyz) (Montiel et al., 2020, [arXiv:2012.04740](https://arxiv.org/abs/2012.04740)), the online machine learning library.

- Drones: [Actuator fault](docs/drones/actuator_fault.md).
- Drones: [Sensor consistency](docs/drones/sensor_consistency.md).
- Drones: [Propeller health: is a propeller damaged?](docs/drones/propeller_health.md).
- Drones: [Battery endurance: how many minutes of flight are left, and how sure?](docs/drones/battery_endurance.md).
- Drones: [Safety filter: is the next move safe?](docs/drones/safety_filter.md).
- Drones: [Link health: is the radio link about to fail?](docs/drones/link_health.md).
- Drones: [Visual landing: is the pad where I think it is, and is the camera moving?](docs/drones/visual_landing.md).
- Drones: [Static vs online: a CART tree and an online Hoeffding tree on the same stream](docs/drones/static_vs_online.md).
- Drones: [Start here: a drone and a board](docs/drones/start_here.md).
- Drones: [ArduPilot against PX4](docs/drones/ardupilot.md).
- Drones: [Jetson power: how much does the power mode change the time per sample?](docs/drones/jetson_power.md).
- Drones: [Jetson energy: the joules per 1000 samples of the online and batch pipelines, and one command to measure them on a board](docs/drones/jetson_energy.md).
- Deep Learning: [Neural bricks: how do I check that a neural piece learns the right gradient?](docs/neural/index.md).
- Deep Learning: [Layers: how do I build a layer, check its gradient and use it in a stream?](docs/neural/layers.md).
- Deep Learning: [Chains: how do I join layers into a network and train it one sample at a time or on a batch?](docs/neural/chains.md).
- Deep Learning: [Convolution and pooling: how do I look for a pattern in an image and shrink it?](docs/neural/vision_layers.md).
- Deep Learning: [Recurrent layers and attention: how do I keep a memory of the past samples?](docs/neural/sequence_layers.md).
- Deep Learning: [Training: how do I train a network on a dataset I already have, and stop it before it memorises?](docs/neural/training.md).
- Deep Learning: [Continuous training: how do I keep a network learning from a stream, without blocking the loop and without losing it when the program stops?](docs/neural/continuous.md).
- Deep Learning: [Encoders and decoders: how do I squeeze a stream of joint states into a few numbers, or into a few integers, and expand it back or into a forecast?](docs/neural/encoder_decoder.md).
- Deep Learning: [Deep: what does a deep neural network need beyond the layers, and where is each piece in the library?](docs/neural/deep.md).
- Deep Learning: [Attention: how does a network look at the other steps of a sequence, and where is each piece in the library?](docs/neural/attention.md).
- Deep Learning: [Composable attention: how do I make a network look at one sensor only when another one says it matters, and fuse two sensors through a few numbers?](docs/neural/composable_attention.md).
- Deep Learning: [Continual learning: how do I keep a network from forgetting the past, and how do I measure what it remembers?](docs/neural/continual_learning.md).
- Deep Learning: [Sparse mixture of experts and test-time compute: how do I make a network use only part of its parameters on each sample, and how do I spend more compute at inference to get a better answer?](docs/neural/moe_testtime.md).
- Machine learning: [the families of `dense_armor.ml`](docs/ml/index.md), grown on the whole table: statistics, preprocessing, linear models, trees, forests, ensembles, anomaly and change detection, vision.
