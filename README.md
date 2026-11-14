<p align="center">
<img width="1800" height="400" alt="banner_nuovo" src="https://github.com/user-attachments/assets/9557ee09-fc5d-4ae9-8e1a-684017013f6e" />
</p>

<p align="center">
  <img alt="tests" src="https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/tatopenn-cell/Dense-Armor"><img alt="codecov" src="https://codecov.io/gh/tatopenn-cell/Dense-Armor/branch/master/graph/badge.svg"></a>
  <a href="https://pypi.org/project/dense-armor"><img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.24030156"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.24030156.svg"></a>
  <img alt="license" src="https://img.shields.io/badge/license-AGPL_3.0-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="vision" src="https://img.shields.io/badge/vision-native-9cf.svg">
  <img alt="robot control" src="https://img.shields.io/badge/robot%20control-rate%20limit%20%2B%20CBF-9cf.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<h3 align="center">Library of Primitives from__Robot__to__AI__ </h3>

<p align="center"><strong>A low-level learning toolkit on three levels: online learning one sample at a time, deep learning, and machine learning on a whole dataset. With a graph ontology, audio, vision and video, robot motion control, drift and anomaly detection, and finance blocks with a tamper-evident decision log. Built to run inside the loop, and to be defended in front of a reviewer.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

<p align="center"><a href="https://github.com/tatopenn-cell/Dense-Armor/archive/refs/heads/master.zip"><img alt="Download for Windows" src="https://img.shields.io/badge/Download-for_Windows-0078D4?style=for-the-badge&logo=windows&logoColor=white"></a> <a href="#dashboard-one-command"><img alt="How to run" src="https://img.shields.io/badge/Dashboard-how_to_run-2ea44f?style=for-the-badge&logo=gitbook&logoColor=white"></a></p>

---

## Who is this for

Python developers working on systems where data arrives one sample at a time and the model has to learn, or stay stable, while it runs. That is a trading desk watching returns and orders, a neural network training loop that cannot stop, a drone streaming sensors, a robot arm holding a payload, or any pipeline that never asks for the whole dataset first. A fixed dataset works too: the [Machine learning](https://tatopenn-cell.github.io/Dense-Armor/ml/) classes of `dense_armor.ml` read the whole table at once, with the same interface as the online ones.

## Use cases

The primitives are domain-agnostic: anything that ingests data one sample at a time and has to react when the stream changes. They have been measured on stock and crypto prices, card transactions, network flows and a telemarketing log (the [finance pages](https://tatopenn-cell.github.io/Dense-Armor/finance/)), on neural networks trained one batch at a time, on drone and sensor streams, and on a robot arm (CASPER on a UR3e). The layers above and below the model are the same shape everywhere: one dict in, one answer out.

| domain | what Dense-Armor brings |
|---|---|
| Trading and market surveillance | pre-trade risk (size, price band, duplicate, rate, notional), a tamper-evident decision log, regime and drift on returns, order flow imbalance, portfolio weights |
| Neural network training | layers, attention (rotary, causal, cross), convolutional and recurrent blocks, batch and streaming training, continual learning (EWC, GEM, A-GEM), memory consolidation, a decoder language model |
| Drones and IoT | actuator fault, sensor consistency, battery endurance, link health, visual landing, noisy multichannel input |
| Robot control loops | bounded commands (rate limits, CBF), drift and anomaly on joint streams, calibrated intervals |
| Deployed ML models | input drift detection, runtime shielding of `NaN` and spikes, no retraining required |
| Live monitoring | Prometheus-style metrics with drift detection, sketches, quantiles in constant memory |
| Search and ranking | a monitoring layer around the ranker: drift on user behaviour, streaming metrics |
| Antivirus and security | event stream statistics, behavioural drift, anomaly scoring one event at a time |

## What it expects from you

No specific model. Any estimator that follows the same contract: `learn_one(x, y)` updates the model from one sample, `predict_one(x)` returns a prediction, `transform_one(x)` returns a transform or a residual, `score_one(x)` returns an anomaly score (high = anomalous), `update(x)` advances a drift detector. Inputs and outputs are dicts of numpy-compatible scalars or arrays. A trading tick, a sensor reading and a neural network mini-batch all look the same to it: one dict in, one answer out. It runs on CPU or GPU through JAX, and a shield can wrap any callable from tensor to tensor.

## What changes when you use it

- An order is checked against the five limits of SEC Rule 15c3-5(c)(1) before it leaves, and every decision lands in a hash chain that anyone can verify.
- A slow regime change in returns is caught by an explicit detector, instead of silently moving the risk model.
- Anomalies in a price-and-volume stream come back as a score, one day at a time.
- A neural network keeps training on a stream without blocking the loop, and does not forget what it learned on the previous task.
- A drone's battery estimate carries a calibrated interval, and its actuator residual is watched on every tick.
- A robot's commands are bounded before they reach the actuator, so a spike or a NaN never turns into motion.
- The model keeps learning even when the stream has missing samples, and every correction reports how far it moved.

In this library, *shielding* means cleaning the data a model sees — NaN, spikes, anomalous samples — and bounding the commands a robot receives, around any online learner that follows the `learn_one` / `score_one` contract. It is not the formal shield of safe reinforcement learning.

## What it does

Dense-Armor is a low-level learning toolkit on three levels: online learning, deep learning and machine learning. Every online estimator learns one sample at a time, at control-loop speed, on JAX, and the machine learning classes of `dense_armor.ml` read a whole dataset with `fit`. The same `learn_one` / `predict_one` / `transform_one` contract runs on a market tick, a neural network mini-batch, a drone sensor stream, a robot joint stream, a video frame, a token stream. On top of that, `dense_armor.core.deep.neural` has the deep-learning pieces — layers, attention, convolutional and recurrent blocks, batch and streaming training, continual learning with EWC, GEM and A-GEM, memory consolidation with vector quantization, product quantization and a residual tree — and `dense_armor.core.signals.stats` and `.timeseries` have the robust models: Student-t, Gaussian mixture, GARCH, and the AR, VAR and Kalman families.

---

## Quickstart

A reproducible Nasdaq feed meets the five pre-trade controls of SEC Rule 15c3-5(c)(1), and every decision lands in a hash chain that anyone can verify. No external data, no app, no API key.

```bash
pip install dense-armor
```

```python
from pathlib import Path
from dense_armor.core.finance import PreTradeChecker, DecisionLog
from dense_armor.core.finance.synthetic import synthetic_itch
from dense_armor.core.finance.feed import iter_itch

# 1. a Nasdaq ITCH feed that is byte-identical on every machine, from a seed
Path("feed.bin").write_bytes(synthetic_itch(1000, seed=0))

# 2. the five pre-trade controls, and a tamper-evident decision log
checker = PreTradeChecker(
    max_notional=50_000.0, max_size=0.2, price_band=0.001,
    max_orders=3, window_s=10.0, duplicate_window_s=1.0,
)
log = DecisionLog("decisions.jsonl")

for msg in iter_itch("feed.bin"):
    if msg["type"] not in ("A", "F"):
        continue
    side = "buy" if msg["side"] == "B" else "sell"
    d = checker.check(
        msg["stock"], side, msg["shares"] / 100,
        msg["price"], msg["price"], t=msg["timestamp"] / 1e9,
    )
    log.append(t=msg["timestamp"] / 1e9, payload={
        "account": msg["stock"], "approved": d.approved, "reason": d.reason,
    })

print(f"Decisions: {len(log)}")
print(f"Verify:    {log.verify()}")
```

```
Decisions: 493
Verify:    True
```

Now edit one record on disk and verify again: the chain breaks.

```python
lines = Path("decisions.jsonl").read_text().splitlines(keepends=True)
lines[1] = lines[1].replace('"reason":"size"', '"reason":"ok"')
Path("decisions.jsonl").write_text("".join(lines))
print(f"Verify after tamper: {DecisionLog('decisions.jsonl').verify()}")
```

```
Verify after tamper: False
```

What just happened: every order has been approved, refused or clamped against the five limits — size, price band, duplicate, rate, notional — and the outcome of each one has been chained to the previous decision. The edit has broken the chain, and verify() has caught it. This is the same log a reviewer will ask for, and the same check they will run. The specification is in docs/finance/pretrade.md and docs/finance/decision_log.md.

---

## Dashboard: one command

The web dashboard of the finance system (chain of pieces, decisions, emergency stop) is not in the pip package: it lives in the repository, in `tools/dashboard/`.

**Windows, like any program:**

1. Press **Download for Windows** at the top and unzip the file.
2. Double-click `tools\dashboard\install-dashboard.bat`. It installs Python 3.12 if the computer has none, makes a private environment and installs the packages (a few minutes the first time).
3. The icon **Dense-Armor Dashboard** appears on the desktop. Double-click it: the dashboard opens in the browser at http://127.0.0.1:8080/.

The program is installed in `%LOCALAPPDATA%\Dense-Armor`; you can delete the downloaded folder. Running the installer again updates it.

**Without installing**, from a clone of the repository:

```bash
git clone https://github.com/tatopenn-cell/Dense-Armor.git
cd Dense-Armor
sh tools/dashboard/start-dashboard.sh
```

On Windows the same without the icon is `tools\dashboard\start-dashboard.bat`. Options are passed on, for example `--port 9000`. Python 3.11 or later is needed.

---

## Install

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[vision]"       # + camera (OpenCV), image folder (Pillow), in-memory
pip install "dense-armor[ros]"          # + rosbag readers
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
pip install "dense-armor[graph]"        # + networkx, for page rank and to_networkx()
pip install "dense-armor[mcp]"          # + MCP server for LLM agents
pip install "dense-armor[quantum]"      # + Dense-Evolution (NISQ simulator)
pip install "dense-armor[sklearn]"      # + scikit-learn bridge
pip install "dense-armor[torch]"        # + PyTorch bridge
pip install "dense-armor[tensorflow]"   # + TensorFlow bridge
```

Works on Python 3.10+. JAX is the backend for every estimator.

---

## Features

Dense-Armor provides the following families of algorithms, online unless a line says otherwise:

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
- [Deep learning](https://tatopenn-cell.github.io/Dense-Armor/neural/) (layers, attention, convolutional and recurrent blocks, batch and streaming training, continual learning with EWC, GEM, A-GEM, memory consolidation with VQ, PQ, residual tree, dual and archival memory, a decoder language model with RMSNorm, rotary attention and SwiGLU, and generation against a static key/value cache)
- [Audio](https://tatopenn-cell.github.io/Dense-Armor/audio/) (sound as a value, WAV files and live input, filters, effects, synthesis, pitch, time and pitch change, mel and MFCC, event detection, and one adapter that gives sound to every estimator, detector and scaler)
- [Ontology](https://tatopenn-cell.github.io/Dense-Armor/ontology/) (a typed graph with rules and actions, graph spectrum and filters, an execution DAG of estimators with a hash-chained provenance, anomaly and root-cause scores, recovery and learning of a graph, online graph estimation with `AdaCGP`, causal effects)
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
- Bridges: scikit-learn, PyTorch and TensorFlow
- Checks (`check_estimator`, p99 budget, memory growth, clone, pickle, state dict, roc_auc)
- Core engine (`UniversalMemoryGuard`, `AdaptiveSignalStabilizer`, `AIHardwareProfiler`)

Check out [the API](https://tatopenn-cell.github.io/Dense-Armor/api/) for a comprehensive overview.
---

## Why Dense-Armor

**Same interface, no rewrite.** Any estimator that follows the `learn_one` / `predict_one` / `predict_proba_one` / `score_one` contract plugs into a pipeline. If you already have an online loop, you do not rewrite it to use this library.

**The areas the online-ML toolchain leaves open.** Each of these comes with a measured claim against an external reference, not a promise. Neural network training on a stream, with an error well below a constant baseline and a classifier that reaches full accuracy on a standard dataset. Streaming recommenders, with an error an order of magnitude below the mean. Seven streaming metrics — log-loss, Brier, Cohen's kappa, precision, recall, F-beta2 and softmax cross-entropy — that agree with their batch counterparts to numerical precision. Model selection on a stream, where the best candidate clearly outperforms the average. Bandits, where the policy earns far more than random play. Anomaly detection from recent literature, with a strong ranking quality on a public credit-card fraud dataset, zero training labels, and a fit measured in seconds.

**The operational layer.** These sit next to the model, not in a separate project. Pre-trade risk against the five controls of SEC Rule 15c3-5(c)(1): size, price band, duplicate, rate, notional. A tamper-evident decision log, built as a SHA-256 hash chain in JSON Lines, with the same chain in PostgreSQL and a trigger that forbids editing or deleting a record. A Nasdaq ITCH decoder and a synthetic feed that reproduces byte-identically from a seed. A FIX gateway, TWAP and VWAP schedules, and a streaming value-at-risk that updates without rescanning the past. Market surveillance with two independent detectors that must agree, on years of S&P 500 history. Shielding for NaN, spikes and anomalous samples around any online learner. Rate-limited and barrier-bounded commands on a real robot. Drone and sensor pipelines. Native vision.

**What this is not.** A larger model, a faster GPU kernel, a new optimizer. It is the layer that turns a model into something you can put in front of a desk, a flight controller or a sensor, and defend in front of a reviewer.

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
- [DOI](https://doi.org/10.5281/zenodo.24030156)

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

`stable_var` is the compensated float32 variance. On `[4096, 4096]`, `axis=1`, it costs 3.59x the plain two-term formula on CPU (range [3.55, 4.60] over three sessions) and 2.24x on a T4 GPU (range [2.20, 2.27]): the compensated tree is `log2(n)` vectorised levels, so it runs on both backends. The maximum relative error against `np.var(float64)` is below the baseline formula on three of the four datasets of the task in each combination (CPU eager, CPU jit, GPU eager, GPU jit). The two JSON files are `benchmarks/stable_variance_float32.json` (CPU) and `benchmarks/stable_variance_float32_gpu.json` (GPU). The shift by the first element, on which the kernel is built, is the shifted-data algorithm with K = x[0] (Healy 2022, Algorithm 2); the local centring, the compensated TwoSum tree and the float32 evaluation with `vmap` are by Salvatore Pennacchio.

The Hoeffding Anytime tree (`HoeffdingAnytimeTreeClassifier`) is 26-58x faster than the adaptive random forest on the drift benchmark of `benchmarks/neural/efdt_vs_arf.py` (21-51 µs per sample against 723-1563 µs): at parity on sudden and gradual drift, 30.0 pp behind after the change on incremental drift, no native drift detection. An opt-in speed option, not a replacement.

The robot-side components stay inside a 1 kHz control-loop budget
(1000 µs) at the median: 36.5 µs for the rate-limited follower, 0.9 µs
for the control barrier function filter, 8.1 µs for the kinematic
tracking controller, 66 µs for the rigid-body mass matrix and 55 µs for
the gravity forces (the last two through `jax.jit`).

---

## Team

Maintained by [Salvatore Pennacchio](mailto:jtatopenn@libero.it). Issues and pull requests on [GitHub](https://github.com/tatopenn-cell/Dense-Armor/issues); questions and general discussion in [GitHub Discussions](https://github.com/tatopenn-cell/Dense-Armor/discussions).

---

## Contributing

Contributions are welcome. The library is built to be extended one estimator at a time, following the same interface (`learn_one` / `predict_one` / `transform_one`) and the same checks (`dense_armor.checks`). Code style is checked with `ruff check dense_armor`, configured in `pyproject.toml`.

---

## Citation

```bibtex
@software{pennacchio_dense_armor,
  author    = {Pennacchio, Salvatore},
  title     = {Dense-Armor: Low-Level Learning toolkit},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24030156}
}
```
---


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
- Deep Learning: [Language-model training: how do I train a small language model with Dense-Armor?](docs/neural/lm_training.md).
- Deep Learning: [Rotary decoder: how do I generate text without running the whole sequence again at every token?](docs/neural/rotary_decoder.md).
- Machine learning: [the families of `dense_armor.ml`](docs/ml/index.md), grown on the whole table: statistics, preprocessing, linear models, trees, forests, ensembles, anomaly and change detection, vision.
---

## References

Dense-Armor builds on 236 works across online learning, streaming statistics, robot control, deep learning and finance. The full list is in [REFERENCES.md](REFERENCES.md).

Key references: [River](https://riverml.xyz) (Montiel et al., 2020, [arXiv:2012.04740](https://arxiv.org/abs/2012.04740)), [scikit-learn](https://scikit-learn.org) (Buitinck et al., 2013), [CASPER](https://arxiv.org/abs/2303.01300) (Kayan et al., 2023), ADWIN (Bifet & Gavaldà, 2006), CBF (Ames et al., 2019), EWC (Kirkpatrick et al., 2017).

   
## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
Releases up to 1.1.32 were published under the Business Source License 1.1:
[LICENSE-BUSL-1.1-until-1.1.32.md](LICENSE-BUSL-1.1-until-1.1.32.md).
A commercial licence is available on request.

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Twin project of [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution)
(NISQ quantum circuit simulator).

