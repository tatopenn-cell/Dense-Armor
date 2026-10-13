<p align="center">
  <img src="docs/assets/banner.jpg" alt="Dense-Armor — Robot-AI Online Shielding &amp; Learning" width="900">
</p>

<p align="center">
  <img alt="tests" src="https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/tatopenn-cell/Dense-Armor"><img alt="codecov" src="https://codecov.io/gh/tatopenn-cell/Dense-Armor/branch/master/graph/badge.svg"></a>
  <a href="https://pypi.org/project/dense-armor"><img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.23265098"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23265098.svg"></a>
  <img alt="license" src="https://img.shields.io/badge/license-BSL_1.1-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="vision" src="https://img.shields.io/badge/vision-native-9cf.svg">
  <img alt="robot control" src="https://img.shields.io/badge/robot%20control-rate%20limit%20%2B%20CBF-9cf.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<h3 align="center">Robot-AI Online Shielding &amp; Learning</h3>

<p align="center"><strong>A library for online learning on robot, vision, and AI streams, one sample at a time, with streaming statistics and protection, drift and anomaly detection, and safe robot commands.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

---

## What it does

Dense-Armor is a Python library for online learning. Every estimator learns one sample at a time, at control-loop speed, on JAX: nothing is retrained in batch, nothing revisits the past. The same `learn_one` / `predict_one` / `transform_one` contract runs on a table, a time series, a video frame, an audio buffer, a robot joint stream, a token stream, and a shield can sit in front of any of them.

---

## Quickstart

A table, a video and an audio stream. One model, one metric, one loop. Everything is one sample at a time:

```python
import numpy as np
from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.vision import ArrayStream, FrameFeatures
from dense_armor.utility.preprocessing.scale import StandardScaler
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics import Accuracy

rng = np.random.default_rng(0)

tabular = DriftStream(n_samples=2000, n_features=2, n_drifts=1, kind="sudden", seed=0).stream()
video = ArrayStream(rng.normal(size=(200, 16, 16)), fps=30.0)
audio = rng.normal(size=200).cumsum()

model = StandardScaler() | OnlineGaussianNB()
metric = Accuracy()
ff = FrameFeatures(n_bins=4, cells=(1, 1))

for x, y in tabular:
    metric.update(y, model.predict_one(x))
    model.learn_one(x, y)

for i, frame in enumerate(video):
    x = ff.transform_one(frame.array)
    metric.update(i % 2, model.predict_one(x))
    model.learn_one(x, i % 2)

for i, v in enumerate(audio):
    x = {"a": float(v)}
    metric.update(i % 2, model.predict_one(x))
    model.learn_one(x, i % 2)

print(metric)
```

```
Accuracy(n=2399, value=0.9107961650687787)
```

Every estimator in the library uses the same shape. `predict_one(x)` returns a prediction, `metric.update(y, y_pred)` tracks the score, `learn_one(x, y)` updates the model from that one sample, then the sample is gone. Add a new kind of stream and the loop does not change: read the sample, predict, score, learn.

---

## On a robot

The same interface on a real Panda arm. A joint torque stream carries a `1e6` spike at sample 50; a rate-limited follower caps the applied jump at `max_vel` per step, so a downstream controller never sees the spike:

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

`RigidBodyModel` loads dynamics from a real URDF (`.xacro` and `<mimic>` joints included), `quintic_trajectory` generates a minimum-jerk path, `cbf_safety_filter` enforces a spatial barrier around an obstacle, and drift and anomaly detectors run on the joint stream at the same time as the model learns.

---

## Install

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[vision]"       # + OpenCV frame readers
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
- Decision trees and online forests
- Clustering
- Anomaly detection
- Drift detection
- Time series (AR, VAR, Kalman, adaptive Kalman, cycle residual)
- Online classifiers
- Online dynamics
- Calibration and metric learning
- Imbalanced learning
- Vision (frame features, patches, vocabulary, online reduction)
- Robot control (rate limits, control barrier functions, trajectories, kinematic tracking)
- Robot dynamics (URDF, passivity + CBF controllers)
- Shielding (1D series cleanup, input/output shield, per-point routing)

Dense-Armor also provides other online utilities:

- Feature extraction and selection
- Online statistics and sketches
- Metrics for streams and robots
- Preprocessing
- Streams (CSV, arrays, rosbag, frame folders, merge by time, shuffle)
- Built-in datasets
- Progressive model validation
- Model pipelines
- LLM bridge (embeddings as features, estimators as JSON tools)
- Checks (`check_estimator`, p99 budget, memory growth, clone, pickle, state dict, roc_auc)

Check out [the API](https://tatopenn-cell.github.io/Dense-Armor/api/) for a comprehensive overview.

---

## Should I be using Dense-Armor?

The answer is probably no. Most of the time, batch learning does the job just fine. An online approach fits when:

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
- [DOI](https://doi.org/10.5281/zenodo.23265098)

---

## Contributing

Contributions are welcome. The library is built to be extended one estimator at a time, following the same interface (`learn_one` / `predict_one` / `transform_one`) and the same checks (`dense_armor.checks`).

---

## Citation

```bibtex
@software{pennacchio_dense_armor,
  author    = {Pennacchio, Salvatore},
  title     = {Dense-Armor: Robot-AI Online Shielding \& Learning},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23265098}
}
```

---

## License

Business Source License 1.1 — free non-commercial use, converts to Apache 2.0
on `2029-06-01`. See [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Twin project of [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution)
(NISQ quantum circuit simulator).
