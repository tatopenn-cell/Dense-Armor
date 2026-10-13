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
  <img alt="training" src="https://img.shields.io/badge/training%20required-no-brightgreen.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <img alt="detectors" src="https://img.shields.io/badge/anomaly%20detectors-12-blueviolet.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<h3 align="center">Robot-AI Online Shielding &amp; Learning</h3>

<p align="center"><strong>A library for online learning on robot, vision, and AI streams, one sample at a time, with streaming statistics and protection, drift and anomaly detection, and safe robot commands.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

---

## What it does

A sensor that sends a lost reading (`NaN`) or an absurd value (`1e6` instead of `1.2`) silently breaks any downstream model. Dense-Armor sits between the raw data and the model that consumes it, and learns from the clean stream one sample at a time:

```
  corrupted data ──► [ INPUT SHIELD ] ──► AI model ──► [ OUTPUT SHIELD ] ──► clean output
                     purifies vs             │             purifies vs
                     reference               │             response-to-reference
                     (or robust blind        │             (or self-consistency)
                     estimate)
```

- **Learning**: every estimator (`learn_one` / `predict_one` / `transform_one`) updates its state from one sample at a time. No batch retraining, no revisiting the past, fit for a control loop.
- **Shield**: the corrupted sample goes toward a clean reference if one exists, toward a robust blind estimate if it does not. Every correction carries an error margin: small correction → trust it, large correction → flag it.
- **Robot**: command-side safety on the same foundation — rate limits, control barrier functions, minimum-jerk trajectories, rigid dynamics from a real URDF, passivity + CBF controllers up to 6-DoF.
- **Vision**: frame streams, frame features and patches, online vocabulary and reduction.

Same interface everywhere, JAX-native (CPU, GPU, TPU), NaN-safe.

---

## Quickstart

Two classes of frames (`16x16`) stream at 30 fps. Two samples are corrupted on the way: a `1e6` spike at sample 100 and a `NaN` at sample 150. The shield cleans them, an online Hoeffding tree learns one sample at a time from the frame features, and accuracy updates after every prediction:

```python
import numpy as np
import jax.numpy as jnp
from dense_armor.utility.vision import ArrayStream, FrameFeatures
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from dense_armor.utility.metrics import Accuracy
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
frames = np.zeros((200, 16, 16), dtype=np.float32)
labels = np.zeros(200, dtype=int)
for i in range(200):
    label = i % 2
    f = rng.normal(size=(16, 16)).astype(np.float32) * 0.5
    if label:
        f[6:10, 6:10] += 3.0
    frames[i] = f
    labels[i] = label

frames[100, 0, 0] = 1e6
frames[150, 0, 0] = np.nan

stream = ArrayStream(frames, fps=30.0, gray=True)
feat = FrameFeatures(n_bins=4, cells=(1, 1))
model = HoeffdingTreeClassifier()
metric = Accuracy()
shield = Orca()

k = 0
for frame in stream:
    x_in = jnp.asarray(frame.array[None, :, :])
    x_ref = jnp.asarray(frames[k][None, :, :])
    clean = shield.protect_and_forward(None, x_in, x_reference=x_ref,
                                       use_model_injection=False,
                                       use_output_shield=False)
    f = feat.transform_one(np.asarray(clean[0]))
    x = {"mean": float(f["mean"]), "contrast": float(f["contrast"])}
    y = int(labels[k])
    y_pred = model.predict_one(x)
    if k > 0 and y_pred is not None:
        metric.update(y, y_pred)
    model.learn_one(x, y)
    k += 1

print(metric)
```

```
Accuracy(n=199, value=0.9899497487437185)
```

Every estimator in the library uses the same shape: `learn_one(x, y)` updates the model, `predict_one(x)` returns the prediction, `transform_one(x)` returns a transform or a residual, `update(x)` advances a drift detector, `score_one(x)` scores an anomaly (high = anomalous). The shields use the same interface, so a shield can sit in front of any model without a wrapper.

The same code runs on a robot stream (`SyntheticArm`, `Casper`), on a drift stream (`DriftStream`), on a text stream, or next to an LLM: the estimator does not need to know where the sample comes from.

---

## On a robot

A joint torque stream from `SyntheticArm` carries a `1e6` spike at sample 50; a rate-limited follower caps the applied jump at `max_vel` per step, so a downstream controller never sees the spike:

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

The same `learn_one` / `predict_one` contract runs on the robot side: `RigidBodyModel` loads dynamics from a real URDF (`.xacro` and `<mimic>` joints included), `quintic_trajectory` generates a minimum-jerk path, `cbf_safety_filter` enforces a spatial barrier around an obstacle, and drift and anomaly detectors run on the joint stream at the same time as the model learns.

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
- Built-in datasets
- Progressive model validation
- Model pipelines
- LLM bridge (embeddings as features, estimators as JSON tools)

Check out [the API](https://tatopenn-cell.github.io/Dense-Armor/api/) for a comprehensive overview.

---

## Should I be using Dense-Armor?

The answer is probably no. Most of the time, batch learning does the job just fine. An online approach fits when:

- you want a model that learns from new data without revisiting past data
- you want a model that is robust to concept drift
- the production context is event-based, not batch-based, and you want the code to look like it
- you want the model to be protected from corrupted input, NaN, spikes, or slow drift without retraining it
- you work with a real robot and need command-side safety in the same library that learns the model
- you need a prediction with an online-calibrated uncertainty interval

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

- open a discussion on the repository for questions or ideas
- open an issue if you spot a bug or a performance problem
- check `CONTRIBUTING.md` for the workflow and the code style, if the file exists in the repository

---

## Citation

If Dense-Armor has been useful to you, please cite the Zenodo archive:

```bibtex
@software{pennacchio_dense_armor,
  author       = {Pennacchio, Salvatore},
  title        = {Dense-Armor: Robot-AI Online Shielding \& Learning},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23265098},
  url          = {https://doi.org/10.5281/zenodo.23265098}
}
```

---

## License

Business Source License 1.1 — free non-commercial use, converts to Apache 2.0
on `2029-06-01`. See [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Twin project of [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution)
(NISQ quantum circuit simulator).
