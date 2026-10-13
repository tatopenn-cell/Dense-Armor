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
  <img alt="combination" src="https://img.shields.io/badge/combination-Lagrange%20(BLUE)-9cf.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<h3 align="center">Robot-AI Online Shielding &amp; Learning</h3>

<p align="center"><strong>A library for online learning on robot, vision, and AI streams, one sample at a time, with streaming statistics and protection, drift and anomaly detection, and safe robot commands.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

---

## `$ what it does`

A sensor that sends lost readings (`NaN`) or spits out an absurd value (`1e6` instead of `1.2`) silently breaks any downstream pipeline. Dense-Armor steps in between the raw data and the model that consumes it:

```
  corrupted data ──► [ INPUT SHIELD ] ──► AI model ──► [ OUTPUT SHIELD ] ──► clean output
                     purifies vs             │             purifies vs
                     reference               │             response-to-reference
                     (or robust blind        │             (or self-consistency)
                     estimate)
```

- **Input**: cleans the corrupted data toward a clean reference, if you have one — or toward a robust estimate derived from the data itself, if you do not.
- **Output**: checks that the model's response is not itself corrupted, comparing it with the response the model would give to the clean reference.
- **Error margin**: for each corrected value, returns how much it was shifted to clean it. Small correction → trust it. Large correction → treat with caution.

It leaves the weights untouched. It runs at runtime. It works from 1D up to 11D, tested (see `test/test_orca2.py`).

**Same discipline, a second domain**: from `$ rate_limiter` onward the package also covers command and trajectory safety for real robots — velocity/acceleration limits, spatial control barrier functions, minimum-jerk trajectory generation, rigid dynamics from real URDF (including `.xacro`, including `<mimic>` joints), passivity+CBF controllers up to 6-DoF. JAX backend shared with `Armatura`/`Orca`, same verification discipline on real data/robots — a different application domain, not a different package.

**A third domain, online learning**: estimators that learn one sample at a time on a robot or next to an LLM, all on the same foundation (`dense_armor.roles`): `learn_one` / `predict_one` / `score_one`, timestamped `Signal` samples with units, a real-time contract (p99 latency within the control period), conformal uncertainty, `SafeEstimator` with health and checksummed checkpoints, unit checks and URDF joint limits. On top of it: online statistics and sketches, metrics with predict-then-learn evaluation and event metrics for robots (detection delay, false alarms per hour), and four drift detectors (CUSUM, Page-Hinkley, ADWIN, KSWIN). LLM side: embeddings as features and estimators exposed as JSON tools for agents. This layer is being built for robots from the ground up: it takes the best-established online-learning techniques from the original papers (streaming statistics, predict-then-learn evaluation, sequential change detection, conformal prediction) and extends them with time, units, real-time budgets, uncertainty and safety. Native vision has started: frame streams, frame features and online reduction.


---

## Install

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[quantum]"      # + Dense-Evolution (NISQ simulator)
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
pip install "dense-armor[vision]"       # + OpenCV frame readers
pip install "dense-armor[ros]"          # + rosbag readers
```

Works on Python 3.10+. JAX is the backend for every estimator.

---

## Quickstart

A joint torque stream with a NaN and a spike. A shield cleans the samples
before a downstream estimator sees them, and reports how far each sample
was moved:

```python
import numpy as np
from dense_armor import Armatura

serie = np.array([1.0, 1.1, 1.2, 9999.0, 1.15, np.nan, 1.2, 1.25])
arm = Armatura(livello_ia=0.0)
pulito, K, anomalie = arm.analizza(serie)
print([round(float(v), 2) for v in pulito])
print(anomalie)
```

```
[1.0, 1.1, 1.05, 1.1, 1.15, 1.2, 1.2, 1.2]
[3, 5]
```

Every estimator in the library shares the same interface (`learn_one` /
`predict_one` / `transform_one`). The shields use it too, so a shield can
sit in front of any model without a wrapper.

---

## Features

### Online learning (the core)

- Linear models with a wide array of optimizers (`optim/`)
- Decision trees, online forests (`tree/`)
- Clustering, including online k-means (`cluster/`)
- Calibration (Platt, isotonic), metric learning (`learn/`)
- Time series: online AR, VAR, Kalman filters, adaptive Kalman, cycle residual (`timeseries/`)
- Anomaly detection: half-space trees, LODA, one-class SGD, online isolation forest, robust Mahalanobis (`anomaly/`)
- Drift detection: CUSUM, Page-Hinkley, ADWIN, KSWIN (`drift/`)
- Online statistics, sketches, quantiles (`stats/`)
- Metrics with predict-then-learn evaluation, event metrics for robots (`metrics/`)

### Robot control (`control/`)

- Velocity and acceleration rate limiter
- Spatial control barrier function filter
- Minimum-jerk and quintic trajectory generation
- Kinematic controller with tracking

### Robot dynamics (`dynamics/`)

- Rigid-body dynamics loaded from a real URDF file (including `.xacro`)
- `<mimic>` joints handled natively
- Passivity + CBF controllers up to 6-DoF
- Same JAX backend, same verification discipline as the rest of the library

### Vision (`vision/`)

- Frame streams from a camera
- Frame features and patches
- Online vocabulary and reduction for real-time use
- Live demo available (`demo.md`)

### Shielding (optional, on top)

- `Armatura`: 1D series cleanup with a reference or a robust blind estimate
- `Orca`: input and output shield, with the error margin of every correction
- `Arbiter`: per-point routing between clean and observed values
- Robust filters, curvature-based streaming check

### Evaluation and tools

- Model validation with predict-then-learn (`utility/evaluate/`)
- Calibration and metric learning (`learn/`)
- Streaming estimators exposed as JSON tools for LLM agents (`mcp.md`)
- Toolkit for benchmark and diagnostic scripts (`toolkit/`)
- Preprocessing pipeline (`preprocessing/`)
- Signal abstraction with units and timestamps (`stream/`)



---

## Should I be using Dense-Armor?

The answer is probably no. Most of the time, batch learning does the job just
fine. An online approach fits when:

- you want a model that learns from new data without revisiting past data
- you want a model that is robust to concept drift
- you want the code to look like the production context, which is
  event-based, not batch-based
- you want the model to be protected from corrupted input, NaN, spikes, or
  slow drift without retraining it
- you work with a real robot and need command-side safety (rate limits, CBF,
  minimum-jerk trajectories) in the same library that learns the model

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

Contributions are welcome. The library is built to be extended one estimator
at a time, following the same interface (`learn_one` / `predict_one` /
`transform_one`) and the same checks (`dense_armor.checks`).

- open a discussion on the repository for questions or ideas
- open an issue if you spot a bug or a performance problem
- check [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and the code
  style, if the file exists in the repository

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
