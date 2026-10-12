# Getting Started

Dense-Armor is a library of online estimators that learn one sample at a
time from robot and AI data. It detects anomalies and drift, protects a
model's inputs and outputs, and bounds the commands sent to a real robot.
Everything runs on JAX, so the same code runs on CPU, GPU or TPU.

This page is a guided tour, from the smallest unit (one sample) to a full
robot loaded from its URDF file. Every example is runnable as-is.

## 1. Install

```bash
pip install dense-armor
pip install "dense-armor[quantum]"
pip install "dense-armor[audio,data]"
```

`[quantum]` adds the Dense-Evolution NISQ simulator; `[audio,data]` adds
WAV, HDF5 and NetCDF readers.

Then, in Python, enable 64-bit floats before importing anything else:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

The JAX configuration must run before any other `dense_armor` import. The
shields work in double precision, and JAX defaults to 32-bit floats.

## 2. Quickstart: flag an anomaly in one line

From the command line:

```bash
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

The CLI reads a small JSON array and prints one line per anomaly found.

## 3. Score one sample at a time

The smallest useful tool: a detector that reads a scalar stream and flags
whether the current sample looks like a deviation from the recent past.

```python
from dense_armor.utility.anomaly.streaming import StreamingDeviationDetector

det = StreamingDeviationDetector()
xs = [1.0] * 20 + [9999.0] + [1.0] * 10
flags = [bool(det.update(x)) for x in xs]
```

`update(x)` returns `True` at the one sample that stands out and `False`
everywhere else. The detector keeps a causal window of the recent past;
no future value is used. The window defaults are wide enough that a single
isolated spike does not shift the baseline.

## 4. Clean a whole 1D series

When the input is a full array (a loss curve, a sensor channel, a token
stream), the `Armatura` shield cleans it in one call and tells you where
the anomalies were.

```python
import numpy as np
from dense_armor import Armatura

serie = np.array([1.0, 1.1, 1.2, 9999.0, 1.15, np.nan, 1.2, 1.25])
arm = Armatura(livello_ia=0.0)
pulito, K, anomalie = arm.analizza(serie)
```

`pulito` is the cleaned series with the same length as the input,
`K` is a per-sample indicator array (`1.0` where the point was flagged),
and `anomalie` is the list of flagged indices. With `livello_ia=0.0` the
shield actively filters; with `livello_ia=1.0` it only marks.

## 5. Protect a real model

`Orca` wraps any callable from a tensor to a tensor and shields both its
input and its output.

```python
import numpy as np
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0

orca = Orca(min_free_ram_percentage=0.05)
protected = orca.protect_and_forward(lambda z: 2.0 * z, data, x_reference=ref)
```

`x_reference` is optional but recommended: if a clean reference is
available (the previous reading, a nominal trajectory), the input shield
uses it; otherwise it falls back to a robust blind estimate.
`orca.margine_ingresso_medio` and `orca.margine_uscita_medio` report how
much the shield trusts the input and the output.

## 6. Route each point to the right corrector

The **Arbiter** classifies each point as `clean`, `spike` or `regime`
against a causal reference window (only points before it). It is off by
default.

```python
import numpy as np
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0

orca = Orca(min_free_ram_percentage=0.05)
protected = orca.protect_and_forward(
    lambda z: 2.0 * z, data, x_reference=ref, use_arbiter=True,
)
labels = orca.etichette_arbitro
```

- `spike` (an isolated impulse) is hard-rejected to the window median.
- `regime` (a sustained level change) is passed through raw, fully trusted.
- `clean` is whatever the standard four-phase shield produced.

## 7. Load a real robot from its URDF

The dynamics module reads the robot's own URDF file. For the Panda arm of
`test/fixtures/urdf/panda.urdf`, the model has nine degrees of freedom
(seven revolute joints plus two prismatic finger joints).

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

m = RigidBodyModel("test/fixtures/urdf/panda.urdf")
q = np.zeros(m.n)
qd = np.zeros(m.n)
M = m.mass_matrix(q)
g = m.gravity_forces(q)
fk = m.forward_kinematics(q)
```

`m.mass_matrix(q)` returns the joint-space inertia as a `(9, 9)` matrix.
`m.gravity_forces(q)` returns the torque needed to hold the arm still as a
`(9,)` vector. `m.forward_kinematics(q)` returns a dictionary with the
position, orientation and axis of every link, keyed by link name.

## 8. Plan and bound a joint command

Two standalone control utilities: a closed-form trajectory and a
rate-limited follower.

```python
import numpy as np
from dense_armor.utility.control.trajectory import quintic_trajectory

q0 = np.zeros(7)
qf = np.ones(7)
t, q, v, a = quintic_trajectory(q0, qf, T=2.0, n_samples=200)
```

`quintic_trajectory` returns position, velocity and acceleration at each
of the `n_samples` time steps, starting and ending at rest by default.

```python
import numpy as np
from dense_armor.utility.control.rate_limiter import rate_limited_follower

q_raw = np.linspace(0.0, 0.1, 20)
cmd = rate_limited_follower(q_raw, max_vel=1.0, max_accel=2.0, dt=0.01)
```

`rate_limited_follower` tracks a raw incoming command stream and returns
the applied command, whose velocity and acceleration never exceed the
limits. It uses only the previous applied state and the current target;
future values are never used.

## 9. Learn online

A classifier that learns one sample at a time and answers immediately.

```python
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

clf = OnlineGaussianNB()
for _ in range(20):
    clf.learn_one({"x": 0.0}, "a")
    clf.learn_one({"x": 1.0}, "b")

pred = clf.predict_one({"x": 0.1})
proba = clf.predict_proba_one({"x": 0.1})
```

With well-separated classes, `pred` is `"a"` and `proba` is
`{"a": 1.0, "b": 0.0}`.

## 10. Detect a multivariate anomaly

When a fault touches several channels at once, use a multivariate
detector. All five share the same `learn_one` / `score_one` interface.

```python
import numpy as np
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis

det = OnlineRobustMahalanobis()
rng = np.random.default_rng(0)
for _ in range(50):
    det.learn_one({"a": float(rng.normal()), "b": float(rng.normal())})

sc = det.score_one({"a": 5.0, "b": 5.0})
```

`sc` is high for a point far from the learned distribution. `5.0` on both
channels against a standard-normal training set gives a score around
`9.2`. The default flagging threshold is `7.0`.

## 11. See a frame in patches

For vision streams, cut a frame into patches and describe each one.

```python
import numpy as np
from dense_armor.utility.vision.patches import PatchFeatures

pf = PatchFeatures(patch=8, stride=8)
img = np.random.default_rng(0).random((32, 32))
out = pf.transform_one(img)
```

For a `32x32` frame with `patch=8` and `stride=8`, `out["descriptors"]`
has shape `(16, 11)`: 16 patches, 11 features each. `out["positions"]`
has shape `(16, 2)`: the row and column of each patch.

## 12. Toolkit

A second, independent part of the package. Two examples out of 14 modules.

```python
from dense_armor.core import DynamicAICodegen

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "l2_normalize"])
out = codegen.run_dynamic_pipeline([-2.0, 3.0, -1.0, 4.0], ops)
```

`out` is `[0., 0.6, 0., 0.8]`: relu clips negatives, then L2-normalize.

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()
```

Raises `MemoryPressureError` if free RAM drops below 10 %.

See [Toolkit](toolkit/toolkit.md) for the full list.

## Details

### What changed from the previous version

The earlier version of this page was written before the robot, learning
and vision modules existed. This rewrite adds them, moves the examples
from the old `panda_joint1..7` naming to the current URDF-driven
`RigidBodyModel`, and replaces two functions that no longer exist
(`RateLimiter`, `Trajectory`) with their current forms
(`rate_limited_follower`, `quintic_trajectory`).

### Verified numbers

All outputs printed in this page were produced by the installed package
at the time of writing: `m.mass_matrix` is `(9, 9)`, `m.gravity_forces`
is `(9,)`, `OnlineRobustMahalanobis.score_one` on a 5-sigma outlier is
about `9.2` against a default threshold of `7.0`, `PatchFeatures` on a
`32x32` frame with `patch=8, stride=8` yields `(16, 11)` descriptors.
