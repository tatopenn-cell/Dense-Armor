# Data in, one sample at a time

Every estimator in the library learns from a stream of samples: a `Signal` (named channels with
units and a timestamp) and, when there is one, a target `y`. Robot data arrive in many shapes:
log files, ROS bags, arrays, live sensors at different rates. This page turns all of them into
the same stream, merges sensors in time order, makes test streams whose changes happen at a known
moment, and ends with a model that refuses to learn from a fault.

## 1. A robot log file becomes a stream

A CSV file with the columns of the CASPER recording as it is distributed on Kaggle
(Kayan et al. 2023), one row per sample, with a timestamp column and a column holding the six
joint velocities as a list; here three joints and three rows.

```python
from pathlib import Path
from dense_armor.utility.stream import iter_csv

p = Path("arm.csv")
p.write_text(
    "Timestamp,Actual Joint Velocities\n"
    '0.00,"[0.1, -0.2, 0.0]"\n'
    '0.05,"[0.2, -0.1, 0.0]"\n'
    '0.10,"[0.3, 0.0, 0.1]"\n'
)
for sig, y in iter_csv(p, time_col="Timestamp", array_cols=["Actual Joint Velocities"]):
    print(sig.t, {k: round(float(v), 2) for k, v in sig.items()})
```

```
0.0 {'Actual_Joint_Velocities_0': 0.1, 'Actual_Joint_Velocities_1': -0.2, 'Actual_Joint_Velocities_2': 0.0}
0.05 {'Actual_Joint_Velocities_0': 0.2, 'Actual_Joint_Velocities_1': -0.1, 'Actual_Joint_Velocities_2': 0.0}
0.1 {'Actual_Joint_Velocities_0': 0.3, 'Actual_Joint_Velocities_1': 0.0, 'Actual_Joint_Velocities_2': 0.1}
```

Each row becomes one `Signal` with its timestamp; the list column is split into one
channel per joint. The file is read one row at a time, so a 24-hour log uses the same memory as a
three-row one; `start` and `stop` read only a time window and stop at the end of it. An empty or
unreadable cell becomes a missing channel, not an error.

## 2. Two sensors, one timeline

A camera at 30 frames per second and joint encoders at 100 Hz give two streams; an estimator
needs them as one stream, in time order.

```python
import numpy as np
from dense_armor.utility.stream import iter_array, merge_by_time

cam = iter_array(np.array([[1.0], [2.0]]), t=np.array([0.0, 1 / 30]), names=["frame"])
joints = iter_array(np.array([[0.1], [0.2], [0.3], [0.4]]), t=np.arange(4) / 100, names=["q0"])
for sig, _ in merge_by_time(cam, joints):
    print(round(sig.t, 3), sig.names[0])
```

```
0.0 frame
0.0 q0
0.01 q0
0.02 q0
0.03 q0
0.033 frame
```

The samples come out sorted by time across the two sensors; at equal times (0.0 here)
the order of the arguments decides, camera first. The merge keeps only one sample per stream in
memory, so it works on live sensors as well as on files.

## 3. Choosing and reshaping channels

A sample with a joint position, a torque and a temperature; the model needs only the
first two, with the torque scaled.

```python
from dense_armor.roles import Signal
from dense_armor.utility.compose import FuncTransformer, Select

sig = Signal(values=[0.5, 2.0, 9.0], names=["q0", "tau0", "temp"], units=["rad", "N m", "C"], t=1.0)
pipe = Select(keys=("q0", "tau0")) | FuncTransformer(lambda v: v * 2.0, keys=("tau0",))
out = pipe.transform_one(sig)
print({k: float(v) for k, v in out.items()}, out.t, out.units)
```

```
{'q0': 0.5, 'tau0': 4.0} 1.0 ['rad', 'N m']
```

`Select` keeps the named channels (`Discard` drops them), `FuncTransformer` applies a
function to some channels; joined with `|` they form a pipeline like any other in the library.
The timestamp and the units of the kept channels travel with the sample.

## 4. A test stream whose concept changes

To test a drift detector you need a stream whose rule changes at a known place. This binary
stream changes its concept once, half way, gradually: near the change each sample comes from the
old or the new concept with a probability that follows a sigmoid.

```python
from dense_armor.utility.datasets import DriftStream

ds = DriftStream(n_samples=20000, n_features=2, n_drifts=1, kind="gradual", spacing=20.0, seed=0)
ds.data()
c = ds.concepts_
print([round(float(c[a:a + 1000].mean()), 2) for a in range(7000, 13000, 1000)])
```

```
[0.07, 0.22, 0.42, 0.61, 0.82, 0.92]
```

A *concept* is the rule that ties the input features to the class. In this generator each
concept is two Gaussian clusters, one per class, in a random direction; the paper's own
`StreamGenerator` builds them from `make_classification` and the Madelon rules (section 3,
page 7). The stream visits one concept, then the next; the change sits at sample 10000. Each
number in the output is the share of samples from the new concept in a window of 1000, from
sample 7000 to 13000; it rises smoothly through the change (0.42 just before, 0.61 just
after).

The weight of the new concept at position $p \in [0, 1]$ of the stream is the sigmoid that
Ksieniewicz and Zyblewski (2020) introduce for all three drift types in section 3.2.1
(page 8). The paper does not give its closed form; the generator uses the standard logistic
form,

$$s(p) = \frac{1}{1 + e^{-\text{spacing}\,(p - d)}}$$

with $p$ the position of the sample as a fraction of the stream, $d = 0.5$ the position of the
drift (half way) and `spacing` how sharp the change is. At $p = d$ the sigmoid is exactly $0.5$;
with `spacing` $= 20$ it goes from $0.12$ at $p = 0.4$ to $0.88$ at $p = 0.6$. The same sigmoid
drives the three drift types of the paper (section 3.2, pages 8–9): `"sudden"` is the limit for
a large `spacing` and swaps the concept at once, `"gradual"` draws the old or the new concept at
each sample with probability `s(p)`, `"incremental"` moves the clusters continuously with the
same weight.

## 5. A robot arm with a fault at a known time

A labelled joint stream from the real URDF of a Panda arm. *Torque* is the turning force the
motors apply at each joint; on a moving arm it is not free, it is fixed by the motion through the
arm's own dynamics. The stream has a periodic motion, torques from the library's inverse
dynamics, sensor noise, and from second 2 a 5 kg payload on the last link.

```python
import numpy as np
from dense_armor.utility.datasets import SyntheticArm

ds = SyntheticArm("test/fixtures/urdf/panda.urdf", period_s=2.0, rate_hz=50.0, n_cycles=2,
                  fault_at_s=2.0, fault="payload", payload_mass=5.0, seed=0)
r0, r1 = [], []
for sig, y in ds.stream():
    q, qd, qdd = ds.trajectory(sig.t)
    r = float(sig["tau_1"] - ds.nominal_torque(q, qd, qdd)[1])
    (r1 if y else r0).append(r)
print(ds.payload_link, round(np.mean(r0), 3), round(np.std(r0), 3), round(np.mean(r1), 2))
```

```
panda_link7 -0.004 0.051 -12.54
```

*The residual* is what is left of the measured torque once the expected one is subtracted:

$$r = \tau_{\text{meas}} - \tau_{\text{model}}.$$

On a healthy arm it is noise: mean -0.004 N·m and standard deviation 0.051 N·m, exactly the
torque noise the stream adds. On a faulty arm it carries the fault: it jumps to a mean of
-12.54 N·m.

The expected torque comes from the arm's inverse dynamics,

$$\tau = M(q)\,\ddot q + C(q, \dot q)\,\dot q + g(q),$$

with $q$ the joint positions, $\dot q$ the joint velocities, $\ddot q$ the joint accelerations,
$M(q)$ the mass matrix (how the arm's inertia couples the joints), $C(q, \dot q)\,\dot q$ the
Coriolis and centrifugal torques, and $g(q)$ the gravity torques. The payload adds its own
gravity torque

$$\tau_{\text{payload}} = m\,g\,J_z(q),$$

where $m$ is the payload mass, $g$ the acceleration of gravity and $J_z(q)$ the vertical row of
the Jacobian of the link that carries the payload (here `panda_link7`, the flange): the vertical
speed of that link for a unit joint velocity. `y` is 0 before the fault and 1 after, so the
stream also scores detectors.

## 6. A model that does not learn the fault

A recursive least-squares model estimates the *residual offset*, the constant value the model
adds to its prediction to best fit the samples it has seen; on a healthy arm it is about zero.
`Protected` asks an anomaly detector before each sample and skips learning on the ones the
detector flags; the detector is a Hampel scorer wrapped in `AnomalyGate`, which also keeps it
from learning flagged samples. The same model without protection is shown next to it for
comparison.

```python
from dense_armor.roles.anomaly_detector import AnomalyGate
from dense_armor.roles.protection import Protected
from dense_armor.utility.anomaly.filters import HampelScorer
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares

ds = SyntheticArm("test/fixtures/urdf/panda.urdf", period_s=2.0, rate_hz=50.0, n_cycles=2,
                  fault_at_s=2.0, fault="payload", payload_mass=5.0, seed=0)
gate = AnomalyGate(HampelScorer(radius=10, n_sigmas=5.0, feature="r"))
prot = Protected(model=RecursiveLeastSquares(feature_keys=["one"]), detector=gate)
bare = RecursiveLeastSquares(feature_keys=["one"])
for sig, y in ds.stream():
    q, qd, qdd = ds.trajectory(sig.t)
    x = {"r": float(sig["tau_1"] - ds.nominal_torque(q, qd, qdd)[1]), "one": 1.0}
    gate.learn_one(x)
    prot.learn_one(x, x["r"])
    bare.learn_one(x, x["r"])
print(round(prot.model.predict_one({"one": 1.0}), 3), round(bare.predict_one({"one": 1.0}), 3))
```

```
-0.003 -6.271
```

After the fault, the protected model still says the healthy offset is -0.003 N·m; the
unprotected one has averaged the fault in and says -6.271 N·m, about half of the -12.5 N·m step
because half of its samples came after the fault. In the test suite the same run flags 99 of the
100 faulty samples and 1 of the 100 healthy ones. A model protected this way can be used to keep
estimating "normal" while the robot is in trouble.

## API reference

::: dense_armor.utility.stream

::: dense_armor.utility.datasets

::: dense_armor.utility.compose

---

## Details

- `iter_csv` uses Python's `csv` module, one row at a time; a list cell `"[a, b, c]"` becomes
  channels `<name>_0, <name>_1, ...` (spaces in the name become underscores); `start` / `stop`
  select a window on `time_col` and stop reading after `stop`.
- `iter_rosbag(path, topics=None)` reads ROS 1 and ROS 2 bags with the pure Python package
  `rosbags` (extra `[ros]`); `sensor_msgs/msg/JointState` becomes channels `q_<joint>`,
  `qd_<joint>`, `tau_<joint>` with the time of the message header.
- `merge_by_time` is a k-way merge with a heap; it assumes each input stream is already in time
  order. `shuffle(stream, buffer_size, seed)` shuffles streams without timestamps in a bounded
  buffer.
- `Casper(path).stream(columns, start_h, stop_h)` reads `right_arm.csv` from a local copy of the
  CASPER dataset (Kayan et al. 2023, section III-A: a UR3e arm logged at about 20 Hz, a 24-hour
  test with about half of the samples anomalous); hours are counted from the first timestamp.
  The code repository of the dataset is MIT (https://github.com/hkayann/CASPER-PerCom), the data
  are CC BY-SA 4.0
  (https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection); the library
  reads the user's copy and redistributes nothing.
- `DriftStream`: each concept is two Gaussian clusters, one per class, placed along a random
  direction; the class of each cluster alternates between concepts, so a model trained on one
  concept is wrong on the next. Each sample uses the drift nearest to its position, so every
  transition is continuous on both sides. `"sudden"` is the limit of the sigmoid for a large
  `spacing` and does not use it; in `"incremental"` the label follows the concept with the larger
  weight. Implemented from the description in Ksieniewicz and Zyblewski (2020), sections 3.2.1
  to 3.2.3 (pages 8–9).
- `SyntheticArm`: sinusoidal joint trajectories at 30 % of each joint's range, inverse dynamics
  with `RigidBodyModel`, compiled once per URDF and process; faults: `"friction"` (viscous
  `b * qd` on one joint) or `"payload"` (a mass on the last link of the longest chain, or on
  `payload_link`).
- `Protected` passes the timestamp only to models that accept it; `HampelScorer.threshold` is
  its `n_sigmas`, which `AnomalyGate` uses to decide.
- Sources: Ksieniewicz, P., Zyblewski, P. (2020), "stream-learn — open-source Python library for
  difficult data stream batch analysis", arXiv:2001.11077. Kayan, H., Rana, O., Burnap, P.,
  Perera, C. (2023), "CASPER: Context-aware anomaly detection system for industrial robotic
  arms".
