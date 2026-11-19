# SyntheticArm: a robot arm with a fault at a known time

`SyntheticArm` makes a labelled joint stream from the URDF of a robot arm, with a fault that starts at a known time. It is a robot dataset by nature, so its examples are robot arms. The overview of the section is on [Datasets](index.md).

## A robot arm with a fault at a known time

A labelled joint stream from the real URDF of a Panda arm. *Torque* is the turning force the
motors apply at each joint; on a moving arm it is not free, it is fixed by the motion through the
arm's own dynamics. The stream has a periodic motion, torques from the library's inverse
dynamics, sensor noise, and from second 2 a 5 kg payload on the last link.

```python
import numpy as np
from dense_armor.core.signals.datasets import SyntheticArm

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

## A model that does not learn the fault

A recursive least-squares model estimates the *residual offset*, the constant value the model
adds to its prediction to best fit the samples it has seen; on a healthy arm it is about zero.
`Protected` asks an anomaly detector before each sample and skips learning on the ones the
detector flags; the detector is a Hampel scorer wrapped in `AnomalyGate`, which also keeps it
from learning flagged samples. The same model without protection is shown next to it for
comparison.

```python
from dense_armor.roles.anomaly_detector import AnomalyGate
from dense_armor.roles.protection import Protected
from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.signals.datasets import SyntheticArm
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

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

::: dense_armor.core.signals.datasets.synthetic_arm

---

## Details

- `SyntheticArm`: sinusoidal joint trajectories at 30 % of each joint's range, inverse dynamics
  with `RigidBodyModel`, compiled once per URDF and process; faults: `"friction"` (viscous
  `b * qd` on one joint) or `"payload"` (a mass on the last link of the longest chain, or on
  `payload_link`).
- `Protected` passes the timestamp only to models that accept it; `HampelScorer.threshold` is
  its `n_sigmas`, which `AnomalyGate` uses to decide.
- `Protected` passes the timestamp only to models that accept it; `HampelScorer.threshold` is
  its `n_sigmas`, which `AnomalyGate` uses to decide.
