# ResidualDynamicsLearner: the torque the model misses

`ResidualDynamicsLearner` takes the model of a robot (its URDF: masses, inertias, gravity) and learns, one sample at a time, the torque that the model does not know: the friction of the joints, a payload in the gripper, the wear of months. It uses one [RecursiveLeastSquares](recursive_least_squares.md) per joint on five features, and it adds the learned torque to the one of the model, so the controller does not need exact masses or friction coefficients. The overview is on [Online dynamics](online_dynamics.md).

This class needs a rigid-body model of a robot, so its example is a robot arm. The same idea with money, a fee formula that is missing a fixed charge and whose commission changes, is on the [RecursiveLeastSquares](recursive_least_squares.md) page.

## Gamestory

London Theater, deep night. Second night of the show. The young puppeteer has found the maker's official instruction manual inside the puppet's box.

Everything is written down to the letter in the manual: "To lift the arm, pull the string with strength 10". The puppeteer trusts it and pulls with exact strength 10, but on stage a disaster happens: the arm barely rises and gets stuck halfway! Why? The maker had written the manual by testing a brand-new puppet in a workshop. He couldn't know that the wooden hinges had gotten rusty and that, for tonight's scene, the puppeteer had put a super-heavy apple in the puppet's hand!

So the puppeteer calls his loyal assistant with pen and little sheet back behind the curtains to put the ResidualDynamicsLearner method into action:

- Instead of throwing away the official manual, the puppeteer decides to use it as a base: "You just watch what is missing!" he tells the assistant. The manual says to pull 10, but to actually make the arm rise it takes 16. Those 6 extra points of strength are the "leftover" part the manual doesn't know about.
- The assistant prepares a small sheet for every single hinge of the arm (one for the shoulder and one for the elbow) and, move after move, jots down only that extra strength by watching 5 exact clues: how fast the arm snaps, at what speed it moves, which way it turns, how much the apple pulls downward, and a tiny fixed friction of the wood.
- On every tug, the puppeteer does a lightning addition: he takes the number written in the official manual and adds the extra strength just written on the assistant's little sheet. Within seconds the puppet's arm moves as light and accurate as a feather!

In the end, what is ResidualDynamicsLearner? It is a smart method that doesn't redo all the math from scratch, but pairs the official manual with a small helper for each hinge that learns only the "extra" strength needed to overcome weight, friction, and real-life surprises!

![The torque error of a 2-link arm with a payload that the URDF does not have: the URDF alone is at 0.497 N·m; the URDF plus ResidualDynamicsLearner is worse for the first ten samples (0.65 to 0.73) and then falls to 0.32 after 12 samples, 0.22 after 40 and 0.21 after 100.](../assets/learn/residual_learning.png)

The picture follows the learner on a 2-link arm that carries a 0.5 kg payload that the URDF does not have. The error is measured on 20 held-out moves: 0.497 N·m for the URDF alone (orange). For the first ten samples the learner is **worse** than the URDF alone, between 0.65 and 0.73: its first weights, with a very large initial covariance, overshoot. After 12 samples it is better (0.32), after 40 it is at 0.22 and after 100 at 0.21.

## The two torques

A joint reports what it actually applied: the measured torque `tau_measured`. The URDF
model predicts what a *perfect* robot would have applied: the nominal torque. The
difference is what the learner has to model.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import (
    ResidualDynamicsLearner,
    write_minimal_urdf,
)

nom = RigidBodyModel(write_minimal_urdf())
real = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
tau = lambda m, q, qd, qdd: (
    np.asarray(m.mass_matrix(q)) @ qdd + np.asarray(m.bias_forces(q, qd))
)
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for q, qd, qdd in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(q, qd, qdd, tau(real, q, qd, qdd))
q, qd, qdd = rng.uniform(-1, 1, (3, 2))
print(
    np.round(tau(real, q, qd, qdd), 3),
    np.round(tau(nom, q, qd, qdd), 3),
    np.round(learner.predict_torque(q, qd, qdd), 3),
)
```

```
[1.496 0.558] [0.85  0.287] [1.422 0.478]
```

Two copies of the same 2-link arm, one with a 0.5 kg payload in the gripper (the "real" robot)
and one without (the URDF the controller knows). After 40 random moves the learner, starting
from the payload-free URDF, predicts torques of 1.422 and 0.478 N·m where the real arm needs 1.496
and 0.558; the URDF alone says 0.85 and 0.287. `write_minimal_urdf` writes this small test robot
to a file, so the example runs anywhere.

## The five features per joint

For each joint `j`, the learner regresses the residual torque on five features:

```
[qdd_j, qd_j, sign(qd_j), g_j, 1]
```

- `qdd_j` — joint acceleration; captures inertia-like effects (a payload adds mass
  that scales with acceleration).
- `qd_j` — joint velocity; captures viscous friction.
- `sign(qd_j)` — sign of velocity; captures Coulomb friction (a constant torque
  opposing motion, independent of speed).
- `g_j` — gravity torque from the URDF at this configuration; captures a constant
  payload offset.
- `1` — a constant bias.

The five together are the minimal basis that captures a first-order residual on a
revolute joint. Everything else (higher-order friction, stiction, thermal effects) is
either negligible or requires more data than a single joint gives.

The target is the measured torque minus the nominal

```
tau_residual = tau_measured − M(q) qdd − C(q, qd) qd − g(q)
```

so the learner only has to model what the URDF does not.

## On a simulated arm with a payload

A 2-link URDF, true torques = URDF torques + payload + viscous friction. The learner
starts from the URDF prediction and adds the residual. Test check: the learner's
predicted torques have lower RMSE than the nominal URDF prediction on held-out data.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import (
    ResidualDynamicsLearner,
    write_minimal_urdf,
)

nom, real = (
    RigidBodyModel(write_minimal_urdf()),
    RigidBodyModel(write_minimal_urdf(payload_mass=0.5)),
)
tau = lambda m, s: (
    np.asarray(m.mass_matrix(s[0])) @ s[2]
    + np.asarray(m.bias_forces(s[0], s[1]))
)
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for s in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(*s, tau(real, s))
test = rng.uniform(-1, 1, (20, 3, 2))
err = lambda f: np.sqrt(np.mean([(tau(real, s) - f(s)) ** 2 for s in test]))
print(
    round(err(lambda s: tau(nom, s)), 3),
    round(err(lambda s: learner.predict_torque(*s)), 3),
)
```

```
0.403 0.193
```

On 20 new moves the torque error (RMSE, N·m) is 0.403 for the URDF alone and 0.193 for the
learner after 40 samples: half the error.

The nominal model misses the payload; the learner picks it up in the five weights
(the `qdd` coefficient absorbs the payload's inertia, the `1` absorbs the constant
offset, the `sign(qd)` coefficient absorbs Coulomb friction).

## When a parameter changes mid-run

The payload of a 2-link arm is 0.5 kg for the first 200 samples and 1.5 kg for the next 200. Four learners see the same 400 samples, each learned after it is used: `lam = 1` (it weighs every sample the same), `lam = 0.99`, `lam = 0.9` and the drift-aware one. After 200, 250, 300 and 400 samples the torque error (RMSE, in N·m) of each learner is measured on 30 fixed states against the torque of the arm that holds the payload of that moment; the last column is the model alone.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import (
    DriftAwareResidualDynamicsLearner,
    ResidualDynamicsLearner,
    write_minimal_urdf,
)


def torque(m, s):
    return np.asarray(m.mass_matrix(s[0])) @ s[2] + np.asarray(
        m.bias_forces(s[0], s[1])
    )


nom = RigidBodyModel(write_minimal_urdf())
light = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
heavy = RigidBodyModel(write_minimal_urdf(payload_mass=1.5))
rng = np.random.default_rng(0)
states = rng.uniform(-1, 1, (400, 3, 2))
test = rng.uniform(-1, 1, (30, 3, 2))
learners = {
    "lam=1": ResidualDynamicsLearner(nom, lam=1.0),
    "lam=0.99": ResidualDynamicsLearner(nom, lam=0.99),
    "lam=0.9": ResidualDynamicsLearner(nom, lam=0.9),
    "drift-aware": DriftAwareResidualDynamicsLearner(nom, lam=0.99),
}


def error(predict, plant):
    sq = [(torque(plant, s) - predict(*s)) ** 2 for s in test]
    return round(float(np.sqrt(np.mean(sq))), 3)


for i, s in enumerate(states):
    plant = light if i < 200 else heavy
    for learner in learners.values():
        learner.learn_one(*s, torque(plant, s))
    if i + 1 in (200, 250, 300, 400):
        row = {k: error(v.predict_torque, plant) for k, v in learners.items()}
        row["model alone"] = error(lambda *a: torque(nom, a), plant)
        print(i + 1, row)
```

```
200 {'lam=1': 0.255, 'lam=0.99': 0.252, 'lam=0.9': 0.254, 'drift-aware': 0.251, 'model alone': 0.503}
250 {'lam=1': 1.031, 'lam=0.99': 0.923, 'lam=0.9': 0.796, 'drift-aware': 0.793, 'model alone': 1.508}
300 {'lam=1': 0.953, 'lam=0.99': 0.818, 'lam=0.9': 0.837, 'drift-aware': 0.761, 'model alone': 1.508}
400 {'lam=1': 0.848, 'lam=0.99': 0.758, 'lam=0.9': 0.838, 'drift-aware': 0.76, 'model alone': 1.508}
```

Before the change all four learners are at 0.25 N·m, half of the 0.50 of the model alone: the learner does not bring the error to zero. After the change the model alone is at 1.51. Fifty samples after it, `lam = 1` is at 1.03, `lam = 0.99` at 0.92, `lam = 0.9` at 0.80 and the drift-aware learner at 0.79. After 200 samples `lam = 1` is at 0.85, `lam = 0.99` at 0.76, `lam = 0.9` at 0.84 and the drift-aware learner at 0.76.

So forgetting helps a little (0.76 against 0.85 after 200 samples for `lam = 0.99`, and the drift-aware learner is the lowest after 100 samples), but no learner comes back to the 0.25 that it had before the change: all end at three times that value. This test does not show that `lam = 0.99` tracks the new payload, and it does not show a clear winner among the three forgetting learners; it is one run, one payload change and one set of 30 test states. Why the error stays above its earlier level was not investigated.

## Performance note

`RigidBodyModel` methods are pure-Python recursive walks over dicts and retrace the JAX
graph on every call, so `jax.jit` on the model does **not** cache across calls the way
it does for a pure function. In a control loop, patch the model once at construction:

```python
import jax
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import write_minimal_urdf

model = RigidBodyModel(write_minimal_urdf())
model.mass_matrix = jax.jit(model.mass_matrix)
model.bias_forces = jax.jit(model.bias_forces)
model.gravity_forces = jax.jit(model.gravity_forces)
print(model.n)
```

```
2
```

Without this, the learner's per-sample cost is in the millisecond range rather than
tens of microseconds. For a 2-DoF arm the difference is roughly 1.9 ms vs 2.7 ms per
sample depending on JAX cache state; the bottleneck is `forward_kinematics`, not the
RLS update itself.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `model` | the `RigidBodyModel` of the robot, from [`urdf_dynamics`](../dynamics/urdf_dynamics.md) | (required) |
| `lam` | the forgetting factor of every per-joint RLS | 1.0 |
| `delta` | the initial covariance scale of every per-joint RLS | 1e6 |

## API reference

::: dense_armor.core.online.learn.online_dynamics.ResidualDynamicsLearner
    options:
      docstring_style: numpy

---

## Details

- `learn_one(q, qd, qdd, tau)` takes the joint positions, velocities and accelerations and the measured torque; `predict_torque(q, qd, qdd)` returns the torque of the model plus the learned residual.
- Dense-Evolution-Discovery Experiment 61 built the same Euler-Lagrange dynamics, hardcoded to one Kinova Gen3; Experiment 62 replaced the hardcoded tables with a real URDF parser and re-validated. The learner itself was re-validated on the 2-link benchmark after that.
- See also: [Rigid-body dynamics](../dynamics/urdf_dynamics.md), the nominal model the learner builds on; [Platt scaling](calibration.md), if the residual is used as a score.
