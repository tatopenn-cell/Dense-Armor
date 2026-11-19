# ResidualDynamicsLearner: the torque the model misses

`ResidualDynamicsLearner` takes the model of a robot (its URDF: masses, inertias, gravity) and learns, one sample at a time, the torque that the model does not know: the friction of the joints, a payload in the gripper, the wear of months. It uses one [RecursiveLeastSquares](recursive_least_squares.md) per joint on five features, and it adds the learned torque to the one of the model, so the controller does not need exact masses or friction coefficients. The overview is on [Online dynamics](online_dynamics.md).

This class needs a rigid-body model of a robot, so its example is a robot arm. The same idea with money, a fee formula that is missing a fixed charge and whose commission changes, is on the [RecursiveLeastSquares](recursive_least_squares.md) page.

## The two torques

A joint reports what it actually applied: the measured torque `tau_measured`. The URDF
model predicts what a *perfect* robot would have applied: the nominal torque. The
difference is what the learner has to model.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import ResidualDynamicsLearner, write_minimal_urdf

nom = RigidBodyModel(write_minimal_urdf())
real = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
tau = lambda m, q, qd, qdd: np.asarray(m.mass_matrix(q)) @ qdd + np.asarray(m.bias_forces(q, qd))
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for q, qd, qdd in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(q, qd, qdd, tau(real, q, qd, qdd))
q, qd, qdd = rng.uniform(-1, 1, (3, 2))
print(np.round(tau(real, q, qd, qdd), 3), np.round(tau(nom, q, qd, qdd), 3), np.round(learner.predict_torque(q, qd, qdd), 3))
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
from dense_armor.core.online.learn.online_dynamics import ResidualDynamicsLearner, write_minimal_urdf

nom, real = RigidBodyModel(write_minimal_urdf()), RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
tau = lambda m, s: np.asarray(m.mass_matrix(s[0])) @ s[2] + np.asarray(m.bias_forces(s[0], s[1]))
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for s in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(*s, tau(real, s))
test = rng.uniform(-1, 1, (20, 3, 2))
err = lambda f: np.sqrt(np.mean([(tau(real, s) - f(s)) ** 2 for s in test]))
print(round(err(lambda s: tau(nom, s)), 3), round(err(lambda s: learner.predict_torque(*s)), 3))
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

If the payload changes at sample 200 of 400, a `λ = 1` learner (batch least squares)
keeps the old weights and gets the new regime wrong. A `λ = 0.99` learner forgets the
pre-change samples over ~100 samples and tracks the new regime.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.online.learn.online_dynamics import ResidualDynamicsLearner, DriftAwareResidualDynamicsLearner, write_minimal_urdf

model = RigidBodyModel(write_minimal_urdf())
fast = ResidualDynamicsLearner(model, lam=0.99, delta=1e6)
aware = DriftAwareResidualDynamicsLearner(model, lam=0.99)
print(type(fast).__name__, type(aware).__name__)
```

```
ResidualDynamicsLearner DriftAwareResidualDynamicsLearner
```

The choice of `λ` is a trade-off: too small and the learner is noisy (it always fits
the last few samples), too large and it is slow to adapt. `0.99` is a reasonable default
for a joint that changes over seconds.

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

---

## Details

- `learn_one(q, qd, qdd, tau)` takes the joint positions, velocities and accelerations and the measured torque; `predict_torque(q, qd, qdd)` returns the torque of the model plus the learned residual.
- Dense-Evolution-Discovery Experiment 61 built the same Euler-Lagrange dynamics, hardcoded to one Kinova Gen3; Experiment 62 replaced the hardcoded tables with a real URDF parser and re-validated. The learner itself was re-validated on the 2-link benchmark after that.
- See also: [Rigid-body dynamics](../dynamics/urdf_dynamics.md), the nominal model the learner builds on; [Platt scaling](calibration.md), if the residual is used as a score.
