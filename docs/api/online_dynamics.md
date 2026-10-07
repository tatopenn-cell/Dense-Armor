# Online dynamics (residual torque learning)

Learns the gap between a URDF-nominal rigid-body model and the real robot's
measured joint torques, one sample at a time, using one recursive least
squares per joint on a five-feature basis `[qdd, qd, sign(qd), g, 1]`. The
nominal model comes from [`urdf_dynamics`](urdf_dynamics.md); the learner
adds a residual torque on top so the controller does not need exact masses
or friction coefficients. A CUSUM/Hampel-triggered drift-aware variant is
also provided, kept as an API but not recommended as the default (see the
honest result below).

```python
import jax
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.online_dynamics import ResidualDynamicsLearner

model = RigidBodyModel("robot.urdf")
model.mass_matrix = jax.jit(model.mass_matrix)      # see performance note
model.bias_forces = jax.jit(model.bias_forces)
model.gravity_forces = jax.jit(model.gravity_forces)

learner = ResidualDynamicsLearner(model, lam=1.0, delta=1e6)
for q, qd, qdd, tau_measured in stream:
    tau_hat = learner.predict_torque(q, qd, qdd)
    learner.learn_one(q, qd, qdd, tau_measured)
```

The five features per joint are the minimal basis that captures a first-order
residual: an inertia-like term scaling with `qdd` (payload), viscous friction
scaling with `qd`, Coulomb friction scaling with `sign(qd)`, a gravity-residual
term (constant payload offset) and a free constant. The target is the measured
torque minus the nominal `M(q) qdd + C(q,qd) qd + g(q)`, so the learner only
has to model what the URDF does not: payload, friction, wear.

## What the tests check

- With `lam = 1` and a large `delta`, the RLS weights equal `numpy.linalg.lstsq` on the same
  samples; with `lam < 1` the RLS tracks a parameter that changes halfway through the stream.
- On a simulated 2-link arm whose true torques add a payload and viscous friction to the URDF
  model, `ResidualDynamicsLearner` predicts the torques with a lower RMSE than the nominal model.
- The drift-aware variant runs and stays finite; whether it re-adapts faster than the plain
  learner depends on the data and on the detector's parameters, so it is offered as an option,
  not as the default.

## Performance note

`RigidBodyModel` methods are pure-Python recursive walks over dicts and
retrace the JAX graph on every call, so `jax.jit` on the model does not
cache across calls the way it does for a pure function. In a control loop,
patch the model once at construction — `model.mass_matrix = jax.jit(model.mass_matrix)`
and likewise for `bias_forces` and `gravity_forces` — or the learner's
per-sample cost is in the millisecond range rather than tens of microseconds.
For a 2-DoF arm the difference is roughly 1.9 ms vs 2.7 ms per sample
depending on JAX cache state; the bottleneck is `forward_kinematics`, not
the RLS update.

## Parameters

- `lam` (default 1.0): forgetting factor in `(0, 1]`. `lam = 1.0` weights all
  past samples equally and matches batch least squares; smaller values weight
  recent samples more and let the model track a parameter that changes.
- `delta` (default 1e6): initial covariance scale `P_0 = delta * I`. Larger
  values mean less prior confidence in the zero initial weight.
- `feature_keys` (default `None`): explicit feature order for `RecursiveLeastSquares`.
  `None` sorts the keys of the first dict seen, so the result does not depend
  on dict ordering.
- `ResidualDynamicsLearner` uses the same `lam` and `delta` for every per-joint
  RLS.

## References

- Ljung, L., Soderstrom, T. (1983). *Theory and Practice of Recursive
  Identification*. MIT Press. (RLS with exponential forgetting.)
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41, 100-114.
  (Original CUSUM.)
- Hampel, F. R. (1974). The influence curve and its role in robust estimation.
  *JASA* 69(346), 383-393. (Hampel trigger.)

::: dense_armor.dynamics.online_dynamics
