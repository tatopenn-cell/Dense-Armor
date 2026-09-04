# Rigid-body dynamics (real URDF, any robot)

Every other module in this package (`rate_limiter`, `cbf_filter`, `trajectory`,
`kinematic_controller`) works at the single-integrator level: you give it a joint velocity,
it gives you back a safe joint velocity. `RigidBodyModel` is different -- it needs an actual
physical robot description (a real URDF file) and gives you real torque-level dynamics.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
model.n
```

`model.n` is the number of real, non-fixed joints the file describes -- read from the file,
not assumed. Point `RigidBodyModel` at any real URDF and it builds the same thing: a mass
matrix, a gravity vector, and everything you need to simulate or control that specific robot.

## The mass matrix

```python
import jax.numpy as jnp

q = jnp.zeros(model.n)
M = model.mass_matrix(q)
```

`M(q)` is the joint-space mass matrix at configuration `q` -- symmetric, positive-definite,
built from the real link masses and inertia tensors in the URDF via the standard Lagrangian
construction (kinetic energy from each link's own center-of-mass Jacobian and inertia tensor,
summed).

## Full dynamics: gravity, Coriolis, and forward simulation

```python
qd = jnp.zeros(model.n)
tau = jnp.zeros(model.n)

g = model.gravity_forces(q)
c_qd = model.bias_forces(q, qd)
qdd = model.forward_dynamics(q, qd, tau)
```

`forward_dynamics` solves `M(q)*qdd + C(q,qd)*qd + g(q) = tau` for `qdd` -- the real joint
acceleration a real torque command `tau` would produce at this state. `gravity_forces` and
`bias_forces` are available on their own too, for building a custom controller (a
gravity-compensating PD term, for instance).

## Kinematics of any link, not only the end effector

```python
p = model.link_position(q, "panda_hand")
J = model.link_jacobian(q, "panda_hand")
```

Any link name from the URDF works -- useful for checking an elbow's position, not only the
end effector.

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Two-step promotion.** Dense-Evolution-Discovery Experiment 61 built this same Euler-Lagrange
dynamics -- `jax.grad`/`jax.jvp` on the kinetic/potential energy, not hand-derived Christoffel
symbols -- but with every mass, inertia tensor, and joint origin hand-transcribed from one
specific Kinova Gen3's URDF. That was the explicit reason it wasn't promoted at the time: every
other module here is generic across any joint array, that one worked for exactly one robot.
Experiment 62 replaced the hardcoded tables with a real parser (`xml.etree.ElementTree`, no new
dependency) and re-validated from scratch.

**Validated on three independent real robots**, not one:

| robot | source | DoF | joint types |
|---|---|---|---|
| Kinova Gen3 7-DoF | the same URDF Kurtz, Wensing & Lin (2021, arXiv:2109.13349) use | 7 | all revolute |
| Kinova Gen3 6-DoF | github.com/vincekurtz/kinova_drake -- a structurally different chain | 6 | all revolute |
| Franka Emika Panda | bulletphysics/bullet3's real pybullet data -- a different manufacturer | 9 | 7 revolute + 2 prismatic |

Cross-checked against Experiment 61's own hardcoded numbers on the Gen3 7-DoF (mass matrix and
gravity forces match to machine precision, 1e-16). On all three: mass matrix
symmetric/positive-definite at 20 random configurations, and free (torque-free) dynamics
conserve energy with the correct 4th-order RK4 convergence as the integration step shrinks --
the Panda's converges tighter (rel. drift 5.9e-7 -> 6.0e-11 -> 5.5e-15) since its published
inertia tensors are simpler placeholder values, not a difference in correctness.

**Mimic joints are not modeled.** The Panda's real URDF ties its two finger joints together
with a `<mimic>` tag; this parser doesn't read it, so each non-fixed joint (including both
fingers) is treated as an independent DOF -- a real, disclosed limitation, not a silent one.

**Reproducing this**: `pytest test/test_urdf_dynamics.py`.
