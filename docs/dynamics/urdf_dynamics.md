# Rigid-body dynamics (real URDF, any robot)

The control modules on [Control](../control/index.md) treat the robot as a
single-integrator plant: they take a joint velocity, they return a joint velocity. That
is the level of the driver's API and it is enough for bounding commands, but it is not
the actual physics of the arm.

`RigidBodyModel` is the module that reads the physics. You point it at a URDF file and
it gives you the real mass matrix, gravity vector, Coriolis terms, forward dynamics,
and the kinematics of any link in the chain. Everything downstream — from gravity
compensation to the CBF controllers that need the Jacobian — is built on this.

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
print(model.n)
```

```
9
```

`model.n` is the number of real, non-fixed joints the file describes — read from the
file, not assumed. For a Franka Panda it is 7 (the arm) or 9 (arm plus the two gripper
fingers), depending on which URDF you load. The file used here has the hand, so it is 9.

## 1. Joint limits

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
print(model.q_min[:3], model.q_max[:3], model.qd_max[:3])
```

```
[-2.9671 -1.8326 -2.9671] [2.9671 1.8326 2.9671] [2.175 2.175 2.175]
```

Each joint's real position and velocity limits, from the URDF's own `<limit>` tag, or
`±inf` wherever the URDF declares none. The
[passivity + singularity-CBF controller](passivity_cbf_controller.md) uses these to keep
every commanded configuration inside its box; if the URDF does not declare a limit, no
row is added to the constraint.

## 2. The mass matrix

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
M = model.mass_matrix(jnp.zeros(model.n))
print(M.shape, bool(jnp.allclose(M, M.T)), float(jnp.linalg.eigvalsh(M).min()) > 0)
```

```
(9, 9) True True
```

`M(q)` is the joint-space mass matrix at configuration `q`. It is symmetric and
positive-definite by construction, built from the link masses and inertia tensors in the
file via the standard Lagrangian construction: the kinetic energy is the sum over links
of `½ m vᵀv + ½ ωᵀ I ω` (each link's mass `m` and inertia tensor `I`, with `v` and `ω` the
velocity of its centre of mass and its angular velocity, both linear in `q̇`).
The `jax.grad` / `jax.jvp` machinery does the differentiation; no Christoffel symbols
are hand-derived.

Both properties — symmetric and positive-definite — hold without any explicit
symmetrization or fix-ups: they come out of the derivation, at every configuration.

## 3. Gravity and Coriolis terms

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
q, qd, q_ref = jnp.zeros(model.n), jnp.zeros(model.n), jnp.full(model.n, 0.1)
tau = model.gravity_forces(q) + 50.0 * (q_ref - q) - 10.0 * qd
print(jnp.round(tau[:4], 3))
```

```
[5.    2.174 5.    1.483]
```

`gravity_forces(q)` is the gravity torque at each joint. `bias_forces(q, qd)` is the
`C(q, qd) qd` term — Coriolis and centrifugal together, as they always are in the
Lagrangian form. The block above builds a **gravity-compensating PD**: the gravity term
plus a proportional term toward a reference plus a derivative term on velocity.

Without gravity compensation, a PD controller on the joint positions needs a large
`kp` to hold the arm against gravity, and that `kp` produces overshoot when the arm
moves. With it, the PD is a small correction on top of a physical baseline.

## 4. Forward dynamics

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
q, qd = jnp.zeros(model.n), jnp.zeros(model.n)
qdd = model.forward_dynamics(q, qd, model.gravity_forces(q))
print(float(jnp.abs(qdd).max()))
```

```
0.0
```

`forward_dynamics` solves `M(q) qdd + C(q, qd) qd + g(q) = tau` for `qdd` — the joint
acceleration that a torque command `tau` would produce at this state. Feeding it exactly
`gravity_forces(q)` at zero velocity yields zero acceleration: the model is in
equilibrium. With `tau = 0` it is the free (torque-free) dynamics, useful for verifying
the model: the total energy `½ q̇ᵀ M(q) q̇ + U(q)` should be conserved.

With RK4 and `tau = 0`, the relative energy drift decreases as `dt⁴` as the integration
step shrinks: `5.9e-7 → 6.0e-11 → 5.5e-15` on the Panda for `dt = 10⁻², 10⁻³, 10⁻⁴ s`.
The convergence order is what confirms the equations are correct; a wrong sign or a
transposed inertia tensor would give a much worse ratio.

## 5. Kinematics of any link

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
q = jnp.zeros(model.n)
p, J = model.link_position(q, "panda_hand"), model.link_jacobian(q, "panda_hand")
print(jnp.round(p, 3), J.shape)
```

```
[ 0.088 -0.     0.926] (3, 9)
```

Any link name from the URDF works — useful for checking the elbow's position, or for
running a CBF on a link that is not the end effector. `link_jacobian` returns the 3×N
position Jacobian.

For a link's **full pose** (position and orientation) and its 6×N **spatial Jacobian**
— needed by the [full 6-DoF controller](six_dof_pbc_cbf_controller.md) — use
`link_pose` and `link_spatial_jacobian`:

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
q = jnp.zeros(model.n)
p, R = model.link_pose(q, "panda_hand")
print(R.shape, model.link_spatial_jacobian(q, "panda_hand").shape)
```

```
(3, 3) (6, 9)
```

`Jspatial` has shape `(6, N)`: the top three rows are angular, the bottom three are
linear. The order matches `Twist = [ω; v]`, the standard spatial-velocity convention.

## 6. Xacro files and mimic joints

A `.xacro` path is expanded automatically. A joint's `<mimic>` tag (a gripper's two
fingers tied together) is respected: `model.n` counts real independent DOF, not raw
`<joint>` tags. Both are their own pages:
[xacro support](xacro_support.md), [mimic joints](mimic_joints.md).

## API reference

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Two-step promotion.** Dense-Evolution-Discovery Experiment 61 built the same
Euler–Lagrange dynamics — `jax.grad` / `jax.jvp` on the kinetic and potential energy,
not hand-derived Christoffel symbols — but with every mass, inertia tensor, and joint
origin hand-transcribed from one specific Kinova Gen3's URDF. That was the explicit
reason it was not promoted at the time: every other module in the package is generic
across any joint array, and that one worked for exactly one robot.

Experiment 62 replaced the hardcoded tables with a real parser (`xml.etree.ElementTree`,
no new dependency) and re-validated from scratch.

**Validated on three independent real robots**, not one:

| robot | source | DoF | joint types |
|---|---|---|---|
| Kinova Gen3 7-DoF | the same URDF Kurtz, Wensing & Lin (2021, [arXiv:2109.13349](https://arxiv.org/abs/2109.13349)) use | 7 | all revolute |
| Kinova Gen3 6-DoF | `github.com/vincekurtz/kinova_drake` — a structurally different chain | 6 | all revolute |
| Franka Emika Panda | `bulletphysics/bullet3`'s real pybullet data — a different manufacturer | 9 | 7 revolute + 2 prismatic |

Cross-checked against Experiment 61's own hardcoded numbers on the Gen3 7-DoF (mass
matrix and gravity forces match to machine precision, 1e-16). On all three: mass matrix
symmetric / positive-definite at 20 random configurations, and free dynamics conserve
energy with the correct 4th-order RK4 convergence as the integration step shrinks — the
Panda's converges tighter (relative drift `5.9e-7 → 6.0e-11 → 5.5e-15`) since its
published inertia tensors are simpler placeholder values, not a difference in
correctness.

**Reproducing this**: `pytest test/test_urdf_dynamics.py`.

**See also**: [Passivity + singularity-CBF controller](passivity_cbf_controller.md) —
the task-space controller built on this model.
[Online dynamics](../learn/online_dynamics.md) — the module that learns what the URDF
does not model (payload, friction, wear).
