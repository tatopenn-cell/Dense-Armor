# Passivity + singularity-CBF controller

A robot arm moving at full speed is easy to control when everything goes well. The hard
part is the moment when the arm is **fully extended**: at that configuration, the
Jacobian is rank-deficient, the arm cannot move in certain task-space directions no
matter what the joints do, and the joint torques needed to keep the tracking error
bounded blow up to infinity. A naive controller keeps asking for more torque and the
arm "snaps" through the singularity in a way that is not safe on a real robot.

This module is a **task-space controller** that guarantees two things at once:

- **Passivity** — the tracking error does not inject energy into the closed loop. This
  is what makes the controller safe around humans and delicate objects.
- **Singularity avoidance** — the manipulability index stays above a user-chosen floor,
  so the arm never reaches a configuration from which it cannot move in the task space.

Both properties hold simultaneously, and the controller never becomes infeasible.

## The robot

Any robot loaded from its URDF. Here a Franka Panda, the same robot used everywhere in
this section.

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
```

The controller is called `solve_control_qp` because it is a **quadratic program**
solver: at every control tick, it takes the current state, the desired task-space
target, and returns the joint torque that meets both guarantees with minimum deviation
from the standard passivity-based control command.

## 1. What "manipulability" means

The **manipulability index** at a configuration `q` is

```
μ(q) = √( det( J(q) J(q)ᵀ ) )
```

where `J(q)` is the task-space Jacobian at `q`. When the arm can move freely in the
task space, `μ > 0`; when the arm is at a singularity (fully extended, or in a wrist
singularity), `μ = 0`. The index is a smooth function of `q` and gives a single number
for "how far from singularity we are right now".

The controller enforces `μ(q) ≥ ε` with a user-chosen floor `ε` (typically `0.03`).

## 2. The controller call

```python
q = jnp.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5])
qd = jnp.zeros(model.n)
p_des = jnp.array([0.5, 0.0, 0.5])
pd_des = jnp.zeros(3)
pdd_des = jnp.zeros(3)

qdd, tau, mu, h = solve_control_qp(
    model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03,
)
```

`solve_control_qp` returns four values:

- `qdd` — the joint acceleration command.
- `tau` — the joint torque command.
- `mu` — the manipulability index at the current configuration.
- `h` — the barrier function value at the current configuration, `h(q) = μ(q) − ε`.
  Positive means "inside the safe set"; the controller keeps it non-negative.

The first argument `model` is the `RigidBodyModel`. The second `"panda_hand"` is the
**name of the link being tracked** — the same name the URDF uses. `p_des`, `pd_des`,
`pdd_des` are the desired position, velocity, and acceleration of that link at this
instant (typically from the [trajectory generator](../control/trajectory.md)).
`eps` is the manipulability floor.

## 3. What the QP actually solves

At each call, the QP is:

```
minimize_qdd   ‖ M(q) qdd + C(q, qd) qd + g(q) − tau_pd ‖²
subject to     V̇(q, q̇, qdd) ≤ 0                    (passivity)
               μ̇(q, q̇, qdd) + α · (μ(q) − ε) ≥ 0   (singularity CBF)
               q_min ≤ q + qd · dt ≤ q_max         (joint limits, if declared)
```

### Symbols

- `M(q)`, `C(q, qd)`, `g(q)` — the mass matrix, Coriolis term, gravity vector from
  the [URDF dynamics](urdf_dynamics.md).
- `tau_pd` — the nominal task-space PD command: `Kp (p_des − p) + Kd (pd_des − pd)`,
  mapped through the transposed Jacobian `Jᵀ`. The QP finds the `qdd` that realizes
  this torque **as closely as possible** while still satisfying the constraints.
- `V` — the tracking-error storage function (a Lyapunov-like function). Passivity
  means `V̇ ≤ 0`, computed from the current state and the requested `qdd`.
- `μ` — the manipulability index above.
- `α` — the CBF gain. Larger `α` pushes the arm away from the floor more aggressively;
  smaller `α` allows getting closer to it.
- `q_min`, `q_max` — the joint limits from the URDF's own `<limit>` tags. Rows are
  added **only** for joints that declare a real limit; for unlimited joints no row
  is added.

The four constraint families are all affine in `qdd`, so this is a small QP. On a
desktop it solves in well under a millisecond; the module uses OSQP
([Stellato et al. 2020](https://doi.org/10.1007/s12532-020-00179-2)).

## 4. Hand case: joint limit CBF in action

Franka Panda, joint `joint4`, real range `[−3.1416, 0.0]`. Put the joint right at its
bound with velocity driving past it:

```
q[3] = 0.0        (at the upper edge of the range)
qd[3] = 1.0       (moving further up, past the limit)
```

The nominal (unconstrained) PD command would produce `qdd[3] = −205.8`. Without the
CBF row, the QP would return this value, and the joint would immediately overshoot its
hard limit. With the joint-limit CBF active, the constraint box for `qdd[3]` at this
state is `[−7.175, −4.999]` (computed from the URDF limit, the current position, and
the CBF recovery rate). The solver returns `qdd[3] = −7.175` — the box's own edge.

The motor will not be commanded to violate the joint limit, no matter what the PD
command wants.

## 5. On three real robots

At each robot's own true kinematic singularity, the minimum manipulability over a
200-step trajectory under the controller:

| robot | link | min(μ), no CBF | min(μ), CBF ε=0.03 |
|---|---|---|---|
| Kinova Gen3 7-DoF | `end_effector_link` | 0.00003 | **0.02947** |
| Kinova Gen3 6-DoF | `bracelet_with_vision_link` | 0.00006 | **0.02995** |
| Franka Panda | `panda_hand` | 0.00296 | **0.02997** |

Each row drives the named link toward that robot's own true kinematic singularity.
Without the CBF, the controller reaches it: `μ` drops to near zero. With the CBF,
`μ` stays within 0.1–1.8 % of the declared floor. The controller does not stop the
arm from approaching the singular configuration (that would be over-restrictive);
it stops the arm from *being* singular.

## 6. What "passivity" means here

The tracking-error storage function `V` is a scalar measuring how far the joint is
from the reference, weighted by the mass matrix. The passivity constraint is `V̇ ≤ 0`
at every control tick: the controller is not allowed to add energy to the tracking
loop. On a real robot this is what makes the arm safe around humans — an operator
pushing on the arm cannot be surprised by the arm pushing back harder than it was
pushed.

Without the passivity constraint, the singularity CBF alone would still prevent
singularities but would allow configurations where the arm is doing work on the
tracking error, which is exactly the failure mode passivity-based control was
designed to avoid.

## 7. A real bug, found and fixed

OSQP can report the passivity + CBF QP **jointly infeasible**. The passivity
constraint's coefficients go numerically near-zero exactly when tracking is already
good (a mathematically inert row on a state where the error is already zero), which
combined with a tight CBF margin occasionally leaves no feasible point under OSQP's
default tolerances. In that case OSQP returns its **infeasibility certificate** — a
vector with norm in the billions — as if it were a real solution.

The fix: check the solver status; on infeasibility, drop the soft passivity constraint
and re-solve with **only** the hard, safety-critical CBF constraint. The CBF is the
constraint that must hold; passivity is the soft one that can be relaxed momentarily.
The regression test
`test_controller_stays_finite_near_a_documented_infeasible_state` reproduces the exact
state that triggered this.

## API reference

::: dense_armor.dynamics.passivity_cbf_controller.solve_control_qp

---

## Details

**Two-step promotion.** Dense-Evolution-Discovery Experiment 61 implemented Kurtz,
Wensing & Lin's (2021, [arXiv:2109.13349](https://arxiv.org/abs/2109.13349)) controller
but hardcoded to one Kinova Gen3's kinematics. Experiment 63 replaced the hardcoded
calls with `RigidBodyModel`'s API and re-validated on the same three robots
`RigidBodyModel` itself was validated on.

**Scope**: task-space **position** tracking only (3 DoF). For full 6-DoF (position +
orientation) tracking, see [six_dof_pbc_cbf_controller](six_dof_pbc_cbf_controller.md).
Mimic-joint constraints (e.g. a gripper's two fingers tied together) are modeled by
`RigidBodyModel` — see [coupled joints via mimic](mimic_joints.md) — so a mimic joint
contributes no independent column of its own to this controller's QP.

**Joint limits, one more number.** The joint-limit box is added only when a robot's
URDF has a real finite limit somewhere. An unconditional (but mathematically inert)
row was tried first and rejected: it measurably perturbed OSQP's internal scaling and
broke the machine-precision cross-check for a robot with no real limits declared.

**See also**: [Full 6-DoF controller](six_dof_pbc_cbf_controller.md) — the same QP
structure for position + orientation. [Rate limiter](../control/rate_limiter.md) —
the single-integrator level counterpart, if you do not need the torque model.
