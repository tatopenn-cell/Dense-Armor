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

In theory the QP always has a solution away from singular configurations (Proposition 2 of the
paper); numerically, when the solver reports it infeasible, the controller keeps the safety
constraint and relaxes passivity (step 6).

## The robot

Any robot loaded from its URDF. Here a Franka Panda, the same robot used everywhere in
this section.

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
q = jnp.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5, 0.02, 0.02])
p_des, z = jnp.array([0.5, 0.0, 0.5]), jnp.zeros(3)
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, jnp.zeros(model.n), p_des, z, z, eps=0.03)
print(tau.shape, round(float(mu), 4), round(float(h), 4))
```

```
(9,) 0.0921 0.0621
```

The controller is called `solve_control_qp` because it is a **quadratic program**
solver: at every control tick, it takes the current state, the desired task-space
target, and returns the joint torque that meets both guarantees with minimum deviation
from the standard passivity-based control command.

The four return values:

- `qdd` — the joint acceleration command.
- `tau` — the joint torque command.
- `mu` — the manipulability index at the current configuration.
- `h` — the barrier function value, `h(q) = μ(q) − ε`. Positive means "inside the safe
  set"; the controller keeps it non-negative.

`"panda_hand"` is the **name of the link being tracked** — the same name the URDF uses.
`p_des`, `pd_des`, `pdd_des` are the desired position, velocity, and acceleration of
that link at this instant. `eps` is the manipulability floor.

## 1. What "manipulability" means

The **manipulability index** at a configuration `q` is

```
μ(q) = √( det( J(q) J(q)ᵀ ) )
```

where `J(q)` is the task-space Jacobian at `q`. When the arm can move freely in the
task space, `μ > 0`; when the arm is at a singularity (fully extended, or in a wrist
singularity), `μ = 0`. The index is a smooth function of `q` and gives a single number
for "how far from singularity we are right now".

The controller enforces `μ(q) ≥ ε` with a user-chosen floor `ε`. The call above uses
`eps=0.03`, the value the paper's own simulations use (Kurtz, Wensing & Lin 2021,
Section IV). At the configuration in the example the index is `0.0921`, well above the
floor, and the barrier `h = 0.0921 − 0.03 = 0.0621` is correspondingly positive.

## 2. What the QP actually solves

At each call the controller first computes a nominal joint acceleration $\ddot q_{nom}$ (task-space
PD with gains `kp_task`, `kd_task`, plus damping `kd_null` in the redundant null space), then
finds the acceleration closest to it that satisfies the constraints:

$$\min_{\ddot q}\; \tfrac12 \lVert \ddot q - \ddot q_{nom} \rVert^2 \quad \text{s.t.} \quad a_1^\top \ddot q \le u_1 \;(\text{passivity}),\quad a_2^\top \ddot q \le u_2 \;(\text{singularity CBF}),\quad \ddot q_{lb} \le \ddot q \le \ddot q_{ub} \;(\text{joint limits}).$$

### Symbols

- $\ddot q_{nom}$ — what an unconstrained task-space PD controller would command.
- $a_1^\top \ddot q \le u_1$ — the passivity row: the storage function of the tracking error
  does not grow.
- $a_2^\top \ddot q \le u_2$ — the singularity row, an *exponential* CBF on $h = \mu(q) - \varepsilon$:
  because $h$ depends on $q$ only, the constraint acts on its second derivative,
  $\ddot h + k_1 \dot h + k_0 h \ge 0$, with the gains `ka = (k0, k1) = (100, 20)`.
- $\ddot q_{lb}, \ddot q_{ub}$ — the joint-limit box, from the URDF's own `<limit>` tags; rows are
  added **only** when the robot declares a real limit.
- The torque is then $\tau = M(q)\ddot q + C(q,\dot q)\dot q + g(q)$ from the
  [URDF dynamics](urdf_dynamics.md).

All rows are affine in $\ddot q$, so this is a small QP, solved with OSQP
([Stellato et al. 2020](https://doi.org/10.1007/s12532-020-00179-2)).

## 3. Hand case: joint limit CBF in action

Franka Panda, `panda_joint4`, real range `[−3.1416, 0.0]`. The joint sits right at its upper
bound (`q[3] = −0.001`) and moves further up (`qd[3] = 5.0`):

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model, z = RigidBodyModel("panda.urdf"), jnp.zeros(3)
q = jnp.array([0.0, 0.5, 0.0, -0.001, 0.0, 1.5, 0.0, 0.0, 0.0])
qd = jnp.zeros(9).at[3].set(5.0)
p_des = model.link_position(q, "panda_hand") + jnp.array([0.0, 0.0, 0.3])
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, z, z, eps=0.03)
print(round(float(qdd[3]), 3))
```

```
-7.175
```

The nominal PD command alone would ask `qdd[3] = −205.8`. At this state the joint-limit box for
`qdd[3]` is `[−7.175, −4.999]` (from the URDF limit, the position and the velocity); the solver
returns `−7.175`, the edge of the box. The motor is never commanded past the joint limit,
whatever the PD command wants.

## 4. On three real robots

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

## 5. What "passivity" means here

The tracking-error storage function `V` is a scalar measuring how far the joint is
from the reference, weighted by the mass matrix. The passivity constraint is `V̇ ≤ 0`
at every control tick: the controller is not allowed to add energy to the tracking
loop. On a real robot this is what makes the arm safe around humans — an operator
pushing on the arm cannot be surprised by the arm pushing back harder than it was
pushed.

Without the passivity constraint, the singularity CBF alone would still prevent
singularities but would allow configurations where the arm is doing work on the
tracking error, which is exactly the failure mode passivity-based control was
designed to avoid. The paper's own Section II-C shows why a standard constrained PBC
(the classic QP with `Jᵀτ = f_des` plus additional constraints, equation (15) of the
paper) loses its passivity guarantee the moment any additional constraint becomes
active. The controller here keeps it: the passivity constraint is inside the QP, not
assumed.

## 6. A real bug, found and fixed

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

**Source of the two guarantees.** The controllability of the singularity barrier — the
fact that `J_μ q̈ ≥ b` always has a solution for any real `b`, because `J_μ` is a
nonzero `1×n` row — is Proposition 1 of the paper (the barrier (22) is an exponential CBF).
Proposition 2 states that the QP (24) has a feasible solution for any non-singular joint
configuration. The joint
limits are an additional CBF row on top of the paper's singularity constraint, added
in this implementation and validated separately.

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
