# Passivity + singularity-CBF controller (any robot)

[Rigid-body dynamics](urdf_dynamics.md) gives you M(q), gravity, and forward dynamics for any
robot. `solve_control_qp` uses them to drive that robot toward a task-space target safely --
guaranteeing passivity of the tracking error and staying away from kinematic singularities,
both as constraints in a small QP, not as separate ad-hoc checks.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03)
```

`link_name` is any link in the URDF; `p_des`/`pd_des`/`pdd_des` are the desired
position/velocity/acceleration of that link at the current instant (e.g. from
[quintic_trajectory](trajectory.md)). `eps` is the minimum manipulability index the controller
will maintain -- `mu` (returned) never drops far below it, even when the commanded target would
otherwise drive the robot through a singularity.

## What the QP actually solves

Every call solves for the joint acceleration `qdd` that gets closest to a nominal
operational-space PD command, subject to two constraints:

- **Passivity**: `Vdot <= 0`, where `V` is the tracking-error storage function -- guarantees
  the closed loop doesn't inject energy it shouldn't.
- **Singularity avoidance**: an exponential CBF keeping the manipulability index `mu(q)` above
  `eps`.
- **Joint limits**: `model`'s own real per-joint position/velocity bounds (parsed from the
  URDF's `<limit>` tags), added only for joints the URDF actually declares a limit for.

Both constraints are affine in `qdd`; their coefficients are extracted by evaluating the
constraint function and its gradient at `qdd=0` (exact, since the function is affine), not by
deriving them by hand.

::: dense_armor.dynamics.passivity_cbf_controller.solve_control_qp

---

## Details

**Two-step promotion**, same as `RigidBodyModel`: Dense-Evolution-Discovery Experiment 61
implemented Kurtz, Wensing & Lin's (2021, arXiv:2109.13349) controller but hardcoded to one
Kinova Gen3's kinematics. Experiment 63 replaced the hardcoded calls with `RigidBodyModel`'s
API and re-validated on the same three robots `RigidBodyModel` itself was validated on:

| robot | link | min(mu), no CBF | min(mu), CBF eps=0.03 |
|---|---|---|---|
| Kinova Gen3 7-DoF | `end_effector_link` | 0.00003 | 0.02947 |
| Kinova Gen3 6-DoF | `bracelet_with_vision_link` | 0.00006 | 0.02995 |
| Franka Panda | `panda_hand` | 0.00296 | 0.02997 |

Each row drives the named link toward that robot's own true kinematic singularity (`mu=0`).
Without the CBF, the controller reaches it; with it, manipulability stays within 0.1-1.8% of
the declared floor in every case.

**A real bug, found and fixed**: OSQP can report the passivity+CBF QP jointly infeasible (the
passivity constraint's coefficients go numerically near-zero exactly when tracking is already
good, which combined with a tight CBF margin occasionally leaves no feasible point under OSQP's
default tolerances) and return its infeasibility certificate -- a vector with norm in the
billions -- as if it were a real solution. Fixed by checking the solver status and, on
infeasibility, dropping the soft passivity constraint and re-solving with only the hard,
safety-critical CBF constraint -- the regression test
`test_controller_stays_finite_near_a_documented_infeasible_state` reproduces the exact state
that triggered this.

**Scope**: task-space position tracking only (3 DoF). For full 6-DoF (position + orientation)
tracking, see [six_dof_pbc_cbf_controller](six_dof_pbc_cbf_controller.md). Mimic-joint
constraints (e.g. a gripper's two fingers tied together) are not modeled by `RigidBodyModel`
underneath this, so each non-fixed joint is treated as independently controllable.

**Joint limits, real numbers**: Franka Panda `joint4` (real range `[-3.1416, 0.0]`) sitting
right at its bound with velocity driving past it -- unconstrained nominal command `qdd=-205.8`,
real CBF box `[-7.175, -4.999]`, solved `qdd=-7.175` (the box's own edge). The box rows are
only added when a robot's URDF has a real finite limit somewhere: an unconditional (but
mathematically inert) row was tried first and rejected, since it measurably perturbed OSQP's
internal scaling and broke the machine-precision cross-check above for a robot with no real
limits declared.
