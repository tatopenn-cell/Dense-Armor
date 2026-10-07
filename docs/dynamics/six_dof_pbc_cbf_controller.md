# Full 6-DoF passivity + singularity-CBF controller

[Passivity + singularity-CBF controller](passivity_cbf_controller.md) tracks a link's position
only. `six_dof_pbc_cbf_controller.solve_control_qp` extends the same QP to the link's full pose
-- position and orientation together -- using its 6xN spatial Jacobian instead of only the
translational one.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des,
                                    r_des, w_des, wd_des, eps=0.03)
```

`r_des` is the desired orientation (rotation matrix) of the tracked link; `w_des`/`wd_des` are
its desired angular velocity/acceleration in world frame. Everything else matches
`passivity_cbf_controller.solve_control_qp` -- same passivity + singularity-avoidance QP
structure, same real per-joint limit CBF.

## The attitude error

Orientation error uses Lee, Leok & McClamroch (2010)'s SO(3) formula:

```
e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)
```

Smooth everywhere and zero iff `R == R_des`, unlike a roll-pitch-yaw based error, which has a
real gimbal-lock singularity this formulation avoids.

::: dense_armor.dynamics.six_dof_pbc_cbf_controller.solve_control_qp

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 65**, built directly on Experiment 63's
`RigidBodyModel`-based controller (`passivity_cbf_controller.py` here). Validated two ways:

- **Exact gravity compensation at zero error**: at the exact desired pose (position and
  orientation) with zero velocity, the solved `qdd` is zero and `tau` equals gravity
  compensation exactly, to machine precision (1e-9) -- a correctness check of the whole
  pipeline (spatial Jacobian, rotation error, task-space Lambda), not just "doesn't crash".
- **Real closed-loop convergence**: a 10cm position offset plus a 30-degree orientation offset
  (about world z), RK4-integrated under the real rigid-body dynamics for 1000 control ticks at
  5 physics substeps each, converges to position error 1e-6 m and orientation error 1e-4.

**Validated on three robots**, the same singular configurations `passivity_cbf_controller.py`
uses, through the full 6-DoF pipeline instead:

| robot | link | mu | h (mu - eps) | qdd norm |
|---|---|---|---|---|
| Kinova Gen3 6-DoF | `bracelet_with_vision_link` | 0.017607 | -0.012393 | 162.60 |
| Franka Panda | `panda_hand` | 0.032751 | 0.002751 | 5.00 |

**A real second-level infeasibility, found and fixed by this validation.** The 6-DoF (spatial)
manipulability measure can be well below the 3-DoF one at the same configuration (0.0176 vs
0.1128 for Gen3 6-DoF here -- a real wrist-singularity-adjacent case the position-only measure
doesn't see), and `h` can already be negative before the QP solves. When the CBF's required
recovery rate exceeds the velocity-limit box, CBF+box is jointly infeasible -- the single-level
fallback this module inherited from `passivity_cbf_controller.py` (drop passivity, keep CBF+box)
isn't enough there, and silently returned OSQP's infeasibility certificate as `qdd` (norm in the
billions). Fixed with a third-level fallback specific to this module: if CBF+box is still
infeasible, drop the box too and keep only the CBF -- guaranteed feasible in R^n as long as the
manipulability gradient is nonzero, since the CBF (preventing an actual kinematic singularity) is
the harder safety constraint of the two.

**Scope**: otherwise inherits `passivity_cbf_controller.py`'s joint-limit CBF unchanged; see that
module's own Details section for its own real numbers and its (single-level) OSQP-infeasibility
fix.
