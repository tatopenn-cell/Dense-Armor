# Full 6-DoF passivity + singularity-CBF controller

The [passivity + singularity-CBF controller](passivity_cbf_controller.md) tracks a
link's **position only**. Many robot tasks need more: a drill that has to stay
perpendicular to a surface, a gripper that has to approach an object at a specific
orientation, a camera that has to point at a target. Position alone is not enough.

The 6-DoF controller extends the same QP structure to the link's **full pose** —
position and orientation together — by using its 6×N spatial Jacobian instead of the
3×N translational one. Everything else is unchanged.

## The robot and the target

Same Franka Panda, same URDF. This time the target is a full pose: position *and*
orientation.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")

q = jnp.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5])
qd = jnp.zeros(model.n)
p_des = jnp.array([0.5, 0.0, 0.5])
pd_des = jnp.zeros(3)
pdd_des = jnp.zeros(3)
R_des = jnp.eye(3)
w_des = jnp.zeros(3)
wd_des = jnp.zeros(3)
```

## 1. The call

```python
qdd, tau, mu, h = solve_control_qp(
    model, "panda_hand",
    q, qd,
    p_des, pd_des, pdd_des,
    R_des, w_des, wd_des,
    eps=0.03,
)
```

The extra arguments beyond the position-only controller:

- `R_des` — the desired orientation of the tracked link, as a 3×3 rotation matrix.
- `w_des` — its desired angular velocity in the world frame (3-vector).
- `wd_des` — its desired angular acceleration in the world frame.

Everything else is identical: `eps` is the manipulability floor, `mu` is the
manipulability index at the current state, `h` is `mu − eps`.

## 2. The attitude error

The orientation error uses Lee, Leok & McClamroch (2010)'s SO(3) formula:

```
e_R = 0.5 · vee( R_desᵀ R − Rᵀ R_des )
```

### Symbols

- `R` — the current orientation of the tracked link, as a rotation matrix.
- `R_des` — the desired orientation.
- `vee(·)` — the inverse of the *hat* operator: it takes a skew-symmetric 3×3 matrix
  and returns the 3-vector whose hat is that matrix. In practice,
  `vee([[0, −a, b], [a, 0, −c], [−b, c, 0]]) = (c, b, a)`.
- `e_R` — a 3-vector representing the small rotation that takes `R` to `R_des`.

`e_R` is **zero iff `R = R_des`**, and it varies smoothly with `R` everywhere. This
is the standard property you want from an orientation error: no gimbal lock, no
discontinuity at ±180°, and the magnitude of `e_R` is proportional to the angle of
the discrepancy for small angles.

The naive alternative — computing the difference of roll / pitch / yaw angles — has a
real singularity at the poles (`pitch = ±90°`), where two of the three angles become
undefined and the error signal is meaningless. The SO(3) formulation avoids that
entirely.

## 3. The QP, symbol by symbol

The QP is the same structure as the position-only controller:

```
minimize_qdd   ‖ tau_pd_full − (M qdd + C qd + g) ‖²
subject to     V̇ ≤ 0
               μ̇ + α (μ − ε) ≥ 0
               q_min ≤ q + qd dt ≤ q_max
```

The only difference is inside `tau_pd_full`, the nominal PD command now includes both
a position error term and an orientation error term:

```
tau_pd_full = Jᵀ · [ Kp_p (p_des − p) + Kd_p (pd_des − pd)        ;
                     Kp_R · e_R            + Kd_R (w_des − w) ]
```

`J` is the 6×N spatial Jacobian. `Kp_p`, `Kd_p` are the linear gains; `Kp_R`, `Kd_R`
are the rotational gains. The bracket groups the linear and angular parts into a
6-vector that the transpose Jacobian maps to joint torques.

The passivity constraint `V̇ ≤ 0` is computed from the same storage function as in the
position-only case, but with the storage function now including both the linear and
angular tracking errors.

## 4. Hand case: exact gravity compensation

At the exact desired pose (position **and** orientation), with zero velocity, the
tracking error is zero. The QP should then return a `qdd` of zero and a `tau` equal
to gravity compensation exactly.

```python
q = jnp.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.5])
p, R = model.link_pose(q, "panda_hand")
qdd, tau, mu, h = solve_control_qp(
    model, "panda_hand", q, jnp.zeros(model.n),
    p, jnp.zeros(3), jnp.zeros(3),
    R, jnp.zeros(3), jnp.zeros(3),
    eps=0.03,
)
```

`qdd` is zero to machine precision (`1e-9`), and `tau` equals
`model.gravity_forces(q)` to the same precision. This is a correctness check of the
whole pipeline (spatial Jacobian, rotation error, task-space Lambda), not just "does
not crash": the only torque the controller should ask for at the desired pose is the
one holding the arm against gravity.

## 5. Real closed-loop convergence

Start the arm 10 cm away in position and 30° away in orientation (rotation about world
`z`). Integrate under the real rigid-body dynamics with RK4 for 1000 control ticks at
5 physics sub-steps each.

```
final position error:     1e-6 m
final orientation error:  1e-4 rad
```

Both errors converge to zero. This is the same order of magnitude that the position-only
controller reaches for position alone — extending to full pose does not degrade the
convergence.

## 6. A second real bug, found by the 6-DoF validation

The 6-DoF (spatial) manipulability measure can be **well below the 3-DoF one** at the
same configuration. Concretely, on the Kinova Gen3 6-DoF at a certain
wrist-singularity-adjacent state:

```
μ_3-DoF = 0.1128
μ_6-DoF = 0.0176
```

The position-only measure does not see the wrist singularity; the full 6-DoF measure
does. This means `h = μ − ε` can already be **negative** before the QP solves, and the
CBF's required recovery rate can exceed the velocity-limit box.

The single-level fallback this module inherited from
[`passivity_cbf_controller.py`](passivity_cbf_controller.md) (drop passivity, keep
CBF + box) is not enough there, and again silently returned OSQP's infeasibility
certificate as `qdd` (norm in the billions).

The fix is a **third-level fallback** specific to this module: if CBF + box is still
infeasible, drop the box too, and keep only the CBF constraint. The CBF alone is
guaranteed feasible in `ℝⁿ` as long as the manipulability gradient is nonzero — and
preventing an actual kinematic singularity is the harder safety constraint of the two.
The joint-limit box is the one that gives.

## 7. On real robots

| robot | link | μ | h = μ − ε | qdd norm |
|---|---|---|---|---|
| Kinova Gen3 6-DoF | `bracelet_with_vision_link` | 0.017607 | −0.012393 | 162.60 |
| Franka Panda | `panda_hand` | 0.032751 | 0.002751 | 5.00 |

Both are near-singular configurations where the position-only controller would either
run into the singularity or crash on the infeasibility certificate. The 6-DoF
controller with the third-level fallback resolves both.

## API reference

::: dense_armor.dynamics.six_dof_pbc_cbf_controller.solve_control_qp

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 65**, built directly on
Experiment 63's `RigidBodyModel`-based controller
([`passivity_cbf_controller.py`](passivity_cbf_controller.md)).

**Validated two ways:**

- **Exact gravity compensation at zero error**: at the exact desired pose with zero
  velocity, the solved `qdd` is zero and `tau` equals gravity compensation to machine
  precision (`1e-9`). This checks the spatial Jacobian, the rotation error, and the
  task-space Lambda together.
- **Real closed-loop convergence**: a 10 cm position offset plus a 30° orientation
  offset (about world `z`), RK4-integrated for 1000 control ticks, converges to
  position error `1e-6` m and orientation error `1e-4` rad.

**Scope**: otherwise inherits `passivity_cbf_controller.py`'s joint-limit CBF
unchanged; see that module's own Details section for its own real numbers and its
(single-level) OSQP-infeasibility fix.

**See also**: [Passivity + singularity-CBF controller](passivity_cbf_controller.md) —
the position-only version, with its own joint-limit handling.
[Rigid-body dynamics](urdf_dynamics.md) — `link_pose` and `link_spatial_jacobian`,
the two methods this controller uses from `RigidBodyModel`.
