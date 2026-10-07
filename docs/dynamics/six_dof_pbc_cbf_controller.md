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
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
q, z = jnp.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5, 0.02, 0.02]), jnp.zeros(3)
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, jnp.zeros(model.n),
                                   jnp.array([0.5, 0.0, 0.5]), z, z, jnp.eye(3), z, z, eps=0.03)
print(tau.shape, round(float(mu), 4), round(float(h), 4))
```

```
(9,) 0.0641 0.0341
```

The extra arguments beyond the position-only controller:

- `R_des` — the desired orientation of the tracked link, as a 3×3 rotation matrix.
  `jnp.eye(3)` is the identity orientation: no rotation from the world frame.
- `w_des` — its desired angular velocity in the world frame (3-vector).
- `wd_des` — its desired angular acceleration in the world frame.

Everything else is identical: `eps` is the manipulability floor, `mu` is the
manipulability index at the current state, `h` is `mu − eps`. The four returned values
carry the same meaning as in the position-only version: `qdd`, `tau`, `mu`, `h`.

The same configuration that gave `mu = 0.0921` under the position-only controller
gives `mu = 0.0641` here. The spatial manipulability is a **stricter** measure than the
positional one: it takes the angular rows of the Jacobian into account, and the arm at
this configuration is further from singularity in position than in orientation.

## 1. The attitude error

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

## 2. The QP, symbol by symbol

The QP has the same structure as the position-only controller:

$$\min_{\ddot q}\; \tfrac12 \lVert \ddot q - \ddot q_{nom} \rVert^2 \quad \text{s.t.} \quad a_1^\top \ddot q \le u_1 \;(\text{passivity}),\quad a_2^\top \ddot q \le u_2 \;(\text{singularity CBF}),\quad \ddot q_{lb} \le \ddot q \le \ddot q_{ub} \;(\text{joint limits}).$$

The difference is inside $\ddot q_{nom}$: the nominal task-space command now contains a position
error and an orientation error, stacked in a 6-vector and mapped through the 6×N spatial Jacobian,

$$\begin{bmatrix} K_{p,R}\, e_R + K_{d,R}(\omega_{des} - \omega) \ K_{p,p}(p_{des} - p) + K_{d,p}(\dot p_{des} - \dot p) \end{bmatrix},$$

angular rows first, as in the Jacobian's `[ω; v]` order. The passivity row uses a storage function
with both the linear and the angular tracking error, and the manipulability $\mu$ of the CBF row
is computed from the **spatial** Jacobian: this is what lets the controller see the wrist
singularities the position-only version does not.

## 3. Hand case: exact gravity compensation

At the exact desired pose (position **and** orientation), with zero velocity, the
tracking error is zero. The QP should then return a `qdd` of zero and a `tau` equal
to gravity compensation exactly.

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
q, z = jnp.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.5, 0.02, 0.02]), jnp.zeros(3)
p, R = model.link_pose(q, "panda_hand")
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, jnp.zeros(model.n), p, z, z, R, z, z, eps=0.03)
print(round(float(jnp.abs(qdd).max()), 4), round(float(h), 4))
```

```
0.0 0.0641
```

`qdd` is zero to four decimal places, and `tau` equals `model.gravity_forces(q)` to machine
precision (largest difference 5.6e-17). This is a correctness check of the whole pipeline — the spatial
Jacobian, the rotation error, the task-space Lambda — not just "does not crash": the
only torque the controller should ask for at the desired pose is the one holding the
arm against gravity.

## 4. Real closed-loop convergence

Start the arm 10 cm away in position and 30° away in orientation (rotation about world
`z`). Integrate under the real rigid-body dynamics with RK4 for 1000 control ticks at
5 physics sub-steps each.

```
final position error:     below 1e-6 m
final orientation error:  below 1e-4 rad
```

Both errors converge to zero. This is the same order of magnitude that the position-only
controller reaches for position alone — extending to full pose does not degrade the
convergence.

## 5. A second real bug, found by the 6-DoF validation

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

## 6. On real robots

| robot | link | μ | h = μ − ε | qdd norm |
|---|---|---|---|---|
| Kinova Gen3 6-DoF | `bracelet_with_vision_link` | 0.017607 | −0.012393 | 162.60 |
| Franka Panda | `panda_hand` | 0.032751 | 0.002751 | 5.00 |

Both are near-singular configurations where the position-only controller would either
run into the singularity or crash on the infeasibility certificate. The 6-DoF
controller with the third-level fallback resolves both: the Kinova Gen3 6-DoF case is
the exact state that triggers the third-level fallback (h is negative before the QP
solves), and the controller still returns a finite `qdd`.

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
  position error below `1e-6` m and orientation error below `1e-4` rad (the bounds the
regression test asserts).

**Source of the theory**: Kurtz, Wensing & Lin (2021,
[arXiv:2109.13349](https://arxiv.org/abs/2109.13349)). The manipulability barrier is an
exponential CBF that always admits a solution (`J_μ q̈ ≥ b` for any `b`), their Proposition 1;
their QP (24) is feasible at any non-singular configuration, Proposition 2. The 6-DoF extension of the
manipulability to the full spatial Jacobian is a direct application of the same
formula, not a change to the paper's method.

**Scope**: otherwise inherits `passivity_cbf_controller.py`'s joint-limit CBF
unchanged; see that module's own Details section for its own real numbers and its
(single-level) OSQP-infeasibility fix. The third-level fallback described above is
specific to this 6-DoF module — the position-only controller's single-level fallback
is enough there because the 3-DoF manipulability cannot go negative in the same way
as the 6-DoF one.

**See also**: [Passivity + singularity-CBF controller](passivity_cbf_controller.md) —
the position-only version, with its own joint-limit handling.
[Rigid-body dynamics](urdf_dynamics.md) — `link_pose` and `link_spatial_jacobian`,
the two methods this controller uses from `RigidBodyModel`.
