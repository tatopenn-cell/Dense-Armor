# Kinematic controller (closed-form tracking)

[Trajectory](trajectory.md) generates a smooth reference to follow, but something still has
to turn that reference into an actual command -- `kinematic_tracking_controller` is that
piece, at the same single-integrator level `rate_limiter`/`cbf_filter` already use.

```python
from dense_armor.utility.kinematic_controller import kinematic_tracking_controller

u_des = kinematic_tracking_controller(q=[0.2], q_ref=[0.5], qd_ref=[1.0], kp=5.0)
```

`u_des` is a velocity command: feed it into `rate_limiter` and `cbf_filter` before sending it
to a real motor, the same way you would any other desired command.

```python
u = qd_ref + kp * (q_ref - q)
```

For the plant `qdot = u`, this makes the tracking error `e = q_ref - q` obey `edot = -kp*e`
exactly -- closed-form exponential convergence to zero tracking error, for any reference
trajectory, not only a fixed setpoint. `kp` sets the real convergence rate (`1/kp` is the
time constant).

::: dense_armor.utility.kinematic_controller

---

**See also**: [Trajectory](trajectory.md) generates `q_ref`/`qd_ref`; [Rate limiter](rate_limiter.md)
and [CBF filter](cbf_filter.md) keep the resulting `u_des` safe in speed and space.

## Details

Not literally "passivity-based" in the sense of the papers that motivated this search (Wu &
Tan 2025 -- the real target, paywalled with no open-access copy found; Scruggs -- real, needs
infinite-dimensional convex Youla-parameter optimization; Califano et al. -- real, needs
differential-geometric Hamiltonian mechanics). Classical PD-with-gravity-compensation was the
original fallback idea but needs a real second-order dynamics model (mass matrix, Coriolis
terms, gravity vector) -- the exact URDF/dynamics scope this stack has deliberately avoided
elsewhere. What ships here is simpler and honest about it: a real, closed-form, provably
convergent kinematic tracking law at the same dynamical level as the rest of the stack.
Promoted from Dense-Evolution-Discovery after validation on two independent real physical
domains (SO-101, ALOHA), chained with `quintic_trajectory` on the same 20 real joint
excursions used to validate that module: every real excursion recovers from a real disclosed
nonzero initial tracking error and converges.
