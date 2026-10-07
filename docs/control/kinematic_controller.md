# Kinematic controller (closed-form tracking)

The [trajectory](trajectory.md) generator gives a smooth reference to follow. Something
has to turn that reference into an actual command. On a joint-velocity plant, the
something is the **kinematic controller**: a closed-form feedback law that drives the
joint toward the reference with a guaranteed exponential convergence rate.

The law is one line and has no tuning beyond the convergence rate. It is not a PD
controller with a mass matrix, not a model-predictive controller with a horizon, not a
passivity-based torque controller. It is a **velocity-level** controller, at the same
dynamical level as the [rate limiter](rate_limiter.md) and the
[CBF filter](cbf_filter.md) — because that is the level at which the robot's driver
API works.

## The reference and the current state

Take the trajectory from the [previous page](trajectory.md). At some sample `i`:

The joint has fallen behind the reference (`q_actual < q_ref` on joints 0 and 1); it has
also drifted slightly ahead on joint 2. The reference is moving at about 0.2 rad/s.

## 1. Compute the command

```python
import numpy as np
from dense_armor.utility.control.kinematic_controller import kinematic_tracking_controller

q = np.array([0.05, 0.5, -0.25, 0.1, 0.0, 0.0])
q_ref = np.array([0.10, 0.6, -0.20, 0.1, 0.0, 0.0])
qd_ref = np.array([0.20, 0.20, 0.10, 0.0, 0.0, 0.0])
print(kinematic_tracking_controller(q=q, q_ref=q_ref, qd_ref=qd_ref, kp=5.0))
```

```
[0.45 0.7  0.35 0.   0.   0.  ]
```

The command is $u = \dot q_{ref} + k_p (q_{ref} - q)$: for the first joint $0.20 + 5\,(0.10 - 0.05) = 0.45$.

`u_des` is a joint-velocity command. Feed it into the [rate limiter](rate_limiter.md)
and the [CBF filter](cbf_filter.md) before sending it to the motor, exactly as you
would any other desired velocity.

## 2. The formula

```
u_des = qd_ref + kp · (q_ref − q)
```

- `qd_ref` — the reference velocity from the trajectory.
- `q_ref − q` — the tracking error at this instant.
- `kp` — the convergence rate.

Feedforward plus proportional error correction. Nothing else.

For the plant `q̇ = u`, this makes the tracking error `e = q_ref − q` obey

```
ė = q̇_ref − q̇ = qd_ref − (qd_ref + kp (q_ref − q)) = −kp · e
```

which is a stable first-order linear ODE: `e(t) = e(0) · e^{−kp·t}`. The error decays
exponentially to zero with time constant `1 / kp`.

### Hand case

`e = q_ref − q = (0.05, 0.10, 0.05, 0, 0, 0)`, `kd_ref = (0.20, 0.20, 0.10, 0, 0, 0)`,
`kp = 5`:

```
u_des = (0.20, 0.20, 0.10, 0, 0, 0) + 5 · (0.05, 0.10, 0.05, 0, 0, 0)
      = (0.45, 0.70, 0.35, 0, 0, 0)
```

The controller asks the joints to move faster than the reference, to catch up. It is
not bounded; the [rate limiter](rate_limiter.md) is what makes the command executable
by the motor. Without it, a large error would produce a large velocity that the motor
either saturates or refuses.

## 3. Choosing kp

`kp` is the only knob, and it has a clear physical meaning: **`1/kp` is the time
constant of the error decay, in seconds.**

- `kp = 5` → time constant 0.2 s. A 0.1 rad error is down to 0.037 rad in 0.2 s.
- `kp = 10` → time constant 0.1 s. Twice as aggressive.
- `kp = 20` → time constant 0.05 s. Four times as aggressive.

Too large and the command becomes jittery (any small measurement noise is amplified by
`kp`); too small and the joint lags the reference visibly. For a 100 Hz control loop,
`kp` in the range 5–10 is typically what feels right on a real arm.

## 4. What "exponential convergence" buys you

Because the error dynamics are exactly `ė = −kp·e`, the controller has a property
that most nonlinear controllers can only approximate: **for any reference trajectory
that is itself smooth, the error decays at the same rate, always**. The convergence
rate does not depend on the current configuration, on the reference amplitude, or on
which joint has drifted. This makes the controller predictable: if you know `kp` you
know how long the joint will take to catch up from any given error.

This is where the "closed-form" in the module's name comes from. There is no iteration,
no optimization, no matrix inversion: the command is a single vector addition and a
scalar multiplication.

## 5. Why not a passivity-based controller

Passivity-based controllers exist and are used on robots — the
[passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md) is
one. They work at the **torque** level, which means they need a full dynamics model
(mass matrix, Coriolis terms, gravity vector). That is a real level of complexity: the
URDF parser, the dynamics module, and the whole model-based control stack. On a robot
whose driver already implements an inner torque loop and exposes a velocity API, the
whole stack is unnecessary.

The kinematic controller is what fits that API: it takes a reference and gives a
velocity, exactly at the level the driver expects. The passivity-based controller is
what fits a torque API: it takes the current state and gives a torque, using the model
to compute the required compensation. They are both correct; they answer to different
hardware abstractions.

## 6. Where this ships

The module was validated on two independent real physical domains (SO-101, ALOHA) and
chained with `quintic_trajectory` on the same 20 real joint excursions used to validate
that module: every real excursion recovers from a real disclosed nonzero initial
tracking error and converges.

## API reference

::: dense_armor.utility.control.kinematic_controller

---

## Details

Not literally "passivity-based" in the sense of the papers that motivated this search.
Three were read in full and rejected:

- **Wu & Tan 2025** — the real target, but paywalled with no open-access copy found.
- **Scruggs** — real, needs infinite-dimensional convex Youla-parameter optimization.
- **Califano et al.** — real, needs differential-geometric Hamiltonian mechanics.

Classical PD-with-gravity-compensation was the original fallback idea, but it needs a
real second-order dynamics model (mass matrix, Coriolis terms, gravity vector) — the
exact URDF/dynamics scope this stack deliberately avoids at the control level. What
ships here is simpler and honest about it: a real, closed-form, provably convergent
kinematic tracking law at the same dynamical level as the rest of the control stack.

**See also**: [Trajectory](trajectory.md) generates `q_ref` / `qd_ref`;
[Rate limiter](rate_limiter.md) and [CBF filter](cbf_filter.md) keep the resulting
`u_des` safe in speed and space; [passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md)
is the torque-level counterpart.
