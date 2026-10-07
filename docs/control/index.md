# Control

Commands for a robot: **where it may go, how fast a command may change, along which
path, and the velocity that follows it**. Four modules, each bounding a different aspect
of a desired velocity command on the same single-integrator joint model, and they
compose in this order:

```
  desired pose
       │
       ▼
  [ Trajectory ]  ──►  q_ref, qd_ref  ──►  [ Kinematic controller ]  ──►  u_des
                                                                          │
                                                                          ▼
                              safe command  ◄──  [ CBF filter ]  ◄──  [ Rate limiter ]
                                                (WHERE)              (WHEN / HOW FAST)
```

- **[Trajectory](trajectory.md)** — generates a smooth, minimum-jerk point-to-point
  path between two configurations. The reference the other modules keep safe.
- **[Kinematic controller](kinematic_controller.md)** — turns a reference into a joint
  velocity command with an exact exponential convergence guarantee.
- **[Rate limiter](rate_limiter.md)** — bounds how fast the applied command can
  physically change (velocity and acceleration).
- **[CBF filter](cbf_filter.md)** — bounds where the command can go (never enter a
  forbidden region), with a minimally invasive correction when it would.

## Why four modules and not one

Each module answers a different "what could go wrong":

- The trajectory is *what the robot should do* in the ideal world.
- The kinematic controller is *how to get there* on a single-integrator plant.
- The rate limiter is *the real motors cannot execute an instantaneous jump*. A
  command that changes 10 rad/s in one tick is a command the motor will either refuse
  or dangerously approximate.
- The CBF filter is *even a slowly-changing command that goes straight into an obstacle
  is dangerous*. A command moving at a perfectly safe, bounded velocity straight into a
  wall is still a command that reaches the wall.

The four are independent because they bound different aspects of the same command, and
compose because each takes a velocity and returns a velocity.

## The single-integrator model

All four modules share one plant:

```
q̇ = u
```

The joint state is a position vector `q`; the input `u` is a joint velocity; the
dynamics are the identity. This is the level at which most low-level robot APIs work:
you send joint velocities, the inner PID loop of the driver executes them. The four
modules are *safety filters* on that velocity, not on the underlying torques; torque
control is a different problem, and its module is
[`passivity_cbf_controller`](../dynamics/passivity_cbf_controller.md).

## Pages

- **[Rate limiter](rate_limiter.md)** — causal damping of a command stream, based on
  Ruckig's velocity-and-acceleration limits.
- **[CBF filter](cbf_filter.md)** — a Control Barrier Function safety filter, keeping
  commands out of forbidden regions (Ames et al. 2019).
- **[Trajectory](trajectory.md)** — the reference generator, quintic polynomials.
- **[Kinematic controller](kinematic_controller.md)** — closed-form velocity tracking
  with exponential convergence.
