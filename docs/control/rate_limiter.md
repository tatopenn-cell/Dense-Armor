# Rate limiter (causal command damping)

A robot's real motor cannot execute an unbounded instantaneous jump safely. If the
command stream says "joint 1 goes from 0.1 to 1.5 rad/s between this tick and the next",
the motor will either refuse, saturate, or dangerously approximate it. The **rate
limiter** is the module that makes sure the applied command can actually be tracked,
by bounding how fast it is allowed to change.

It is not a filter that decides whether a deviation is a spike or a real change. It
does not need to: it just refuses to move faster than the motors allow. The bound is on
the command, not on the signal.

## The command stream

A joint is commanded to go from 0 to 1.5 rad in a single tick. The control loop runs at
100 Hz, so a tick is 10 ms. The motor's real limit is 5 rad/s of velocity and 20 rad/s²
of acceleration.

```python
import numpy as np
fs = 100
dt = 1.0 / fs
u_des = np.zeros(300)
u_des[100] = 1.5
```

The jump from 0 to 1.5 rad/s in one tick is a 150 rad/s² acceleration — more than
seven times the motor's real limit.

## 1. Apply the limiter

```python
from dense_armor.control.rate_limiter import rate_limited_follower

u = rate_limited_follower(u_des, max_vel=5.0, max_accel=20.0, dt=dt)
```

`u` is the applied command. It is 0 before the step, then ramps up at 20 rad/s² until
it hits 1.5 rad/s. The ramp takes `1.5 / 20 = 0.075 s` — about 7.5 ticks. The command
is the same shape as before, but every rate of change in it is inside the motor's
envelope.

## 2. What "rate limited" means

The limiter tracks two things: the last applied command `u_prev`, and the change it is
allowed to make in one tick. The rule is:

```
Δ = u_des − u_prev
Δ_max = a_max · dt
if |Δ| ≤ Δ_max:
    u = u_des
else:
    u = u_prev + sign(Δ) · Δ_max
```

Two bounds apply, not one. `v_max` clamps the absolute value of the command itself;
`a_max` clamps the change between two consecutive commands. A command that is inside
`v_max` but jumps by more than `a_max · dt` in one tick is still not applied as-is: it
is ramped.

**Hand case.** `dt = 0.01 s`, `a_max = 20 rad/s²`, `v_max = 5 rad/s`, `u_prev = 0`,
`u_des = 1.5`.

```
Δ = 1.5
Δ_max = 20 · 0.01 = 0.2
|1.5| > 0.2, so u = 0 + 0.2 = 0.2 rad/s
```

The applied command moves by 0.2 rad/s. The next tick computes the same, so it takes
`1.5 / 0.2 = 7.5` ticks to reach the target. If the target were above `v_max` (say
6.0), the limiter would ramp up to 5.0 and hold there — clamped by the velocity bound,
not the acceleration bound.

## 3. Why causal damping

A purely local check ("is this command unusually far from the last one?") is not enough
to decide whether the jump is a real intended command or a glitch. What matters is
whether the *motor* can execute it. The rate limiter does not try to classify the
command; it just makes sure the command never asks the motor to do something it cannot
do. The output is always inside the motor's envelope, whether the input was a spike, a
step change, or a smooth ramp that the driver generated.

This is why a rate limiter and a **detector** are not the same tool:

- A detector decides "is this deviation real or noise?". It can be wrong.
- A rate limiter decides "can the motor execute this change in one tick?". It has a
  single, physical answer.

On a real robot you usually want both: the detector says what to trust, the limiter
says what to actually send to the motor.

## 4. Grounding

The limiter is the velocity-and-acceleration-limited special case of the profile that
[Berscheid and Kröger (2021), *Jerk-limited real-time trajectory generation with
arbitrary target state*](https://arxiv.org/abs/2105.04830) compute in full. Ruckig (the
paper's algorithm and open-source library) solves the time-optimal jerk-limited profile
in closed form for any target state, on every control cycle. The limiter here is
simpler: it does not limit jerk, and it does not solve for a target — it just clamps
the change between two consecutive commands.

The choice of the simpler form is deliberate: the target state in a real loop is not a
fixed target but a moving reference (from the [trajectory](trajectory.md) generator),
and the outer controllers on top of the limiter do not need jerk-limited optimality to
guarantee safety. They only need the rate bound.

## 5. Chained with the other control modules

The limiter sits between the kinematic controller and the CBF filter:

```python
u_des = kinematic_tracking_controller(q, q_ref, qd_ref, kp=5.0)
u = rate_limited_follower(u_des, max_vel=5.0, max_accel=20.0, dt=dt)
u_safe = cbf_safety_filter(u, obstacle=obs, dt=dt)
```

The kinematic controller produces a velocity that tracks the reference; the limiter
makes sure that velocity never asks the motor to change faster than it can; the CBF
filter makes sure the final velocity does not point into a forbidden region. Each
module bounds a different aspect of the same command.

## API reference

::: dense_armor.control.rate_limiter

---

## Details

The rate limiter was promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (the SO-101 and ALOHA robot arms) on the same 20 real
joint excursions used to validate the [trajectory](trajectory.md) generator and the
[kinematic controller](kinematic_controller.md). On every excursion the applied command
stays inside the motor's envelope and the joint reaches the reference within the
expected ramp time.

The trade-off is fidelity versus safety: the applied command lags the desired command
by up to `a_max · dt` at every tick. On the SO-101 data the peak velocity of the
limited command is lower than the peak velocity of the recorded command for the same
start, end, and elapsed time — this is expected, not a bug. The limiter is a lower
bound on how fast the robot can respond, not an estimate of how fast it will.

**See also**: [CBF filter](cbf_filter.md) — the spatial complement (WHERE the command
can go). [Kinematic controller](kinematic_controller.md) — where the command comes from.
[CUSUM and ARL theory](../drift/cusum.md) — if the real goal is recovering or
classifying a signal rather than bounding a command's rate of change, a detector is the
right tool, not this module.
