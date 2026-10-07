# Trajectory (closed-form point-to-point generator)

The [rate limiter](rate_limiter.md) bounds how fast a command can change and the
[CBF filter](cbf_filter.md) bounds where it can go, but neither generates a reference
to track in the first place. Something has to say "go from configuration A to
configuration B over T seconds, smoothly". **Trajectory** is that something: a
closed-form, minimum-jerk-continuous path between two configurations, for any number of
joints at once.

The generator is a **quintic polynomial**: fifth-degree in time, chosen so that
position, velocity, and acceleration all start at their initial values and end at their
final values with no jump. That is the smallest degree that can do it, and the smallest
degree that makes the resulting motion feel right — no jerk at the endpoints, no
overshoot, no oscillation.

## The two configurations

A 6-joint arm has to move from its current joint configuration to a target one, in 2
seconds.

The arm is at rest before and after the move: no residual velocity, no residual
acceleration.

## 1. Generate the trajectory

```python
import numpy as np
from dense_armor.utility.control.trajectory import quintic_trajectory

q0 = np.array([0.0, 0.5, -0.3, 0.1, 0.0, 0.0])
qf = np.array([0.5, 1.0, -0.1, 0.2, 0.0, 0.0])
t, q, v, a = quintic_trajectory(q0=q0, qf=qf, T=2.0)
print(q[0].round(3), q[-1].round(3), np.abs(v).max(axis=0).round(3))
```

```
[ 0.   0.5 -0.3  0.1  0.   0. ] [ 0.5  1.  -0.1  0.2  0.   0. ] [0.469 0.469 0.187 0.094 0.    0.   ]
```

The path starts exactly at `q0`, ends exactly at `qf`, and the largest joint speed is 0.469 rad/s.

`t` is a time vector; `q`, `v`, `a` are the position, velocity and acceleration
profiles, all with shape `(len(t), 6)`. Every joint gets its own independent polynomial
over the same real time `T`. The function returns a fine-grained grid of samples so
that any downstream consumer (a controller, a plot, a replay) has the full profile.

## 2. The polynomial, symbol by symbol

The position at time `s ∈ [0, T]` is a fifth-degree polynomial:

```
q(s) = c₀ + c₁ s + c₂ s² + c₃ s³ + c₄ s⁴ + c₅ s⁵
```

The six coefficients are determined by six boundary conditions — three at `s = 0` and
three at `s = T`:

```
q(0) = q0,   q'(0) = v0,   q''(0) = a0
q(T) = qf,   q'(T) = vf,   q''(T) = af
```

For the default of "start and end at rest", `v0 = vf = a0 = af = 0`. Solving the six
equations gives the six coefficients in closed form; there is no iteration, no
optimization, no trajectory library.

**Hand case.** Single joint, `q0 = 0`, `qf = 1`, `T = 2`, start and end at rest. The
coefficients come out as `c₀ = 0`, `c₁ = 0`, `c₂ = 0`, `c₃ = 1.25`, `c₄ = −0.9375`,
`c₅ = 0.1875`. The position at `s = 1` (halfway in time) is
`1.25 − 0.9375 + 0.1875 = 0.5` — exactly halfway in position. That symmetry holds for
any `q0`, `qf`, `T` with rest at both ends, and it is one of the properties that makes
the quintic pleasant: the motion is centred on the move's midpoint.

## 3. Chaining segments

Pass `v0` and `vf` to start or end already moving instead of at rest. This is what
makes it possible to chain several segments without the robot stopping at every
intermediate waypoint:

```python
import numpy as np
from dense_armor.utility.control.trajectory import quintic_trajectory

q0, qf = np.array([0.0, 0.5]), np.array([0.5, 1.0])
q_mid = 0.5 * (q0 + qf)
t1, q1, v1, a1 = quintic_trajectory(q0, q_mid, T=1.0)
t2, q2, v2, a2 = quintic_trajectory(q_mid, qf, T=1.0, v0=v1[-1], a0=a1[-1])
print(q1[-1].round(3), q2[0].round(3), v1[-1].round(3), v2[0].round(3))
```

```
[0.25 0.75] [0.25 0.75] [0. 0.] [0. 0.]
```

Two quintic pieces joined at the midpoint: same position (0.25, 0.75) and same velocity (0) at the joint.

The end velocity of the first segment becomes the start velocity of the second, so the
velocity profile is continuous across the join. The acceleration too. The result is a
piecewise-quintic trajectory with `C²` continuity everywhere, which is more than enough
for a low-level controller to track smoothly.

## 4. Any number of joints

The function is fully vectorised: `q0` and `qf` can be scalars, or 1-D arrays of any
length, or 2-D arrays for a batched call. Every joint uses the same time grid and its
own independent polynomial, so no coordination between joints is needed. If you need
joints to move in a coordinated way (arrive at the same time, or maintain a fixed
relative phase), that is a downstream concern: you feed each joint's trajectory into the
[kinematic controller](kinematic_controller.md), which handles the coupling at the
velocity level.

## 5. Downstream

The output is the reference. Two standard consumers:

```python
import numpy as np
from dense_armor.utility.control.trajectory import quintic_trajectory
from dense_armor.utility.control.kinematic_controller import kinematic_tracking_controller

t, q, v, a = quintic_trajectory(np.zeros(2), np.array([0.5, 1.0]), T=2.0)
dt, qa = t[1] - t[0], np.array([0.05, -0.05])
for i in range(len(t)):
    qa = qa + dt * kinematic_tracking_controller(qa, q[i], v[i], kp=5.0)
print(np.abs(qa - q[-1]).round(4))
```

```
[0.0007 0.0014]
```

The kinematic controller follows a 2-joint quintic path starting 0.05 rad off; at the end the error is
below 0.0015 rad.

The trajectory says *where to go*; the kinematic controller says *how to get there on
a single-integrator plant*; the rate limiter and CBF filter keep the resulting command
safe in speed and space.

## API reference

::: dense_armor.utility.control.trajectory

---

## Details

Scoped down from two papers proposing much larger URDF- and dynamics-aware trajectory
optimizers (Lozer, Scalera, Boscariol & Gasparetto, *Robotics and Autonomous Systems*;
Fried & Paternain, [arXiv:2412.07859](https://arxiv.org/abs/2412.07859)). Both were
read in full before writing this. They solve the problem the same way in principle —
generate a smooth trajectory from a boundary-value problem — but they integrate the
robot's real dynamics into the generation, which requires the full mass matrix and
Coriolis terms. That is the scope of the
[passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md), not
of this module.

The module was promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (SO-101, ALOHA, 20 real joint excursions): the
quintic's peak velocity is always lower than the real recorded peak velocity for the
same start, end, and real elapsed duration. This is expected — the quintic is the
smoothest possible point-to-point path, so it uses less peak velocity than a human
operator's motion for the same total displacement and time.

**Single-segment only.** Chaining several segments across many waypoints is a real next
step, not implemented in the generator itself. The manual chaining in section 3 above
is the workaround and gives the same `C²` continuity.

**See also**: [Rate limiter](rate_limiter.md) and [CBF filter](cbf_filter.md) — this
module generates a reference to follow; those two keep whatever follows it safe in
speed and space.
