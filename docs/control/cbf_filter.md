# CBF filter (geometric command safety)

The rate limiter bounds **how fast** a command can change. It does not bound **where**
the command can take the robot. A command moving at a perfectly safe, bounded velocity
straight into a wall is still a command that reaches the wall. The **CBF filter** is
the module that keeps a command out of forbidden regions, with a minimally invasive
correction when it would enter one.

The idea is the same one that makes ABS brakes work: instead of trying to stop the
command, the filter asks "what is the smallest modification of this command that keeps
me out of the forbidden region?" and applies it.

## The command and the obstacle

A joint is commanded toward a position that is inside a forbidden region. Forbidden
regions are described by a function `h(q)`: positive inside the safe set, zero on its
boundary, negative outside.

```python
import numpy as np
x = 0.0
u_des = 1.0
```

`q_des` is the desired joint configuration. Somewhere between `q` and `q_des` there is
a wall.

## 1. Apply the filter

```python
from dense_armor.control.cbf_filter import cbf_safety_filter

u = cbf_safety_filter(x, u_des, obstacle=0.5, safe_dist=0.1)
```

`obstacle` is a callable that returns `h(q)` for a candidate joint configuration `q`.
The filter takes the desired joint configuration, the current one, and returns a
velocity command `u` that will not enter the region `h(q) < 0`.

If the desired configuration is on the safe side, the filter returns the same command
a naive controller would: `u = (q_des − q) / dt`, clamped by whatever velocity bound
the caller applies afterwards.

If the desired configuration would take the robot through the wall, the filter returns
a *modified* command — the closest one to `(q_des − q) / dt` that still keeps the robot
on the safe side of the wall.

## 2. What a Control Barrier Function is

Let `h(q)` be a function with `h(q) > 0` inside the safe set, `h(q) = 0` on its
boundary, `h(q) < 0` outside. A *Control Barrier Function* is a function `h` that
satisfies, at every `q`,

```
sup_u [ L_f h(q) + L_g h(q) u ]  ≥  −α(h(q))
```

for some class-K function `α`. In words: **there exists a control `u` that keeps `h`
from decreasing too fast**. The condition is affine in `u`, so it becomes a linear
constraint in a quadratic program.

### Symbols

- `h(q)` — the barrier function; the sign tells you which side of the boundary you are
  on.
- `L_f h(q)`, `L_g h(q)` — the Lie derivatives of `h` along the drift and along the
  input. For the single-integrator plant `q̇ = u`, `L_f h = 0` and `L_g h = ∇h(q)`.
- `α` — a strictly increasing function with `α(0) = 0`. The simplest choice is
  `α(s) = k · s`, with `k > 0` a gain that controls how aggressively the filter
  approaches the boundary. Larger `k` allows a faster approach.
- `u` — the joint velocity command (the module's output).

### The QP

At each tick the filter solves

```
minimize_u   ‖ u − u_des ‖²
subject to   ∇h(q) · u ≥ −α(h(q))
```

The objective says "stay as close as possible to the desired command"; the constraint
says "keep the barrier function from falling through zero". The solution is the
minimally invasive safe command.

## 3. Hand case

`q = (0, 0)`, `q_des = (1, 1)`, `dt = 0.01`, `h(q) = 0.2 − ‖q‖`, `α(s) = 10 s`.

```
u_des = (q_des − q) / dt = (100, 100)
∇h(q) = −q / ‖q‖         (undefined at q = 0; use (0, 0) by convention)
h(q) = 0.2
α(h(q)) = 10 · 0.2 = 2.0
```

The constraint is `∇h(q) · u ≥ −2.0`. If `q` is at the origin, `∇h = (0, 0)` and the
constraint is `0 ≥ −2.0`, which is trivially satisfied. The filter returns
`u = (100, 100)` as-is.

Move the joint to `q = (0.15, 0)`, so the boundary is close: `h(q) = 0.2 − 0.15 = 0.05`,
`α(h) = 0.5`, `∇h(q) = (−1, 0)`. The constraint is `−u_x ≥ −0.5`, i.e. `u_x ≤ 0.5`.

```
u_des = (100, 100)
u_x is clamped from 100 down to 0.5
u = (0.5, 100)
```

The command still moves fast along `y` (no wall in that direction) but is now bounded
along `x` (the wall is in that direction). The command is not blocked; it is bent
around the obstacle.

## 4. The discrete-time issue

A CBF guarantees `h(q) ≥ 0` **in continuous time**. In discrete time, one tick is
`dt` long, and a single step can cross the boundary even if the constraint was satisfied
at the start of the tick. The filter handles this by:

1. Evaluating `h` not at the current `q` but at the *predicted* `q + u · dt`.
2. Solving the QP for the velocity that keeps the *predicted* `h` non-negative.
3. Applying a sub-stepping scheme internally when the boundary is close enough that a
   single `dt` step could cross it.

Without the sub-stepping the filter would occasionally let the joint cross the boundary
by a small amount when it started close. This is a real numerical finding from the
validation on SO-101 and ALOHA — the filter bounds the *predicted* violation, not the
*current* one, and the sub-stepping handles the small residual.

## 5. Real-time use

For a live ROS2 or Ignition loop that reacts to one sensor callback at a time,
`cbf_safety_filter_live` is the same filter packaged for a real `dt` per tick, instead
of filtering a whole pre-recorded array off-line.

```python
from dense_armor.control.cbf_filter import cbf_safety_filter_live

u = cbf_safety_filter_live(x, u_des, dt=0.01, obstacle=0.5, safe_dist=0.1)
```

This module was promoted after a real live ROS2 / Ignition loop needed exactly this and
had to reconstruct it by hand from `cbf_filtered_trajectory` (Dense-Evolution-Discovery,
Experiment 58).

## 6. Relation to SAFER-Splat

The underlying theory is the same as [Ames et al. (2019), *Control Barrier Functions:
Theory and Applications*](https://arxiv.org/abs/1903.11199), the same theory that
SAFER-Splat uses. The difference is the perception layer:

- SAFER-Splat uses 3D Gaussian Splatting to compute a signed-distance function from a
  dense visual representation, which needs a GPU.
- The CBF filter here takes the obstacle as a *known geometric primitive* (a ball, a
  wall, a plane), described by a function `h(q)` the caller supplies.

For a robot working with a known obstacle map, or with a forward-kinematics-derived
distance to a known object, that is enough. For a robot that has to reason about a
scene from raw pixels, SAFER-Splat is the right tool. They are not redundant.

## API reference

::: dense_armor.control.cbf_filter

---

## Details

The filter was promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (SO-101, ALOHA). On every scenario the joint stays on
the safe side of the boundary, and the modification of the desired command is
minimally invasive: when the desired command is already safe, the filter returns it
unchanged; when it is not, the filter returns the closest safe command in Euclidean
norm.

**See also**: [Rate limiter](rate_limiter.md) — the kinematic (rate-of-change) complement
to this module's spatial (never-enter-a-region) guarantee; the two are not redundant.
[Passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md) — the
torque-level counterpart, which uses a CBF on the manipulability index instead of on a
geometric obstacle.
