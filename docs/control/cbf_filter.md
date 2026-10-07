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

A joint at position `x` is commanded to move with velocity `u_des`. Somewhere ahead there is
an obstacle at position `obstacle`, and the joint must stay at least `safe_dist` away from it.

## 1. Apply the filter

```python
from dense_armor.utility.control.cbf_filter import cbf_safety_filter

print(cbf_safety_filter(0.0, 1.0, obstacle=0.5, safe_dist=0.1))
print(cbf_safety_filter(0.35, 1.0, obstacle=0.5, safe_dist=0.1))
```

```
0.24
0.04166666666666667
```

Far from the obstacle (x = 0) the command 1.0 is only trimmed to 0.24; at x = 0.35, close to the safety
boundary 0.5 − 0.1 = 0.4, it is cut to 0.042.

When the desired command is already safe, the filter returns it unchanged. When it would bring
the joint too close, the filter returns the smallest change that keeps it safe.

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

- `h` — the barrier function; its sign tells you which side of the boundary you are on.
- `L_f h`, `L_g h` — how `h` changes along the drift and along the input. For the
  single-integrator joint $\dot x = u$ used here, $L_f h = 0$.
- `α` — a strictly increasing function with `α(0) = 0`; here $\alpha(h) = k\,h$ with
  `k = alpha_gain`: larger `k` lets the joint approach the boundary faster.
- `u` — the joint velocity command (the output).

### The QP and its closed form

At each call the filter solves

$$\min_u \tfrac12 \lVert u - u_{des}\rVert^2 \quad \text{s.t.} \quad L_f h + L_g h\, u \ge -\alpha(h),$$

the CBF-QP of Ames et al. (2019). With one input and one constraint it has a closed form, the
min-norm controller. In this module the safe set is "at least `safe_dist` from `obstacle`":

$$h(x) = (x - o)^2 - d^2, \qquad L_g h = 2(x - o), \qquad u = \begin{cases} u_{des} & \text{if } L_g h\,u_{des} \ge -k\,h \ -k\,h / L_g h & \text{otherwise,} \end{cases}$$

with $o$ = `obstacle`, $d$ = `safe_dist`, $k$ = `alpha_gain`.

## 3. Hand case

The two calls of step 1, with $o = 0.5$, $d = 0.1$, $k = 1$, $u_{des} = 1$:

- $x = 0$: $h = 0.25 - 0.01 = 0.24$, $L_g h = -1$. The test $L_g h\,u_{des} = -1 \ge -0.24$ fails, so
  $u = -0.24 / -1 = 0.24$.
- $x = 0.35$: $h = 0.0225 - 0.01 = 0.0125$, $L_g h = -0.3$, $u = -0.0125 / -0.3 = 0.0417$.

The closer the joint gets to the boundary at 0.4, the smaller the allowed velocity: exactly the
numbers printed in step 1.

## 4. The discrete-time issue

A CBF guarantees $h \ge 0$ **in continuous time**. In a control loop one tick lasts `dt`, and a
single step with a constant command can cross the boundary even if the constraint held at the
start of the tick. `cbf_safety_filter_live` therefore splits the tick into `n_substeps`
(default 20) sub-steps, applies the filter at each one, and returns the average velocity over
the tick. Without the sub-steps the joint could cross the boundary by a small amount when it
started close to it; this was found in the validation on SO-101 and ALOHA.

## 5. Real-time use

For a live ROS2 or Ignition loop that reacts to one sensor callback at a time,
`cbf_safety_filter_live` is the same filter packaged for a real `dt` per tick, instead
of filtering a whole pre-recorded array off-line.

```python
from dense_armor.utility.control.cbf_filter import cbf_safety_filter_live

x = 0.0
for _ in range(200):
    x += 0.01 * cbf_safety_filter_live(x, 1.0, dt=0.01, obstacle=0.5, safe_dist=0.1)
print(round(x, 4))
```

```
0.2939
```

Two seconds of a control loop at 100 Hz that keeps asking for velocity 1.0: the position creeps towards
the boundary 0.4 (0.2939 after 200 ticks) and never crosses it.

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

::: dense_armor.utility.control.cbf_filter

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
