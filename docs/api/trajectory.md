# Trajectory (closed-form point-to-point generator)

`rate_limiter` bounds how fast a command can change and `cbf_filter` bounds where it can go,
but neither generates a reference to track in the first place -- `quintic_trajectory` fills
that gap with the simplest real, universal piece: a smooth, minimum-jerk-continuous path
between two points, for any number of joints at once.

```python
import numpy as np
from dense_armor.utility.trajectory import quintic_trajectory

t, q, v, a = quintic_trajectory(q0=[0.0], qf=[10.0], T=2.0)
```

This returns a smooth position/velocity/acceleration profile that starts and ends at rest.
Pass `v0`/`a0`/`vf`/`af` to start or end already moving instead of at rest -- useful for
chaining segments so the robot doesn't stop at every intermediate point.

```python
t, q, v, a = quintic_trajectory([0.0], [10.0], T=2.0, v0=[1.0], vf=[-2.0])
```

Works for any number of joints in one call -- pass `q0`/`qf` as arrays and every joint gets
its own independent polynomial over the same real time `T`.

::: dense_armor.utility.trajectory

---

**See also**: [Rate limiter](rate_limiter.md) and [CBF filter](cbf_filter.md) -- this module
generates a reference to follow; those two keep whatever follows it safe in speed and space.

## Details

Scoped down from two real papers proposing much larger URDF/dynamics-aware trajectory
optimizers (Lozer, Scalera, Boscariol & Gasparetto, *Robotics and Autonomous Systems*; Fried
& Paternain, arXiv:2412.07859) -- both read in full before writing this. Promoted from
Dense-Evolution-Discovery after validation on two independent real physical domains (SO-101,
ALOHA, 20 real joint excursions): the quintic's peak velocity is always lower than the real
recorded peak velocity for the same start, end, and real elapsed duration -- expected, since
it is the smoothest possible point-to-point path, not a bug. Single-segment point-to-point
only; chaining several segments across many waypoints is a real next step, not implemented
here.
