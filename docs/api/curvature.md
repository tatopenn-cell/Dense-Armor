# Curvature (scaled distance-to-reference)

`curvature` scores how far a point has drifted from a reference, as a single number
between 0 (no drift) and 1 (very far). `Orca`'s input shield uses it internally
(`last_kappa`) to size how large the current deviation is.

```python
import jax.numpy as jnp
from dense_armor.utility.curvature import curvature

x_now = jnp.array([5.0])
x_ref = jnp.array([0.0])
curvature(x_now, x_ref)
```

This returns a value close to 1 -- five units away from the reference is already "far" at
the default scale. `scale` sets how many real units count as "far": pass the width of the
zone you actually care about, and the score grows smoothly from 0 to 1 across it instead of
jumping to 1 almost immediately.

```python
curvature(x_now, x_ref, scale=15.0)
```

With `scale=15.0`, five units away from the reference is only partway into the zone, so the
score comes back closer to the middle of the range instead of near 1. Pick `scale` to match
the real units of `x_current`/`x_reference` -- e.g. degrees from a joint limit, meters from
a boundary -- and how wide a "getting close" warning zone should be in those units.

::: dense_armor.utility.curvature

---

**See also**: [Rate limiter](rate_limiter.md) and [CBF filter](cbf_filter.md) -- if the
real goal is bounding or correcting a command near a limit, not just scoring proximity to
one, those modules are the right tool.

## Details

`scale` defaults to `1.0`, which reproduces the function's exact previous behavior --
`Orca`'s existing call site (`orca.py`, `curvature(fh, c_chunk)`) is unaffected.

This parameter was added after checking a claim that `curvature` could work as a
joint-limit-feasibility check, against real SO-101 joint data (Dense-Evolution-Discovery).
The real robot does press against its physical limits during real pick-place episodes
(`elbow_flex`, 120/303 real frames at its true hard-stop -- a few degrees past the
reference URDF's declared limit, consistent with normal manufacturing tolerance on that
one unit). But at the default scale, the raw formula saturates to ~1.0 within about 5 raw
units of any reference regardless of the caller's physical units -- so as a joint-limit
signal in real degrees it was a near-binary near/far indicator, not a graded one:
correlation with raw distance was only 0.235 on `elbow_flex`'s real 0.1-66.9 degree range,
rising to 0.932 at `scale=15.0`.
