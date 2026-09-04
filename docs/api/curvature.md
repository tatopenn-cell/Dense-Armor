# Curvature (scaled distance-to-reference saturation)

A bounded, saturating measure of "how far a point has drifted from a reference" -- used
internally by `Orca`'s input shield (`last_kappa`) to score deviation magnitude in `[0,1)`.

Originally proposed (incorrectly) as a joint-limit-feasibility check. Verified directly
against real SO-101 joint data (Dense-Evolution-Discovery): the real robot really does
press against its physical limits during real pick-place episodes (`elbow_flex`, 120/303
real frames at its true hard-stop), so the underlying idea -- a graded proximity-to-limit
signal -- is real and useful. But the original unscaled formula saturates to ~1.0 within
~5 raw units of any reference regardless of the caller's actual units, so as a joint-limit
signal in real degrees it was a near-binary near/far indicator, not a graded one
(correlation with raw distance 0.235 on `elbow_flex`'s real 0.1-66.9 degree range). Fixed
by adding a `scale` parameter tying the saturation point to a real physical "warning zone"
width (correlation rises to 0.932 at `scale=15.0`) -- default `scale=1.0` reproduces the
original unscaled behavior exactly, so `Orca`'s existing call site is unaffected.

::: dense_armor.utility.curvature

---

**See also**: [Rate limiter](rate_limiter.md) and [CBF filter](cbf_filter.md) -- if the
real goal is *bounding or correcting* a command near a limit, not just scoring proximity
to one, those modules are the right tool.
