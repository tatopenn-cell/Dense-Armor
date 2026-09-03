# CUSUM (slow-drift detection) + ARL theory

A drift too small, at any single step, to cross [Arbiter](arbiter.md)'s instantaneous
threshold still accumulates: `cusum_detector` sums small deviations over time instead of
judging each point in isolation, so a slow sustained shift eventually trips it even when no
single point ever would. `one_sided_arl`/`two_sided_arl`/`detectability_report` are a
pre-flight estimate of how long that takes -- given a detector's real local noise level and a
candidate shift, how many samples until detection, or until a false alarm -- computable
before running a benchmark. Promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (lidar, accelerometer); see `detectability_report`'s own
docstring for the honest, mixed real-world result.

::: dense_armor.utility.cusum

---

**See also**: [Arbiter](arbiter.md) -- the instantaneous per-point detector this module is
the slow-drift complement to.
