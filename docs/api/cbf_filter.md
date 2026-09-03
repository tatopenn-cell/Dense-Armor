# CBF filter (geometric command safety)

A command moving at a perfectly safe, bounded velocity straight into an obstacle is still
dangerous -- `rate_limiter.md` bounds rate of change, this module bounds WHERE a command can
go. `cbf_safety_filter`/`cbf_filtered_trajectory` implement a Control Barrier Function (CBF)
safety filter: never let the applied command enter a forbidden region, with a minimally
invasive correction when it would. Grounded in Ames et al. (2019)'s CBF-QP theory
(2019 ECC, arXiv:1903.11199) -- the same theory SAFER-Splat uses, applied to a known
geometric obstacle instead of SAFER-Splat's GPU-bound Gaussian-Splatting perception (not
available on every machine -- see the module's own docstring). Promoted from
Dense-Evolution-Discovery after validation on two independent real physical domains
(SO-101, ALOHA) -- see the module's own docstring for the real numbers, including a real
numerical finding about discrete-time sub-stepping.

::: dense_armor.utility.cbf_filter

---

**See also**: [Rate limiter](rate_limiter.md) -- the kinematic (rate-of-change) complement
to this module's spatial (never-enter-a-region) guarantee; the two are not redundant.
