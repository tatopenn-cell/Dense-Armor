# Rate limiter (causal command damping)

A robot's real motor cannot execute an unbounded instantaneous jump safely -- `rate_limited_follower`
bounds how fast an applied command can physically change (velocity + acceleration), instead of
trying to classify whether a deviation is a spurious spike or a genuine intended change (the
classification approach a neighbor-consensus filter would need, and which a causal version of it
failed at -- see the module's own docstring). Grounded in Berscheid & Kroger (2021)'s Ruckig
(RSS 2021, arXiv:2105.04830); this is a simpler velocity+acceleration-limited special case, not a
reimplementation of Ruckig's full time-optimal jerk synthesis. Promoted from Dense-Evolution-Discovery
after validation on two independent real physical domains (SO-101, ALOHA) -- see the module's own
docstring for the honest safety-vs-fidelity tradeoff found there.

::: dense_armor.utility.rate_limiter

---

**See also**: [CUSUM + ARL theory](cusum.md) -- if the real goal is recovering/classifying a signal
rather than bounding a command's rate of change, a detector is the right tool, not this module.
