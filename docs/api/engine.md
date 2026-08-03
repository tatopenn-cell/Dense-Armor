# Adaptive engine (Orca Stage 1)

::: dense_armor.core.engine

---

**See also**: [`Orca`](orca.md), which uses `AdaptiveSignalStabilizer` as Stage 1, followed
by a Collatz-based gate (Stage 2) deciding how much to damp toward the clean reference. This
is also the engine exercised directly by the adversarial test suite
(`test/test_boundA-E.py`) behind the [robustness numbers on the home page](../index.md#real-tested-numbers).
