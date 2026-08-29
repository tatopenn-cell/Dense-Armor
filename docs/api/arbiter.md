# Arbiter (per-point routing for Orca)

Routes each point to the right corrector instead of forcing one gate on an entire signal --
grew out of a real benchmark showing [`hybrid_engine`](hybrid_engine.md)-style rejection wins on
isolated impulses while [`pressure_valve`](robust_filters.md)'s tolerance wins on a sustained
level change, with neither beating the other everywhere. Classifies every point as
`clean`/`spike`/`regime` against a wide, causal reference window (only points before it, never
after -- a symmetric window straddling a real transition dilutes its own scale right where it
needs to detect it), then the run-length and internal coherence of consecutive deviant points
decides isolated impulse vs genuine regime change. Wired into
[`Orca.protect_and_forward(..., use_arbiter=True)`](orca.md); off by default.

::: dense_armor.utility.arbiter

---

**See also**: [Orca](orca.md) -- `use_arbiter=True` is where this module is actually used; [Robust filters](robust_filters.md) -- `pressure_valve`'s JSD-adaptive threshold inspired the wide reference window here.
