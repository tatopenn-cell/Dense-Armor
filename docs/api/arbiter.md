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

::: dense_armor.protect.arbiter

---

**See also**: [Orca](orca.md) -- `use_arbiter=True` is where this module is actually used; [Robust filters](robust_filters.md) -- `pressure_valve`'s JSD-adaptive threshold inspired the wide reference window here.

## Streaming Arbiter

`StreamingArbiter` (`dense_armor.protect.streaming_arbiter`) runs the same spike / regime
decision one sample at a time. A spike and the start of a regime look the same when they
begin, so each sample gets its final label within at most `max_delay` samples, as soon as its
run of anomalous points either returns (spike) or stays consistent (regime). With
`max_delay` long enough to cover the longest run, labels and corrected values equal
`classify_segments` + `route_and_correct`. `StreamingHealing` does the same for
`healing_filter`.

```python
import numpy as np
from dense_armor.protect.streaming_arbiter import StreamingArbiter

rng = np.random.default_rng(0)
x = list(rng.normal(0, 0.5, 200))
x[120] = 9.0
x[150:] = [v + 6.0 for v in x[150:]]
arbiter = StreamingArbiter(max_delay=40)
labels = {}
for v in x:
    for i, label, value in arbiter.update(v):
        labels[i] = label
for i, label, value in arbiter.flush():
    labels[i] = label
print(labels[120], labels[150], labels[160])
```

```
spike regime regime
```

::: dense_armor.protect.streaming_arbiter
