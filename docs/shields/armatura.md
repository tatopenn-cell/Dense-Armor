# Armatura (1D series shield)

A loss curve that suddenly jumps to 10⁶. A sensor channel that drops a reading and
returns NaN. A token-likelihood stream that goes from 0.9 to 0 for one sample. These are
the kinds of single-value glitches that break a downstream pipeline silently: the model
does not complain, it just produces wrong numbers from that point on.

`Armatura` is the shield for a **single 1D series**. You give it an array; you get back
the same array, with the broken samples replaced by the local baseline, plus the list of
indices that were replaced.

## 1. Protect the series

One sensor channel at 100 Hz: noise around 0.5, one absurd value (9999) at sample 100, one NaN
at sample 200.

```python
import numpy as np
from dense_armor import Armatura

rng = np.random.default_rng(42)
x = 0.5 + 0.1 * rng.standard_normal(300)
x[100], x[200] = 9999.0, float("nan")
clean, K, anomalies = Armatura(livello_ia=0.0).analizza(x)
print(len(clean), anomalies, round(float(clean[100]), 3), K[100], K[50])
```

```
300 [100, 200] 0.463 1.0 0.0
```


`clean` has the same shape as `x`: the two broken samples are replaced by the local baseline
(9999 becomes 0.463). Ordinary samples are kept, except those the engine's trigger classifies
as static, which are replaced by the local median without being listed in `anomalies` (35 of the
300 here, see [Hybrid engine](hybrid_engine.md), step 3). `anomalies` lists the replaced indices,
`[100, 200]`. `K` is one flag per sample from the engine's trigger: 1 where the spike rule fired
(sample 100), 0 elsewhere; the NaN is cleaned before the trigger, so sample 200 is in `anomalies`
with `K = 0`.

`livello_ia = 0` means "actively filter". `livello_ia = 1` means "only mark, do not
change the values" — useful when you want to log what the shield *would* have done
without altering the stream.

## 2. What "local baseline" means

`Armatura` looks at a window of samples **before** the one being scored (never after): its
baseline is the mean of those previous samples (radius `min(20, max(3, n // 3))`), and a
binary trigger decides whether the sample is an impulse. If it is, it is replaced by the
baseline; otherwise it passes through unchanged. The first two samples are never tested (the
trigger needs two previous points), so a glitch there must be caught by another check.

The consequence, on the running example:

- The value 9999 at sample 100 is far above the baseline of the previous samples, so it is
  replaced. Index 100 appears in `anomalies`.
- The NaN at sample 200 never enters the window (the shield drops non-finite samples
  before the comparison), so the next finite sample is scored against the window ending
  at sample 199. Index 200 appears in `anomalies`, and `clean[200]` is the baseline.

## 3. What it is not for

`Armatura` is a **point-level** shield. It removes isolated spikes and non-finite values.
It does not remove a slow drift, and it does not classify a regime change: those need a
different tool (see [CUSUM](../drift/cusum.md) and [Arbiter](../protect/arbiter.md)).
If your signal is a step change that is genuinely intended, `Armatura` leaves it alone
after the first few samples, because after those samples the step is the new local
baseline.

## 4. The engine inside

`Armatura` is built on `core.hybrid_engine`, the binary-trigger engine that generalises
the logic already verified in Dense-Evolution's own
`ia_utils.vector_healing.enhanced_dense_healing_hybrid`. See
[Hybrid engine](hybrid_engine.md) for the internals.

## API reference

::: dense_armor.armatura

---

**See also**: [Hybrid engine](hybrid_engine.md) — the `core/hybrid_engine.py` module
`Armatura` is built on. [Orca](orca.md) — the full input + output shield for an entire
model, if you have a model rather than a bare series.
