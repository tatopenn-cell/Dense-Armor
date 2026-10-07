# Hybrid engine (Armatura's core)

`Armatura` is the front door for a 1D series: it takes an array, returns a cleaned
array. Behind it sits `hybrid_engine`, the module that actually decides, sample by
sample, whether a value is a genuine reading or a bad one — and if bad, what to put
in its place.

The engine is called *hybrid* because it combines two answers to the same question
("is this value abnormal?") and only replaces a sample when they agree: a **fixed
threshold** on the value itself, and a **binary trigger** that fires when the sample
enters a narrow band far from the local baseline.

## The signal

Same as on the previous page: a 100 Hz channel with one absurd value (9999) at sample
100 and one NaN at 200.

```python
import numpy as np
rng = np.random.default_rng(42)
x = 0.5 + 0.1 * rng.standard_normal(300)
x[100] = 9999.0
x[200] = float("nan")
```

## 1. Two thresholds, one decision

The engine keeps a causal window of past samples. For each new sample it computes a
robust centre `med` and a robust scale `S` (median and scaled MAD, same as everywhere
else in the library), then applies two rules:

- **Rule A (fixed threshold)**: replace the sample if `|x − med| > t · S`, with `t`
  a fixed constant chosen at construction.
- **Rule B (binary trigger)**: replace the sample if it falls inside a narrow band
  centred on a value far from `med`, expressed as an interval of width `w`.

The output value is the window median, but only when *both* rules agree. This is what
makes the engine robust to the two failure modes of a single rule:

- A fixed `t` alone catches big spikes but fails when the noise level itself changes:
  the same `t` becomes too tight or too loose.
- The binary trigger alone is scale-free but flags too aggressively on smooth signals
  that happen to have narrow peaks.

Together, they agree exactly on the samples that are unambiguously bad (a value
thousands of times the local scale, or NaN), and disagree on the ambiguous ones, which
pass through untouched.

## 2. Calling the engine directly

```python
from dense_armor.core.hybrid_engine import hybrid_shield

clean, k, info = hybrid_shield(x)
```

`clean` is the same array with the two broken samples replaced by the local median.
Everything else is bit-identical to `x`.

`radius=15` means the causal window holds 30 samples before the one being scored. The
same window convention as the anomaly detectors on [Streaming](../anomaly/streaming.md):
the sample being judged is never part of its own window.

## 3. Why "hybrid" is not "two detectors in series"

The two rules run *simultaneously*, on the same window, and their verdicts are combined
with a logical AND. The engine does not use the fixed threshold as a pre-filter and the
trigger as a post-check: if it did, the fixed threshold would silently disable the
trigger whenever the noise level fluctuated enough to make `t · S` bigger than the
band, which is exactly the failure mode this design avoids.

The AND is also why the engine never over-corrects a smooth signal. If the fixed
threshold fires but the trigger does not (a smooth but fast-moving signal that briefly
exceeds `t · S`), the value passes through. If the trigger fires but the threshold does
not (a narrow peak that stays inside `t · S` because the window scale is large), the
value also passes through. Only the both-fire case is a spike.

## 4. The nan handling

Non-finite values never enter the window. When the engine sees a NaN, it does not push
it into the buffer, does not compute a score for it, and returns the current window
median as the cleaned value. The next finite sample is then scored against the window
ending at the last finite sample — the NaN leaves no trace in the statistics.

This is why `Armatura.analizza` reports index 200 in `anomalies` even though the NaN
"never happened" from the engine's point of view: the index is reported, the value is
replaced, the window is untouched.

## 5. Provenance

The engine generalises the logic already verified in Dense-Evolution's own
`ia_utils.vector_healing.enhanced_dense_healing_hybrid`. The two rules, the AND, and
the NaN handling are the same; what changed is the interface (`process(value)` instead
of an array in, array out) and the window management (causal only).

## API reference

::: dense_armor.core.hybrid_engine

---

**See also**: [`Armatura`](armatura.md) — the wearable class built on this engine.
