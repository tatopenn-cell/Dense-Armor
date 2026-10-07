# Arbiter (per-point routing)

A shield that always replaces deviating samples looks wrong on a real robot: it will
smooth out the intentional step change the operator just commanded. A shield that never
replaces them looks wrong on a noisy sensor: it lets every glitch through.

The Arbiter decides, point by point, which is which. It labels every sample
**clean**, **spike**, or **regime**, and routes each label to the right corrector:

- `clean` — passes through whatever the standard shield produced.
- `spike` — hard-rejected to the median of the recent causal window.
- `regime` — passed through **raw**, fully trusted.

## The signal

The same 100 Hz joint velocity. Two events: a single-sample collision spike at 15 s, and
a slow drift that starts at 20 s and stabilises at a new level.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2
```

## 1. The three labels

The Arbiter uses the same robust-deviation score as every other detector in the package
(see [Streaming](../anomaly/streaming.md)):

```
med = median(window)
S   = 1.4826 · median(|w − med|)
z   = |x − med| / S
```

A sample is **deviating** when `z > n_sigmas` (default `n_sigmas = 3`). Then the run of
consecutive deviating samples decides the label:

- **run length ≤ `spike_run_max`** (default 2) → **spike**
- **longer run, and it settles** → **regime**
- otherwise (longer run that does *not* settle) → **spike** (the whole run is treated as
  noise)

"Settles" is decided by two conditions on the run:

1. **Internal coherence**: the run's own spread is small compared to its distance from
   the pre-run median:
   `std(run) < 0.5 · |median(run) − med|`.
2. **Persistence**: the median of a short window **after** the run is closer to the run's
   median than to the pre-run median.

The two together are what separates "the signal has moved to a new level" from "the
sensor briefly went haywire and came back".

## 2. Classify the whole series

```python
from dense_armor.utility.protect.arbiter import classify_segments

labels = classify_segments(v, radius=20, ref_mult=3, n_sigmas=3.0,
                           spike_run_max=2)
```

`labels` is an array of `"clean"`, `"spike"` or `"regime"`, one per sample.

On the running example:

- index 1500 → `"spike"` (single-sample run, `run length 1 ≤ 2`)
- indices 2000..2099 → `"regime"` (long run, coherent, persists)
- everything else → `"clean"`

## 3. Route the labels

```python
from dense_armor.utility.protect.arbiter import route_and_correct

corrected, labels, dev = route_and_correct(v, radius=20, ref_mult=3)
```

`corrected` is the same array with each label handled by its rule:

- `spike` → replaced by the median of the causal window ending just before the run.
- `regime` → unchanged.
- `clean` → unchanged.

On the running example, `corrected[1500]` is about 0.5 (the collision removed) and
`corrected[2000:2100]` is identical to `v[2000:2100]` (the new level kept).

## 4. The reference window is causal

The window used to score every sample contains only the samples **before** it, never
after. This matters more than it looks: a symmetric window straddling a real regime
change dilutes its own MAD exactly where the detector needs it, because half the window
is on the other side of the change. Causal-only means the pre-change scale is used to
score the first post-change samples, which is what lets the run be detected as a run.

The same choice was made for every streaming detector in the package.

## 5. In Orca

`Orca.protect_and_forward(..., use_arbiter=True)` runs the Arbiter between Stage 1
(the adaptive stabilizer) and Stage 2 (the Collatz gate). Verified on the same 7
scenarios `test/testKalman.py` uses: never worse than the default (no Arbiter), better
on 5 of 7.

```python
from dense_armor.utility.protect.orca import Orca

orca = Orca()
protected = orca.protect_and_forward(my_model, corrupted, use_arbiter=True)
orca.etichette_arbitro
orca.incertezza_arbitro_media
```

## 6. Streaming Arbiter (bounded delay)

The batch `classify_segments` needs to look *after* a run of deviating samples to decide
whether it settled — which means it is not causal. `StreamingArbiter` gives the same
answer with a bounded delay: each sample's final label is committed at most `max_delay`
samples after it arrives, once the run has closed and enough context is available.

```python
from dense_armor.utility.protect.streaming_arbiter import StreamingArbiter

arb = StreamingArbiter(radius=20, ref_mult=3, max_delay=40)
labels = {}
for i, x in enumerate(v):
    for j, lab, _ in arb.update(x):
        labels[j] = lab
for j, lab, _ in arb.flush():
    labels[j] = lab
```

`update(x)` returns a list of `(index, label, corrected_value)` for the samples whose
final label was decided at this step; `flush()` finalises whatever is still pending at
the end of the stream.

With `max_delay` at least `longest_run + radius * ref_mult`, the labels and corrected
values equal the batch `classify_segments` + `route_and_correct` on the same series.
Smaller `max_delay` trades exactness for latency: labels near run boundaries may change.

## 7. StreamingHealing

`StreamingHealing` is the same bounded-delay idea applied to the healing filter
(`dense_armor.utility.protect.healing.healing_filter`): a point is judged against a wide local
baseline and replaced when the deviation is not shared by the majority of a narrow
window of neighbours. Each sample gets its corrected value as soon as the wide symmetric
window around it is fully available.

```python
from dense_armor.utility.protect.streaming_arbiter import StreamingHealing

sh = StreamingHealing(radius=2, wide_mult=3, max_delay=30)
out = []
for x in v:
    out.extend(sh.update(x))
out.extend(sh.flush())
```

## Hand case

Window of 9 samples `[10, 11, 9, 10, 12, 10, 11, 9, 10]`, then a run of 3 samples
`[15, 16, 15.5]`, then 4 samples `[15, 15, 15, 15]`.

- `med` of the pre-run window is 10, `MAD = 1`, `S = 1.4826`.
- Score for the first run sample: `|15 − 10| / 1.4826 = 3.37 > 3` → deviating.
- Same for the other two run samples.
- Run length = 3 > `spike_run_max = 2`: it is not a spike.
- `std(run) = std([15, 16, 15.5]) = 0.5`, `|median(run) − med| = |15.5 − 10| = 5.5`,
  `0.5 · 5.5 = 2.75`; `0.5 < 2.75`, so the run is internally coherent.
- Median of 4 post-run samples is 15; `|15 − 15.5| = 0.5 < |15 − 10| = 5`: it
  persists.
- Label: **regime**. The 3 samples pass through unchanged.

Change the run to `[15, 8, 16]` (same length, incoherent): `std = 3.6 > 0.5 · |11 − 10| =
0.5`, so the run is **not** internally coherent → label **spike**. All three samples are
replaced by the pre-run median (10).

## API reference

::: dense_armor.utility.protect.arbiter

::: dense_armor.utility.protect.streaming_arbiter

---

## Details

The Arbiter grew out of a real benchmark where `hybrid_engine`-style rejection won on
isolated impulses while `pressure_valve`'s tolerance won on a sustained level change,
with neither beating the other everywhere. The obvious answer — pick one per situation —
required a classifier for "situation", and the run-length + coherence rule is what the
benchmark showed to be the right classifier for the cases that appeared.

The reference window is causal for a specific reason found during validation: a
symmetric window straddling a real transition dilutes its own MAD at exactly the
positions where the transition is being classified, so a run that starts right at the
change is scored against a scale that is half on the old level and half on the new, and
neither side wins. Causal-only means the pre-change scale is used for the first
post-change samples, which is what lets the run be recognized as a run.
