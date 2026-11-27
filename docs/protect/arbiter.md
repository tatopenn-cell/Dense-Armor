# Arbiter (per-point routing)

When a reading looks wrong, there are different right answers: a single glitch should be replaced, a real change should be kept. The Arbiter decides, reading by reading, which case it is, and treats it accordingly.

A shield that always replaces deviating samples looks wrong on a real robot: it will
smooth out the intentional step change the operator just commanded. A shield that never
replaces them looks wrong on a noisy sensor: it lets every glitch through.

The Arbiter decides, point by point, which is which. It labels every sample
**clean**, **spike**, or **regime**, and routes each label to the right corrector:

- `clean` — passes through whatever the standard shield produced.
- `spike` — hard-rejected to the median of the recent causal window.
- `regime` — passed through **raw**, fully trusted.

The examples use a joint velocity at 100 Hz (noise 0.1 around 0.5) with a one-sample collision
at sample 1500 and a step to a new level, +1.0, from sample 2000.

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
import numpy as np
rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
v[2000:] += 1.0
from dense_armor.core.online.protect.arbiter import classify_segments

lab, dev, unc = classify_segments(v, radius=20, ref_mult=3)
print(lab[1500], np.flatnonzero(lab == "regime")[[0, -1]], (lab == "spike").sum())
```

```
spike [1999 2026] 26
```

It returns the label of each sample, its robust deviation `z`, and an uncertainty in [0, 1]
(highest near the `n_sigmas` boundary). The collision is a spike; samples 1999–2026 are one long,
coherent, persistent run, so a regime; afterwards the reference has caught up and samples are clean
again. The other 26 spikes are isolated noise values beyond 3 sigma, out of 3,000.

## 3. Route the labels

```python
import numpy as np
rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
v[2000:] += 1.0
from dense_armor.core.online.protect.arbiter import route_and_correct

corrected, lab, dev = route_and_correct(v, radius=20, ref_mult=3)
print(round(float(corrected[1500]), 3), np.allclose(corrected[lab == "regime"], v[lab == "regime"]))
```

```
0.489 True
```

`corrected` is the same array with each label handled by its rule:

- `spike` → replaced by the median of the causal window ending just before the run.
- `regime` → unchanged.
- `clean` → unchanged.

The collision becomes 0.489, the median before it; regime samples are left as measured (to the rounding of the default float32).

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
from dense_armor.core.online.protect.orca import Orca

import numpy as np
rng = np.random.default_rng(0)
data = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data[0, 50] = 999.0
orca = Orca(min_free_ram_percentage=0.05)
out = orca.protect_and_forward(lambda z: 2.0 * z, data, use_arbiter=True)
print(out.shape, orca.etichette_arbitro.shape)
```

## 6. Streaming Arbiter (bounded delay)

The batch `classify_segments` needs to look *after* a run of deviating samples to decide
whether it settled — which means it is not causal. `StreamingArbiter` gives the same
answer with a bounded delay: each sample's final label is committed at most `max_delay`
samples after it arrives, once the run has closed and enough context is available.

```python
import numpy as np
rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
v[2000:] += 1.0
from dense_armor.core.online.protect.streaming_arbiter import StreamingArbiter

arb = StreamingArbiter(radius=20, ref_mult=3, max_delay=200)
lab = {j: l for x in v for j, l, _ in arb.update(x)}
lab.update({j: l for j, l, _ in arb.flush()})
print(len(lab), lab[1500], lab[2000])
```

`update(x)` returns a list of `(index, label, corrected_value)` for the samples whose
final label was decided at this step; `flush()` finalises whatever is still pending at
the end of the stream.

With `max_delay` at least `longest_run + radius * ref_mult`, the labels and corrected
values equal the batch `classify_segments` + `route_and_correct` on the same series.
Smaller `max_delay` trades exactness for latency: labels near run boundaries may change.

## 7. StreamingHealing

`StreamingHealing` is the same bounded-delay idea applied to the healing filter
(`dense_armor.core.online.protect.healing.healing_filter`): a point is judged against a wide local
baseline and replaced when the deviation is not shared by the majority of a narrow
window of neighbours. Each sample gets its corrected value as soon as the wide symmetric
window around it is fully available.

```python
import numpy as np
rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
v[2000:] += 1.0
from dense_armor.core.online.protect.streaming_arbiter import StreamingHealing

sh = StreamingHealing(radius=2, wide_mult=3, max_delay=30)
out = [o for x in v for o in sh.update(x)] + sh.flush()
print(len(out), out[0])
```

## 8. Hand case

Window of 9 samples `[10, 11, 9, 10, 12, 10, 11, 9, 10]`, then a run of 3 samples
`[15, 16, 15.5]`, then 4 samples `[15, 15, 15, 15]`.

- `med` of the pre-run window is 10, `MAD = 1`, `S = 1.4826`.
- Score for the first run sample: `|15 − 10| / 1.4826 = 3.37 > 3` → deviating.
- Same for the other two run samples.
- Run length = 3 > `spike_run_max = 2`: it is not a spike.
- `std(run) = std([15, 16, 15.5]) = 0.41`, `|median(run) − med| = |15.5 − 10| = 5.5`,
  `0.5 · 5.5 = 2.75`; `0.41 < 2.75`, so the run is internally coherent.
- Median of 4 post-run samples is 15; `|15 − 15.5| = 0.5 < |15 − 10| = 5`: it
  persists.
- Label: **regime**. The 3 samples pass through unchanged.

Change the three samples to `[15, 8, 16]`: 15 and 16 deviate, but 8 does not
(`|8 − 10| / 1.4826 = 1.35 < 3`), so there are two runs of length 1, not one run of three.
Labels: **spike, clean, spike**: 15 and 16 are replaced by 10, the 8 is kept.

## 9. Any detector of the library

The Arbiter decides with its own window by default; with `detector=` any detector of the library can decide, and the Arbiter keeps its rule on the length of the run.

### 9.1 A point detector decides which samples deviate

A price series with one bad tick at the sample 60 and a jump of the level from the sample 150 to the sample 199. A point detector is one with `score_one` and `learn_one`: a sample deviates when its score is above `threshold`.

```python
import numpy as np
from dense_armor.core.online.protect.arbiter import classify_segments
from dense_armor.core.online.anomaly.filters import HampelScorer

r = np.random.default_rng(0)
px = r.normal(0, 1, 300)
px[60] += 15.0
px[150:200] += 8.0
lab, dev, unc = classify_segments(px, detector=HampelScorer(), threshold=3.0)
print([i for i, l in enumerate(lab) if l == 'spike'], [i for i, l in enumerate(lab) if l == 'regime'][:3])
```

```
[4, 6, 60, 219, 270, 284] [150, 151, 152]
```

The sample 60 is a spike. The regime labels start at the sample 150, where the level moved. The detector adapts to the new level, so a run is labelled regime only while its samples still deviate. The window of the Arbiter is not used to decide which samples deviate, but it still gives the reference level that the run is compared with.

### 9.2 The threshold of each detector

`threshold` is the score above which a sample deviates. If it is not given, the Arbiter takes `detector.threshold` when the detector has one, and `n_sigmas` otherwise. The scores of the detectors are not on one scale, so the default is not good for all of them. On the series of 9.1, with the threshold of each detector:

| detector | threshold used | label of the sample 60 | samples labelled spike | samples labelled regime |
|---|---|---|---|---|
| none (the window of the Arbiter) | - | spike | 5 | 23 |
| `HampelScorer` | 3 | spike | 6 | 16 |
| `TukeyScorer` | 3.0 (`n_sigmas`) | spike | 1 | 8 |
| `ChauvenetScorer` | 0.5 | clean | 205 | 72 |
| `SigmaClipScorer` | 3.0 (`n_sigmas`) | spike | 7 | 3 |
| `StreamingDeviationScorer` | 3.0 (`n_sigmas`) | spike | 5 | 23 |
| `LODA` | 4 | clean | 19 | 0 |
| `HalfSpaceTrees` | 1e+09 | clean | 0 | 0 |
| `LocalOutlierFactor` | 1.5 | spike | 23 | 19 |
| `OnlineRobustMahalanobis` | 7 | clean | 0 | 0 |
| `OneClassSGD` | 1e+09 | clean | 0 | 0 |
| `OnlineIsolationForest` | 0.6 | spike | 81 | 0 |

With their own threshold, six detectors label the sample 60 as spike (`HampelScorer`, `TukeyScorer`, `SigmaClipScorer`, `StreamingDeviationScorer`, `LocalOutlierFactor` and `OnlineIsolationForest`) and five leave it clean. `HalfSpaceTrees` and `OneClassSGD` have a threshold of 1e9, so no sample deviates, and `OnlineRobustMahalanobis` with 7 flags none either: for them `threshold` must be given. `LODA` (threshold 4) flags 19 samples and not the sample 60, `ChauvenetScorer` (threshold 0.5) flags 205 samples and not the sample 60, `OnlineIsolationForest` (0.6) flags 81 and `LocalOutlierFactor` (1.5) flags 23: their score is not in the unit of a threshold in sigma, so each one needs the threshold of its own page.

### 9.3 An event detector confirms a regime

An event detector is one with `update` and `drift_detected`. It decides nothing about single samples: when it signals during a run of deviating samples, that run is a regime, even a short one. The label of the sample 60, with no detector, with `ADWIN` and with `BayesianOnlineChangepoint`:

```python
import numpy as np
from dense_armor.core.online.protect.arbiter import classify_segments
from dense_armor.core.online.drift import ADWIN, BayesianOnlineChangepoint

r = np.random.default_rng(0)
px = r.normal(0, 1, 300)
px[60] += 15.0
px[150:200] += 8.0
for d in (None, ADWIN(), BayesianOnlineChangepoint()):
    lab = classify_segments(px, detector=d)[0]
    print(type(d).__name__, lab[60])
```

```
NoneType spike
ADWIN spike
BayesianOnlineChangepoint regime
```

`ADWIN` does not signal during any run, so the labels are those of the Arbiter alone. `BayesianOnlineChangepoint` signals on the single bad tick, so it turns the spike of the sample 60 into a regime: a detector that reacts to one outlier should not be used here.

### 9.4 In a stream

`StreamingArbiter(detector=..., threshold=...)` decides with the same detector. The detector is used as it is given, so it learns the stream; `classify_segments` works on a copy. With `max_delay` large enough the labels are the ones of the batch function:

```python
import numpy as np
from dense_armor.core.online.protect.arbiter import classify_segments
from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.online.protect.streaming_arbiter import StreamingArbiter

r = np.random.default_rng(0)
px = r.normal(0, 1, 300)
px[60] += 15.0
px[150:200] += 8.0
sa = StreamingArbiter(max_delay=400, detector=HampelScorer(), threshold=3.0)
out = [e for v in px for e in sa.update(float(v))] + sa.flush()
batch = classify_segments(px, detector=HampelScorer(), threshold=3.0)[0]
print([l for _, l, _ in sorted(out)] == list(batch))
```

```
True
```

### 9.5 The same on a robot

An accelerometer channel with one glitch at the sample 60 and a change of level from the sample 150. The code is the same, with the names of the robot.

```python
import numpy as np
from dense_armor.core.online.protect.arbiter import classify_segments
from dense_armor.core.online.anomaly.filters import HampelScorer

r = np.random.default_rng(0)
ax = r.normal(0, 1, 300)
ax[60] += 15.0
ax[150:200] += 8.0
lab, dev, unc = classify_segments(ax, detector=HampelScorer(), threshold=3.0)
print([i for i, l in enumerate(lab) if l == 'spike'], [i for i, l in enumerate(lab) if l == 'regime'][:3])
```

```
[4, 6, 60, 219, 270, 284] [150, 151, 152]
```

Any other object is not used: the Arbiter works as if no detector had been given.

## API reference

::: dense_armor.core.online.protect.arbiter

::: dense_armor.core.online.protect.streaming_arbiter

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
