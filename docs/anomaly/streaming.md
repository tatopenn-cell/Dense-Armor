# Streaming detection

A streaming detector answers, for each new sample, one question: **is this sample further
from the recent past than normal noise would explain?** It looks only at the samples
*before* the new one, never at the future. On a robot that runs at 100 Hz this is the
only honest option: the sample at time *t* has to be judged at time *t*.

## The signal

A joint velocity at 100 Hz, nominal 0.5 rad/s. At 15 s a collision gives one large spike;
from 20 s a worn gear makes the signal drift upward.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
```

## 1. Score one sample

The detector keeps the last `radius * ref_mult` samples in a window. For each new sample
`x`, it computes:

- `med` — the median of the window,
- `S` — the *scaled median absolute deviation*: `S = 1.4826 × median(|w − med|)`.

The score is `z = |x − med| / S`. The 1.4826 factor makes `S` equal to the standard
deviation when the window is normal noise; without it the score would be in units of MAD
instead of sigmas, and the thresholds would be unreadable.

**Hand case.** Window of 9 samples `[10, 11, 9, 10, 12, 10, 11, 9, 10]`, new sample
`x = 30`. The median is 10. The absolute deviations from 10 are
`[0, 1, 1, 0, 2, 0, 1, 1, 0]`; their median is 1. So `S = 1.4826 × 1 = 1.4826`, and the
score is `z = |30 − 10| / 1.4826 = 13.49`. The window is quiet (MAD = 1) and the new
sample is 20 units away: the detector says "far".

```python
from dense_armor.anomaly.streaming import StreamingDeviationDetector

det = StreamingDeviationDetector(radius=5, ref_mult=3)
for x in v[:200]:
    det.update(x)
det.update(v[1500])
```

The window holds `5 × 3 = 15` samples; on normal operation the score is under 3, on the
collision spike (2.5 rad/s against a 0.5-rad/s window) it is in the tens.

## 2. Score the whole stream

To score every sample, feed the same detector one value at a time and store the flags.
Because the window grows one sample at a time and never looks ahead, the cost per sample
is `O(window)`.

```python
det = StreamingDeviationDetector(radius=5, ref_mult=3)
flags = np.zeros_like(v, dtype=bool)
for i, x in enumerate(v):
    flags[i] = det.is_outlier(x)
    det.update(x)

t[flags]
```

On the running example the flags are a single sample at index 1500 (the collision) and,
later, a few clustered points around the start of the drift. The exact list is printed by
the code; the point is that the collision is one sample, the drift is a run.

## 3. The four causal filters

The score above is the robust deviation used by `classify_segments` in the batch
[Arbiter](../protect/arbiter.md). The four classic robust filters of the package (Hampel,
Tukey, Chauvenet, sigma clipping) are available as streaming scorers with the *same*
interface, on the same causal window (`2 * radius` samples before the value being
scored). Only Hampel is shown here; the other three are in Details.

**Hampel** flags a sample when `|x − med| > n · S`, with `n = 3` by default. On the hand
case above, `20 > 3 × 1.4826 = 4.4478`, so the sample is flagged.

```python
from dense_armor.anomaly.filters import HampelScorer

s = HampelScorer(radius=15, n_sigmas=3.0)
for x in v[:200]:
    s.learn_one({"v": float(x)})
s.is_outlier({"v": 30.0})
```

`radius=15` is the *half*-width: the window holds `30` past samples. The threshold
`n_sigmas` is the tuning knob; higher values flag fewer points, at the price of missing
smaller anomalies.

## 4. Many channels at once

A robot has six joints, an IMU has three axes. Scoring each channel with its own detector
works, but a fault that shows up as an unusual *combination* of channels (joint 1 high
and joint 2 low at the same time, each alone plausible) is invisible to per-channel
detectors. `MultiChannelStreamingDeviationDetector` runs one detector per channel and
returns a per-channel verdict on each sample.

```python
from dense_armor.anomaly.streaming import MultiChannelStreamingDeviationDetector

det = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=3)
for qd in qd_stream:
    flags = det.update(qd)
```

`qd_stream` is a stream of 6-element joint velocity arrays, one per sample. This is
ergonomics, not a new algorithm: each channel keeps its own window and its own baseline.

## 5. Many channels together: online robust Mahalanobis

To catch *combinations* of channels rather than individual channels, use the online robust
Mahalanobis distance. It tracks, one sample at a time, the *geometric median* (a robust
centre) and the *median covariation matrix* (a robust spread), and scores each sample by
its Mahalanobis distance from them.

![Two joints, in-distribution and outlier](../assets/streaming/mahalanobis.png)

```python
import numpy as np
from dense_armor.anomaly.mahalanobis import OnlineRobustMahalanobis

m = OnlineRobustMahalanobis(feature_keys=["j0", "j1"])
for qd in two_joint_stream:
    m.learn_one({"j0": qd[0], "j1": qd[1]})
m.is_outlier({"j0": 0.0, "j1": 0.0})
```

The score is the Mahalanobis distance from the sample to the tracked geometric median,
in the metric of the tracked covariance. On well-behaved 2-channel normal data it stays
around 3; a sample that is far in *both* channels at once goes above the threshold
(`threshold=7.0` by default).

### Hand case, two channels

Two channels, 100 samples of `N(0, I)`. The geometric median converges to `(0, 0)` and
the covariation matrix to `I`. A new sample `(0, 0)` scores about 0; a new sample
`(40, −40)` scores `sqrt(40² + 40²) = 56.6` — far over any plausible threshold.

## Performance

On one CPU core, µs per sample, window 30:

| detector | µs / sample |
|---|---|
| streaming deviation | 15 |
| Hampel | 90 |
| Tukey | 75 |
| Chauvenet | 28 |
| sigma clipping | 72 |
| online robust Mahalanobis, 6 channels | 95 |

At 100 Hz a control loop has 10 ms per cycle; all of them fit with three orders of
magnitude to spare. The window size dominates the cost: doubling `radius` roughly
doubles the per-sample time.

## Details

### Tukey fences

`Q1`, `Q3` of the window, `IQR = Q3 − Q1`, normal range `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
Flag when the value is outside. On the hand case window above, sorted
`[9, 9, 10, 10, 10, 10, 11, 11, 12]`, `Q1 = 10`, `Q3 = 11`, `IQR = 1`, fences
`[8.5, 12.5]`, and the new sample 30 is outside.

```python
from dense_armor.anomaly.filters import TukeyScorer

s = TukeyScorer(radius=15)
for x in v[:200]:
    s.learn_one({"v": float(x)})
s.is_outlier({"v": 30.0})
```

### Chauvenet

Score is `N · P(|Z| ≥ z)` with `z = |x − mean| / std` on the window. Unlike the other
three, **low means anomalous**: the criterion rejects when `N · P < 0.5`. On the hand
case window, `mean = 10.222`, `std = 0.916`, `z = 21.59`, `N = 10`,
`N · erfc(z/√2) ≈ 2.4e-102 < 0.5`, so the new sample is rejected. The `score_one` method
returns `N · P`, so use `is_outlier`, or threshold `score < 0.5`.

```python
from dense_armor.anomaly.filters import ChauvenetScorer

s = ChauvenetScorer(radius=15)
for x in v[:200]:
    s.learn_one({"v": float(x)})
s.is_outlier({"v": 30.0})
```

### Sigma clipping

Iteratively removes points beyond `n_sigmas`, recomputes mean and std, until stable or
`max_iters`; then scores `|x − mean_clean| / std_clean`.

```python
from dense_armor.anomaly.filters import SigmaClipScorer

s = SigmaClipScorer(radius=15, n_sigmas=3.0)
for x in v[:200]:
    s.learn_one({"v": float(x)})
s.is_outlier({"v": 30.0})
```

### River-compatible scorer

`StreamingDeviationScorer` (`dense_armor.anomaly.deviation`,
`pip install dense-armor[river]`) exposes the same causal deviation as a river anomaly
detector: `score_one` returns `|x − med| / S` on the window learned so far, `learn_one`
adds the value. A score above `n_sigmas` is exactly a `StreamingDeviationDetector` flag.

```python
from dense_armor.anomaly.deviation import StreamingDeviationScorer

model = StreamingDeviationScorer(radius=5, ref_mult=2)
for x in v[:200]:
    model.learn_one({"v": float(x)})
model.score_one({"v": 30.0})
```

### Precision and recall on the four causal filters

5,000 seeded samples, 50 spikes of size 5–15 (1 % contamination), window 30 (radius 15):

| Scorer | stationary noise (σ = 0.5): precision / recall | sine (amplitude 5, period 500) + same noise: precision / recall | µs per sample |
|---|---|---|---|
| Hampel | 0.459 / 1.000 | 0.318 / 1.000 | 90 |
| Tukey | 0.303 / 1.000 | 0.195 / 1.000 | 75 |
| Chauvenet | 0.282 / 0.980 | 0.167 / 1.000 | 28 |
| SigmaClip | 0.588 / 1.000 | 0.370 / 1.000 | 72 |

Every scorer catches the spikes; precision is the tuning knob. Two things lower it: a
trend inside the window (the sine moves about 1.9 units over 30 samples, and no scorer
models a local slope), and the short window, whose robust scale estimate fluctuates.
To raise precision, increase `n_sigmas` or combine the scores as
[`pressure_valve`](robust_filters.md) does.
