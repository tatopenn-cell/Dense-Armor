# Streaming detection (real-time, multi-channel)

Promoted from Dense-Evolution-Discovery after validation on two independent real physical
domains (SO-101 robot arm, real UCI HAR IMU). `StreamingDeviationDetector` is a zero-latency
port of [`classify_segments`](../protect/arbiter.md)' own per-point causal deviation check -- not the
spike-vs-regime label, which looks ahead of a deviant run's end and stays a batch/offline
question by design. `MultiChannelStreamingDeviationDetector` and
`classify_segments_multichannel` remove the need to hand-loop over independent channels
(a robot's joints, an IMU's axes) -- ergonomics, not a new algorithm; each channel keeps
its own independent reference window and baseline.

::: dense_armor.anomaly.streaming

---

**See also**: [Arbiter](../protect/arbiter.md) -- the batch `classify_segments` this module ports the
causal half of.

## River-compatible scorer

`StreamingDeviationScorer` (`dense_armor.anomaly.deviation`, `pip install dense-armor[river]`)
exposes the same causal deviation as a [river](https://riverml.xyz) anomaly detector:
`score_one` returns `|x - median| / scale` over the window learned so far, `learn_one` adds the
value. A score above `n_sigmas` is exactly a `StreamingDeviationDetector.update` flag, so it
plugs into river pipelines and `anomaly.ThresholdFilter`.

```python
from dense_armor.anomaly.deviation import StreamingDeviationScorer

model = StreamingDeviationScorer(radius=5, ref_mult=2)
for v in [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.05]:
    model.learn_one({"v": v})
model.score_one({"v": 50.0})
```

::: dense_armor.anomaly.deviation

## Streaming robust filters

`dense_armor.anomaly.filters` brings the four batch filters of
[`robust_filters`](robust_filters.md) — Hampel, Tukey fences, Chauvenet and iterative sigma
clipping — to one sample at a time. Each scorer looks only at the `2 * radius` samples
before the value being scored (the causal window), scores the new value in the method's own
robust scale, and exposes `is_outlier`. The batch filters use a centred window, which looks at
future samples; a live robot loop cannot, and that is the only difference.
`HampelFilter` also replaces an outlier with the window median.

```python
from dense_armor.anomaly.filters import HampelScorer

s = HampelScorer(radius=15, n_sigmas=3.0)
for v in [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.05]:
    s.learn_one({"v": v})
s.is_outlier({"v": 50.0})
```

```
True
```

Precision and recall on 5,000 seeded samples with 50 spikes of size 5–15 (1 % contamination),
window 30 (radius 15):

| Scorer | stationary noise (σ = 0.5): precision / recall | sine (amplitude 5, period 500) + same noise: precision / recall | µs per sample |
|---|---|---|---|
| Hampel | 0.459 / 1.000 | 0.318 / 1.000 | 90 |
| Tukey | 0.303 / 1.000 | 0.195 / 1.000 | 75 |
| Chauvenet | 0.282 / 0.980 | 0.167 / 1.000 | 28 |
| SigmaClip | 0.588 / 1.000 | 0.370 / 1.000 | 72 |

Every scorer catches the spikes; precision is the tuning knob. Two things lower it: a trend
inside the window (the sine moves about 1.9 units over 30 samples, and no scorer models a local
slope), and the short window, whose robust scale estimate fluctuates from one window to the
next, so a 3-sigma rule fires on some ordinary noise. To raise precision, increase `n_sigmas`
or combine the scores as [`pressure_valve`](robust_filters.md) does. At 100 Hz (10 ms per
cycle) the cost is under 1 % of the loop budget.

::: dense_armor.anomaly.filters

## Multichannel: online robust Mahalanobis distance

`OnlineRobustMahalanobis` (`dense_armor.anomaly.mahalanobis`) scores several
channels together: it tracks the geometric median and the median covariance online
(Guillot, Godichon-Baggioni, Robin & Sansonnet, arXiv:2601.03957) and scores each sample by
its Mahalanobis distance from them, so a fault that shows up as an unusual combination of
channels is caught even when each channel alone looks normal.

```python
import numpy as np
from dense_armor.anomaly.mahalanobis import OnlineRobustMahalanobis

m = OnlineRobustMahalanobis(feature_keys=["a", "b"])
for v in np.random.default_rng(1).normal(0, 1, (300, 2)):
    m.learn_one({"a": float(v[0]), "b": float(v[1])})
print(m.is_outlier({"a": 40.0, "b": -40.0}), m.is_outlier({"a": 0.0, "b": 0.0}))
```

```
True False
```

::: dense_armor.anomaly.mahalanobis
