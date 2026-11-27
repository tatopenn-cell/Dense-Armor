# Streaming anomaly detection

<object type="image/svg+xml" data="../../assets/maps/streaming.svg" style="width:100%"></object>

A streaming anomaly detector reads one sample at a time, compares it with the recent past, and answers at once with a score and a yes/no flag, without seeing what comes next.

A stock-index trading day stream: daily returns in percent around 0.05 with noise 0.8, and one crash day at sample 1000. The detector flags the crash on the same day.

```python
import numpy as np
from dense_armor.core.online.anomaly.streaming import streamDD

rng = np.random.default_rng(42)
r = 0.05 + 0.8 * rng.standard_normal(2000)
r[1000] = -9.0
det = streamDD(radius=30, ref_mult=3, n_sigmas=4.0)
flags = np.flatnonzero([det.update(float(x)) for x in r])
print(flags)
```

```
[   4  949 1000]
```

Sample 1000 is the crash; sample 4 is warm-up; sample 949 is one ordinary noise value out of 1,999, one false alarm. Executed with dense_armor 1.7.2. The robot cases are below.

These tools give every new reading a score that says how unusual it is compared with the last few readings, and a yes/no flag when it is too unusual. They answer at once, one reading at a time, without waiting for what comes next.

A robot joint sends its velocity 100 times a second. Most readings are ordinary noise; once in a
while one is absurd, because the arm bumped into something or a cable gave a false reading. The
controller must know *right now*, on that very sample, without waiting for the future. The tools
on this page give every new sample a score that says how unusual it is compared with the recent
past, and a yes/no flag when it is too unusual.

![A joint velocity stream at 100 Hz with the collision spike flagged](../../assets/streaming/spike.png)

## API reference




## Details

## What the short names mean

Four objects of this section have a short name: `streamDD` is *stream Deviation Detector*, `streamDS` is *stream Deviation Scorer*, `MultistreamDD` is *Multi-channel stream Deviation Detector* and `class_seg_multi` is *classify segments, multi-channel*. The other five keep their names: `HampelScorer`, `HampelFilter`, `TukeyScorer`, `ChauvenetScorer` and `SigmaClipScorer`.

## The pages of the section

| Page | Object |
|---|---|
| [streamDD](streaming_deviation_detector.md) `streaming_deviation_detector` | `streamDD` |
| [streamDS](streaming_deviation_scorer.md) `streaming_deviation_scorer` | `streamDS` |
| [MultistreamDD](multi_channel_streaming_deviation_detector.md) `multi_channel_streaming_deviation_detector` | `MultistreamDD` |
| [HampelScorer](hampel_scorer.md) `hampel_scorer` | `HampelScorer` |
| [TukeyScorer](tukey_scorer.md) `tukey_scorer` | `TukeyScorer` |
| [ChauvenetScorer](chauvenet_scorer.md) `chauvenet_scorer` | `ChauvenetScorer` |
| [SigmaClipScorer](sigma_clip_scorer.md) `sigma_clip_scorer` | `SigmaClipScorer` |
| [HampelFilter](hampel_filter.md) `hampel_filter` | `HampelFilter` |
| [class_seg_multi](class_seg_multi.md) `class_seg_multi` | `class_seg_multi` |
| [Mahalanobis](../mahalanobis.md) | `OnlineRobustMahalanobis`, moved to `docs/anomaly/mahalanobis.md` |
