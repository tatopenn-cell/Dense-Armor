# Streaming anomaly detection

## The map of streaming anomaly detection

Click a module to open its page, or the button at the top to go back to [Detection](../../detection/index.md).

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

## The pages of the section

| Page | What it shows |
|---|---|
| [streamDD](streaming_deviation_detector.md) | a yes or no for each new value: it flags a value that deviates from the recent past |
| [streamDS](streaming_deviation_scorer.md) | the same deviation as a number instead of a flag |
| [MultistreamDD](multi_channel_streaming_deviation_detector.md) | one flag for each channel of a vector, all the joints at once |
| [HampelScorer](hampel_scorer.md) | Hampel, one sample at a time: distance from the median of the window |
| [TukeyScorer](tukey_scorer.md) | Tukey fences, one sample at a time: distance beyond the quartile fences |
| [ChauvenetScorer](chauvenet_scorer.md) | Chauvenet, one sample at a time: the expected number of values this far from the mean |
| [SigmaClipScorer](sigma_clip_scorer.md) | sigma clipping, one sample at a time: clip the far values, then score |
| [HampelFilter](hampel_filter.md) | replaces an outlier with the median of the window |
| [class_seg_multi](class_seg_multi.md) | spike versus regime on a whole recording, channel by channel |
| [OnlineRobustMahalanobis](../mahalanobis.md) | the full page of the anomaly section |

Four objects of this section have a short name: `streamDD` is *stream Deviation Detector*, `streamDS` is *stream Deviation Scorer*, `MultistreamDD` is *Multi-channel stream Deviation Detector* and `class_seg_multi` is *classify segments, multi-channel*. The other five keep their names: `HampelScorer`, `HampelFilter`, `TukeyScorer`, `ChauvenetScorer` and `SigmaClipScorer`.

## The same on a robot

A robot joint sends its velocity 100 times a second. Most readings are ordinary noise; once in a
while one is absurd, because the arm bumped into something or a cable gave a false reading. The
controller must know *right now*, on that very sample, without waiting for the future. The tools
on this page give every new sample a score that says how unusual it is compared with the recent
past, and a yes/no flag when it is too unusual.

![A joint velocity stream at 100 Hz with the collision spike flagged](../../assets/streaming/spike.png)
