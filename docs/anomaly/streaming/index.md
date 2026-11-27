# Streaming anomaly detection

## The map of streaming anomaly detection

Click a module to open its page, or the button at the top to go back to [Detection](../../detection/index.md).

<object type="image/svg+xml" data="../../assets/maps/streaming.svg" style="width:100%"></object>

This section explores the ability to detect, at the exact instant a data point arrives, that this value deviates from those that preceded it. It is an indispensable function both for a trading desk that monitors the execution cost of orders, and for a robot that reads the speed of a joint a hundred times per second: in both cases, the system must make an immediate decision based on the single sample, without being able to wait for subsequent data.

All the tools presented here follow the same basic logic: they keep in memory a short time window of the most recent values, calculate their reference parameters (the central value and the dispersion), and compare each new data point with this profile. Some return a binary answer (yes/no), while others assign a score: the higher the value, the more anomalous the data point is compared to the recent past. Since they process one sample at a time without ever looking ahead, these algorithms are ideal for running inside a real-time control loop.

The section is organized by individual classes and functions:

- streamDD: is the starting point and returns a yes/no indication on a single data stream.
- MultistreamDD: extends the same logic by monitoring multiple channels simultaneously.
- streamDS: provides the same evaluation, but in the form of a numerical score.
- HampelScorer, TukeyScorer, ChauvenetScorer, and SigmaClipScorer: implement four classic statistical rules to establish when a data point is too distant from the norm, each with its own measurement method.
- HampelFilter: applies the Hampel rule to correct anomalies, replacing out-of-scale values with the median of the current window.
- class_seg_multi: operates instead on the entire historical recording to distinguish a simple isolated spike from a lasting regime change.

(Note: the online robust Mahalanobis distance, which analyzes multiple channels jointly, is covered on a dedicated page within the anomalies section).

Each page follows a recurring structure: it opens with an illustrative chart, continues with a use case related to financial execution costs, and concludes with the robot example. The first example below shows the basic usage of streamDD.

Daily trading stream of a stock index: series of daily percentage returns centered around 0.05%, with a background noise of 0.8% and a sudden crash at sample 1000. The detector identifies and flags the anomaly on the very same day.

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
