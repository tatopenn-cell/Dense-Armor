# class_seg_multi

[← Streaming anomaly detection](index.md)

`class_seg_multi` reads a table of readings, one column per channel, and applies `classify_segments` independently to each column. It returns three stacked arrays, one per channel: a label for every sample, a deviation score, and an uncertainty. A spike is a short deviation; a regime is a sustained change. A short batch, offline triage, not the zero-latency streaming flag.

**The name.** `class_seg_multi` is short for *classify segments, multi-channel*: it classifies the stretches of a recording, channel by channel, as spike, regime or clean.

![The share of a shifted run labelled regime for runs of 2 to 60 days and shifts of 3 to 15 standard deviations.](../../assets/streaming/class_seg_multi_3d.png)

A short run is a spike, whatever its size. A run becomes a regime only when it lasts: with a shift of 15 standard deviations, runs of 2, 6 and 12 days are spikes, and about half of a run of 24 days is labelled regime (0.53). Later the window learns the new level and the following days are labelled clean, so a run of 60 days has a regime share of 0.21.

## A first example: three venues

Three venues, 400 days of the cost of executing an order. Venue A has one bad tick at day 100; venue C moves up by 9 from day 300.

```python
import numpy as np
from dense_armor.core.online.anomaly.streaming import class_seg_multi

rng = np.random.default_rng(5)
X = 12.0 + 1.5 * rng.standard_normal((400, 3))
X[100, 0] = 40.0
X[300:, 2] += 9.0
labels = class_seg_multi(X, radius=10)[0]
for c in range(3):
    print(c, {k: labels[:, c].tolist().count(k) for k in ("spike", "regime")})
```

```
0 {'spike': 2, 'regime': 0}
1 {'spike': 1, 'regime': 0}
2 {'spike': 10, 'regime': 8}
```

Venue A has 2 days labelled spike, venue B has 1, and venue C, whose level moved, has 10 spike days and 8 regime days. The function labels each channel on its own, as if you had called it once per column.

![The cost of executing an order on three venues; pink days are spikes and orange days are a regime.](../../assets/streaming/class_seg_multi_finance.png)

Venue A has a bad tick, venue B nothing special and venue C a jump of the level.

## The same on a robot

```python
import numpy as np
from dense_armor.core.online.anomaly.streaming import class_seg_multi

rng = np.random.default_rng(0)
X = rng.normal(0.0, 0.1, (300, 2))
X[100, 0] = 5.0
X[200:, 1] += 2.0
labels = class_seg_multi(X, radius=10)[0]
print([{k: labels[:, c].tolist().count(k) for k in ("spike", "regime")} for c in range(2)])
```

```
[{'spike': 4, 'regime': 0}, {'spike': 11, 'regime': 15}]
```

Joint 1 has one bad reading: 4 samples are labelled spike and none regime. Joint 2 moves to a new level at sample 200: 15 samples are labelled regime, and 11 spike.

## How it works

`class_seg_multi` applies `classify_segments` (`dense_armor.core.online.protect.arbiter`) independently to each column. Exactly the same result per channel as calling it individually, only stacked instead of requiring a manual loop. Each channel has its own robust window; channels never influence each other. `X` has shape `(n_samples, n_channels)`. The causal window holds `radius * ref_mult` values. `spike_run_max` sets the maximum run length counted as a spike; longer runs are `regime`. `n_sigmas` is the flagging threshold in robust-sigma units; `eps` is the numerical guard for a degenerate scale. This is batch: it can see the end of a run to decide `spike` versus `regime`; it is not the zero-latency streaming flag. It does not raise in the streaming flow in the same way; it is an analysis function.

## The parameters

`class_seg_multi` is a function. Signature:

`(X: numpy.ndarray, radius: int = 10, ref_mult: int = 3, n_sigmas: float = 3.0, spike_run_max: int = 2, eps: float = 1e-09)`

| Parameter | Default | What it does |
|---|---|---|
| `X` | required | array of shape `(n_samples, n_channels)` |
| `radius` | 10 | with `ref_mult`, the causal window holds `radius * ref_mult` values |
| `ref_mult` | 3 | window length multiplier |
| `n_sigmas` | 3.0 | flag threshold in robust-sigma units |
| `spike_run_max` | 2 | maximum run length counted as a spike |
| `eps` | 1e-9 | numerical guard for a degenerate scale |

**Returns:** `(labels, deviation, uncertainty)`, each `(n_samples, n_channels)`; `labels` per sample and channel (`clean`, `spike`, `regime`).

## API reference

::: dense_armor.core.online.anomaly.streaming.class_seg_multi

## Details

- **Sources.** `dense_armor/core/online/anomaly/streaming.py`; `classify_segments` in `dense_armor/core/online/protect/arbiter.py`. See `classify_segments` for the per-point rule and labels; `MultistreamDD` is the per-call multi-channel version, see [Multi](multi_channel_streaming_deviation_detector.md). Provenance: promoted from Dense-Evolution-Discovery; `class_seg_multi` applies it per column.
