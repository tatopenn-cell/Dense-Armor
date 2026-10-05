# Streaming detection (real-time, multi-channel)

Promoted from Dense-Evolution-Discovery after validation on two independent real physical
domains (SO-101 robot arm, real UCI HAR IMU). `StreamingDeviationDetector` is a zero-latency
port of [`classify_segments`](arbiter.md)' own per-point causal deviation check -- not the
spike-vs-regime label, which looks ahead of a deviant run's end and stays a batch/offline
question by design. `MultiChannelStreamingDeviationDetector` and
`classify_segments_multichannel` remove the need to hand-loop over independent channels
(a robot's joints, an IMU's axes) -- ergonomics, not a new algorithm; each channel keeps
its own independent reference window and baseline.

::: dense_armor.utility.streaming

---

**See also**: [Arbiter](arbiter.md) -- the batch `classify_segments` this module ports the
causal half of.

## River-compatible scorer

`StreamingDeviationScorer` (`dense_armor.utility.river_anomaly`, `pip install dense-armor[river]`)
exposes the same causal deviation as a [river](https://riverml.xyz) anomaly detector:
`score_one` returns `|x - median| / scale` over the window learned so far, `learn_one` adds the
value. A score above `n_sigmas` is exactly a `StreamingDeviationDetector.update` flag, so it
plugs into river pipelines and `anomaly.ThresholdFilter`.

```python
from dense_armor.utility.river_anomaly import StreamingDeviationScorer

model = StreamingDeviationScorer(radius=5, ref_mult=2)
for v in [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.05]:
    model.learn_one({"v": v})
model.score_one({"v": 50.0})
```

::: dense_armor.utility.river_anomaly
