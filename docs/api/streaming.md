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
