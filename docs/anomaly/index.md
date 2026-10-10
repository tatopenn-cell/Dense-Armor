# Anomaly detection

An anomaly detector gives every sample a score: high means the sample does not look like the
recent past. These detectors work one sample at a time, so they run inside a control loop.

- **[Streaming](streaming.md)** — zero-latency deviation scores, single and multi-channel,
  plus causal Hampel, Tukey, Chauvenet, sigma clipping and online robust Mahalanobis.
- **[Robust filters](robust_filters.md)** — the four classic detectors on whole series, and
  the pressure-valve orchestrator.
- **[Curvature](curvature.md)** — a bounded proximity-to-reference score in [0, 1).

## Multivariate detectors

When several channels move together — the residual of a joint on a robot, the reading of a
sensor bank — a per-channel detector is blind to faults that show only in the combination.
These detectors look at the joint vector, one sample at a time.

| Detector | Bias | Cost | Arm ROC-AUC | Drift ROC-AUC |
|---|---|---|---|---|
| [OnlineRobustMahalanobis](mahalanobis.md) | distance from a robust median under the MCM | 0.045 ms | 1.000 | 0.994 |
| [LODA](loda.md) | rare bins in random 1D projections | 0.16 ms | 1.000 | 0.983 |
| [Half-Space Trees](hst.md) | sparse regions of the space | 0.42 ms | 0.966 | 0.934 |
| [OnlineIsolationForest](oiforest.md) | depth to isolate a point | 0.21 ms | 0.360 | 0.734 |
| [OneClassSGD](ocsvm.md) | distance from a learned kernel boundary | 0.04 ms | 0.394 | 0.523 |

The two benchmark streams are `SyntheticArm` (payload fault, 400 samples, 100 healthy) and
`DriftStream` (one sudden drift at half, 2000 samples). The full table, with false alarms
and per-sample latency for each detector, is in
[`benchmarks/anomaly_streaming.py`](https://github.com/tatopenn-cell/Dense-Armor/blob/master/benchmarks/anomaly_streaming.py).

Each detector has its own bias and its own cost. On a stream whose fault is a shift of the
mean (the arm benchmark), `OnlineRobustMahalanobis` is the strongest. On a stream whose
fault is a change of the distribution's shape, `Half-Space Trees` and `LODA` see things the
mean-shift detector misses. On a stream whose fault proportion is well above the paper's
"anomalies are few" assumption, `OnlineIsolationForest` and `OneClassSGD` need a longer
healthy run to converge — their own pages document the assumption.

All five detectors go through `AnomalyGate` and `Protected` unchanged: each exposes a
`threshold`, and any of them can drive the protection of an estimator that learns online.
