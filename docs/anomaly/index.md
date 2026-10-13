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
| [OnlineRobustMahalanobis](mahalanobis.md) | distance from a robust median under the MCM | 0.090 ms | 1.000 | 0.994 |
| [LODA](loda.md) | rare bins in random 1D projections | 0.62 ms | 1.000 | 0.992 |
| [Half-Space Trees](hst.md) | sparse regions of the space | 0.90 ms | 0.920 | 0.944 |
| [OnlineIsolationForest](oiforest.md) | depth to isolate a point | 0.27 ms | 0.417 | 0.667 |
| [OneClassSGD](ocsvm.md) | distance from a learned kernel boundary | 0.09 ms | 0.426 | 0.978 |

The two benchmark streams are `SyntheticArm` (payload fault, 400 samples, 100 healthy) and
`DriftStream` (one sudden drift at half, 2000 samples). The table reports the
**learn-on-healthy** protocol: each detector learns only from the healthy part of the
stream and is scored on the whole stream. The other two protocols the benchmark computes
(learn-on-all, learn-on-all-online-ranges) and the full table with false alarms are in
[`benchmarks/anomaly_streaming.py`](https://github.com/tatopenn-cell/Dense-Armor/blob/master/benchmarks/anomaly_streaming.py).

Each detector has its own bias and its own cost. On a stream whose fault is a shift of the
mean (the arm benchmark), `OnlineRobustMahalanobis` is the strongest. On a stream whose
fault is a change of the distribution's shape, `Half-Space Trees` and `LODA` see things the
mean-shift detector misses. On a stream whose fault proportion is well above the paper's
"anomalies are few" assumption, `OnlineIsolationForest` needs a longer healthy run to
converge — its own page documents the assumption. `OneClassSGD` reaches 0.978 on
`DriftStream` and 0.426 on the arm; the arm number is low because only 50 healthy
steps follow `n_init`, not because the schedule is slow. The default `step=1.0` gives
the `eta_t = 1 / t` of Suk and Kpotufe (2025, Lemma 5); the earlier `step=0.1` was
10x slower and reached only 0.523 on `DriftStream`.

All five detectors go through `AnomalyGate` and `Protected` unchanged: each exposes a
`threshold`, and any of them can drive the protection of an estimator that learns online.
