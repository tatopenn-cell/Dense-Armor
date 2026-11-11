# API Reference

A starting point for finding the right tool. Say what you have in front of you (an order to check, a stream of returns, a table, a network to train) and this page tells you which part of the library to use. The examples are about finance first, then neural networks.

Start from what you have in front of you:

- You have **an order to approve before it leaves** → the pre-trade checker and the decision log.
- You have **a stream of returns or prices** → the drift and regime detectors.
- You have **transactions to screen for fraud, or a stream with strange points** → the anomaly detectors.
- You have **a portfolio, a risk number or an execution schedule to compute on a stream** → the finance pipelines.
- You have **a whole table** → machine learning (`fit` / `predict`).
- You have **a model that must learn one sample at a time** → online learning.
- You have **a network to train, on a dataset or on a stream** → deep learning.
- You need **to pick one option at a time and learn from the reward** → the bandit policies.
- You need **to ask for a label only on the samples that matter** → the active-learning strategies.

## Finance

The first thing to read. Every page opens with the question it answers, and every example runs without external data.

- **[Pre-trade limits](../finance/pretrade.md)** — the five controls of SEC Rule 15c3-5(c)(1): size, price band, duplicate, rate, notional. `PreTradeChecker.check(...)` returns a `Decision`.
- **[Decision log](../finance/decision_log.md)** — a JSON Lines file with a SHA-256 hash chain; `DecisionLog.verify()` is `False` as soon as one record is edited. The same chain in PostgreSQL is the [audit table](../finance/audit_db.md).
- **[Feed, ITCH and synthetic feed](../finance/feed.md)** — a Nasdaq ITCH decoder and a feed that is byte-identical on every machine, from a seed.
- **[Execution schedules](../finance/schedule.md)** and the **[FIX gateway](../finance/fix_gateway.md)** — TWAP and VWAP, and orders to a test acceptor.
- **[Value-at-Risk](../finance/risk.md)**, **[portfolio](../finance/portfolio.md)** and **[order book](../finance/order_book.md)** — streaming tail risk, minimum-variance weights, order flow imbalance.
- **[Regime](../finance/regime.md)** and **[drift versus noise](../finance/drift_vs_noise.md)** — a change in returns, told apart from noise.
- **[Market surveillance](../finance/surveillance.md)**, **[fraud](../finance/fraud.md)**, **[intrusion](../finance/intrusion.md)** and **[credit scoring](../finance/credit_scoring.md)** — pipelines that chain the blocks below on a stream.
- **[Dashboard](../finance/dashboard.md)**, **[status and emergency stop](../finance/emergency.md)** — the web application that shows the log and stops the system.

The two calls that make the first example, from the quickstart of the README:

```python
from dense_armor.utility.finance import PreTradeChecker, DecisionLog

checker = PreTradeChecker(
    max_notional=50_000.0, max_size=0.2, price_band=0.001,
    max_orders=3, window_s=10.0, duplicate_window_s=1.0,
)
log = DecisionLog("decisions.jsonl")
d = checker.check("MSFT", "buy", 5.0, 100.0, 100.0, t=0.0)
log.append(t=0.0, payload={"account": "MSFT", "approved": d.approved, "reason": d.reason})
print(log.verify())
```

```
True
```

A change in the mean of daily returns (in percent, synthetic), caught by ADWIN while the stream runs:

```python
import numpy as np
from dense_armor.utility.online.drift.adwin import ADWIN

rng = np.random.default_rng(0)
r = np.r_[rng.normal(0.04, 0.6, 400), rng.normal(2.0, 0.6, 400)]
d = ADWIN()
for t, x in enumerate(r):
    d.update(x)
    if d.drift_detected:
        print(t)
        break
```

```
441
```

## Deep learning

The object the library is built towards: networks made of layers, with a gradient that can be checked, trained one batch or one sample at a time.

- **[Foundations](../neural/index.md)**, **[layers](../neural/layers.md)** and **[chains](../neural/chains.md)** — the smallest pieces and how to join them.
- **[Convolution and pooling](../neural/vision_layers.md)**, **[recurrent layers](../neural/sequence_layers.md)** and **[attention](../neural/attention.md)**.
- **[Training](../neural/training.md)** and **[continuous training](../neural/continuous.md)** — schedules, mini-batches, early stopping, a background learner.
- **[Continual learning](../neural/continual_learning.md)** — EWC, GEM, A-GEM, and how to measure what the network remembers.
- **[Encoders and decoders](../neural/encoder_decoder.md)**, **[mixture of experts](../neural/moe_testtime.md)** and **[composable attention](../neural/composable_attention.md)**.
- **[Decoder language models](../neural/decoder_lm.md)**, **[language-model training](../neural/lm_training.md)** and the **[rotary decoder](../neural/rotary_decoder.md)** — generation against a static key/value cache.

## Machine learning

The classes of `dense_armor.ml` read a whole table with `fit(X, y)`. The [overview](../ml/index.md) has one page per family: statistics, preprocessing, linear models, trees, forests, ensembles, anomaly and change detection, vision.

## Online learning

Models that learn one sample at a time, with the same `learn_one` / `predict_one` contract everywhere.

- **[Optimizers](../optim/index.md)**, **[linear models](../linear/glm.md)**, **[trees](../tree/index.md)**, **[clustering](../cluster/index.md)**.
- **[Calibration](../learn/calibration.md)** — online Platt scaling: honest probabilities from any classifier.
- **[Metric learning](../learn/metric_learning.md)** — OASIS, LEGO and POLA, and a k-NN classifier that learns its distance online.
- **[Fraud](../finance/fraud.md)**, **[credit scoring](../finance/credit_scoring.md)** and **[static against online](../finance/static_vs_online.md)** — online trees and ensembles on a stream of transactions.
- **[Ensembles](../ensemble/index.md)** — a group of models that answer together, from a plain vote to a second model on top of the members' predictions.
- **Factorization machines and recommenders** — **[Factorization machines](../facto/index.md)** (`FMRegressor`, `FMClassifier`, `FFMClassifier`, `FwFMClassifier`, `HOFMClassifier` and the regressors), **[recommenders](../reco/index.md)** (`Baseline`, `BiasedMF`, `FunkMF`; `learn_one(user, item, rating)`), `SDFT` and `Skyline`.

## Anomaly and drift detection

- **[Streaming detectors](../anomaly/streaming.md)**, **[robust filters](../anomaly/robust_filters.md)** (Chauvenet, Tukey, Hampel, sigma clipping) and **[curvature](../anomaly/curvature.md)** — one score per sample.
- **[Drift detectors](../drift/detectors.md)** — ADWIN, KSWIN, Page–Hinkley, and the CUSUM family; **[CUSUM and ARL theory](../drift/cusum.md)** for the expected detection and false-alarm latency.
- **[Arbiter](../protect/arbiter.md)** — classifies each point as clean / spike / regime against a wide causal reference window, then routes it.

Five multivariate detectors, for when a change touches several channels at once. Each has a different bias, and all five go through `AnomalyGate` and `Protected` unchanged.

| Detector | Bias | Cost |
|---|---|---|
| `HalfSpaceTrees` (`anomaly.hst`) | sparse regions of the space | O(1) |
| `OnlineIsolationForest` (`anomaly.oiforest`) | depth to isolate a point | O(window) |
| `LODA` (`anomaly.loda`) | rare bins in random 1D projections | O(1) |
| `OneClassSGD` (`anomaly.ocsvm`) | distance from a learned kernel boundary (SONAR) | O(1) |
| `OnlineRobustMahalanobis` (`anomaly.mahalanobis`) | distance from a robust median under the MCM | O(1) |

## The library by section

The same pages, grouped the way the package is organised (`dense_armor.<section>`):

| Section | What it is for | Pages |
|---|---|---|
| [Finance](../finance/index.md) | pre-trade risk, decision log, drift and fraud on streams | the pages above |
| [Deep learning](../neural/index.md) | layers, attention, training, continual learning, language models | Layers, Chains, Attention, Training, Decoder |
| [Machine learning](../ml/index.md) | `fit` / `predict` on a whole table | one page per family |
| [Online learning](../learn/index.md) | models that learn one sample at a time | Calibration, Metric learning, fraud and credit trees |
| [Ensembles](../ensemble/index.md) | a group of models that answer together | Voting, Bagging, Boosting, Stacking |
| [Anomaly detection](../anomaly/index.md) | a score per sample: high means it does not look like the recent past | Streaming, Robust filters, Curvature, multivariate detectors |
| [Drift detection](../drift/index.md) | one alarm when a slow, sustained change is real | Detectors, CUSUM and ARL theory |
| [Toolkit](../toolkit/index.md) | generic tools | Toolkit |

<small>

**Also in the library**, as attributes of the same building blocks and not as the main subject:

- The **shields** [`Armor`](../shields/armor.md) (one 1D series) and [`Orca`](../shields/orca.md) (a model, input and output), built on the [hybrid engine](../shields/hybrid_engine.md) and the [adaptive engine](../shields/engine.md); [Protection](../protect/index.md) groups them with Arbiter.
- **Online learning on a robot**: [online classifiers](../learn/online_classifiers.md) (its state from the joint signals) and [online dynamics](../learn/online_dynamics.md) (the torque a rigid-body model misses).
- **Robot control and dynamics**: [trajectory](../control/trajectory.md), [kinematic controller](../control/kinematic_controller.md), [rate limiter](../control/rate_limiter.md), [CBF filter](../control/cbf_filter.md), [rigid-body dynamics](../dynamics/urdf_dynamics.md), [passivity + singularity-CBF controller](../dynamics/passivity_cbf_controller.md), [6-DoF controller](../dynamics/six_dof_pbc_cbf_controller.md), [xacro](../dynamics/xacro_support.md), [mimic joints](../dynamics/mimic_joints.md); the [real robot benchmark](../benchmarks/index.md) (CASPER, UR3e).

</small>
