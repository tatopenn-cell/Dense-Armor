# Online learning: sequential models and incremental architectures for real-time adaptation

A bank approves or blocks a card payment in under a second. The model that decides was
trained once, on last year's payments, and fraudsters change their behaviour. Four
things need to keep learning while the payments flow, because the world will not stay
the same:

- **"How sure am I that this payment is fraud?"** — a classifier that says 0.8 should be
  right 8 times out of 10. A score whose numbers are not honest cannot be compared with
  a cost: the desk that blocks at 0.8 blocks the wrong payments.
  → [Calibration](calibration.md)

- **"Which past days, orders or payments look like this one?"** — the nearest-neighbour
  step underneath every retrieval and anomaly-detection module needs a way to say "these
  two cases are similar". Euclidean distance is a default, not a decision: on real
  market features it is often wrong.
  → [Metric learning](metric_learning.md)

- **"Is the rule that was right last year still right?"** — a fixed threshold goes stale
  when the stream changes. The online trees of the finance pages learn from every
  transaction, one at a time, and notice the change.
  → [Fraud detection](../finance/fraud.md), [credit scoring](../finance/credit_scoring.md),
  [static against online](../finance/static_vs_online.md)

- **"Did the regime of the market change?"** — a drift detector watches the returns and
  raises one alarm when the change is real, not noise.
  → [Regime](../finance/regime.md), [drift detectors](../drift/detectors.md)

The four share one property: they update one sample at a time, while the stream runs,
without a separate training phase. That is what "online" means in this section.

## Pages

- **[Calibration](calibration.md)** — online Platt scaling: corrects the probabilities
  of any classifier one sample at a time.
- **[Metric learning](metric_learning.md)** — OASIS, LEGO and POLA, and a k-NN
  classifier that learns its distance online.
- **[Fraud detection](../finance/fraud.md)** and **[credit scoring](../finance/credit_scoring.md)** —
  online trees and ensembles on a stream of transactions and customers.

<small>

Two pages use a robot as the example, not a market: [online classifiers](online_classifiers.md)
(the robot's own state, recognised from its joint signals) and
[online dynamics](online_dynamics.md) (the torque a rigid-body model misses).

</small>
