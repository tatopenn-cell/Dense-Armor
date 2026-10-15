# Unsupervised fraud: multi-model anomalous transaction detection

When you launch a new product or open a new market, you have no historical data of past frauds. This page shows how three detectors can score suspicious transactions without any pre-classified labels.



## The scene

When a financial institution opens a new transaction corridor or launches a product in a foreign market, historical training data is unavailable. A traditional supervised fraud model, calibrated on legacy markets, cannot adapt to the baseline dynamics of this new environment. In contrast, an unsupervised detection architecture learns what is normal from the stream itself, without historical labels.

To quantify this advantage, the pipeline was benchmarked using **zero classification labels** during the training phase on 40,000 historical transactions. It was then evaluated against a testing stream of 10,000 transactions containing 102 verified fraud instances, deploying three parallel anomaly detection layers.


## Core Capabilities

| Tool | Strategic Function | Operational Impact |
|---|---|---|
| **LODA** | Multi-dimensional pattern profiling | Flags a transaction with an unprecedented combination of import, location, and frequency. |
| **Half-Space Trees** | Real-time density monitoring | Isolates highly unique, one-off transaction profiles that deviate from established operational baselines. |
| **Online Robust Mahalanobis** | Behavioral distance mapping | Calculates overall operational divergence, ignoring normal user variations while trapping systemic anomalies. |


![Quality score of the label-free detectors against the supervised one on the same 10,000 transactions](../assets/finance/unsupervised_fraud.png)


*Higher metric values denote superior ranking accuracy in isolating fraud instances. The dashed baseline represents a random classification reference (0.5 AUC).*


## 1. Multi-dimensional outlier profiling (LODA)
LODA evaluates incoming transactions across multiple sparse sub-spaces simultaneously. By maintaining real-time density scores for historical attributes, it flags transactions that fall into low-frequency operational brackets, isolating anomalous behavior without requiring pre-existing fraud labels.

## 2. Dynamic stream density isolation (Half-Space Trees)
Half-Space Trees continuously segments the transaction space into hierarchical mass profiles. Legitimate transaction streams naturally populate high-density zones, while fraudulent patterns terminate in isolated, low-mass sectors, enabling detection of anomalous, one-off events.

## 3. Robust operational distance mapping (Mahalanobis)
This module tracks the stable geometric center of legitimate user behavior, remaining resilient to standard transactional noise. The distance framework gauges exactly how far an incoming transaction deviates from normal operational boundaries, detecting systemic anomalies while safeguarding legitimate customer transactions.


## Execution Workflow

For each incoming transaction, the pipeline executes a three-stage validation protocol:

1. **Parallel Multi-Model Scoring:** The raw transaction attributes are processed simultaneously by all three detection layers, each generating an independent anomaly index.
2. **Dynamic Percentile Normalization:** Raw indices are converted into relative percentiles against a moving window of the last 500 operations. This step homogenizes the outputs and ensures scale invariance across different algorithms.
3. **Ensemble Consensus Voting:** The final risk score is computed as the mean of the three percentiles. If this unified consensus exceeds the 0.99 risk threshold, an alert is raised.


## Empirical Performance

The pipeline was benchmarked using the identical dataset split as the fully supervised architecture: a training pool of 40,000 transactions and a testing stream of 10,000 transactions containing 102 verified fraud instances.

| Metric | Value |
|---|---|
| ROC-AUC, LODA | 0.966 |
| ROC-AUC, Half-Space Trees | 0.935 |
| ROC-AUC, Mahalanobis | 0.957 |
| ROC-AUC, ensemble | **0.959** |
| Alerts at the 0.99 cutoff (out of 10,000) | 83 |
| Real frauds among the alerts | 33 (precision 39.8%) |
| Recall on frauds | 32.4% |
| Training time (Kaggle CPU, no labels) | 28 s |

### Strategic Business Takeaway

For comparative context, a fully supervised detection engine—which requires manual classification labels for every single historical transaction—achieves a ROC-AUC of **0.969** on this exact data split. LODA alone scores 0.966, above the ensemble's 0.959.

On ranking quality the unsupervised ensemble reaches **99% of the supervised ROC-AUC** (0.959 against 0.969). At the 0.99 cutoff it catches 33 of the 102 frauds (32.4%), against 69 for the supervised pipeline, and 39.8% of its alerts are real, against 95.8%. What it saves is the labelling work.


## Target Enterprise Use Cases

This unsupervised pipeline is purpose-built for three critical financial infrastructures:

- **New Market & Product Deployment:** For institutions launching new payment products, currency corridors, or geographical expansions where zero historical fraud patterns are available for traditional model training.
- **Asymmetric Feedback Protection:** Designed for streaming environments where official fraud reports suffer from multi-week settlement latencies, providing a score before the fraud reports arrive.
- **Hybrid Defensive Layering:** Runs in parallel with a supervised rule engine and scores the same transactions without labels.


## Details

### Implementation

The pipeline is in `pipelines/finance/unsupervised_fraud.py`.
Primitives chained, each with its own page:

- [LODA](../anomaly/loda.md)
- [Half-Space Trees](../anomaly/hst.md)
- [Online Robust Mahalanobis](../anomaly/mahalanobis.md)

### References

- Pevny, T. (2016). *LODA: Lightweight on-line detector of anomalies.* Machine Learning 102(2), 275-304.
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). *Fast anomaly detection for streaming data.* IJCAI.
- Guillot, A., Godichon-Baggioni, A., Robin, S., Sansonnet, L. (2025). *Online robust covariance and outlier detection.* arXiv:2601.03957.
- Dataset: OpenML creditcard, version 1. Original source: Machine Learning Group, Universite Libre de Bruxelles (ULB).

*The numbers on this page come from a single run on historical data. Not a forecast, not financial advice.*
