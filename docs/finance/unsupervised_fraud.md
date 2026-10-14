# Unsupervised fraud: no labels needed

**In plain words:** A new country, a new merchant, a new payment type has no history. This page shows three detectors that flag a strange transaction even if nobody has ever labelled a fraud before.

## The scene

A bank opens a new corridor to a foreign market. On day one, transactions arrive. Zero labels. The supervised detector, trained on the old corridor, knows nothing about this market. The unsupervised detector, which watched the same stream from the start, already knows what 'normal' looks like here.

Now the numbers. Trained with **no labels at all** on 40,000 real credit card transactions, tested on 10,000 with 102 real frauds. Three detectors in parallel.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| LODA | Histogram every transaction in 50 random directions | A transaction lands in an empty bin |
| Half-Space Trees | Cut the space into rectangles | A transaction ends in a rectangle with one sample |
| Online Robust Mahalanobis | Distance from the robust median | A transaction far outside the cloud |

## 1. Rare bins in random projections (LODA)

For each of 50 random directions, LODA keeps a histogram of recent transactions. A transaction falls in a bin. If the bin is almost empty, the transaction is strange.

## 2. Rare leaves (Half-Space Trees)

Half-Space Trees cuts the space into rectangles. A normal transaction lands in a busy rectangle. A strange one lands in a rectangle with one or two samples.

## 3. Distance from the median (Mahalanobis)

The geometric median is the 'typical' transaction, robust to outliers. The Mahalanobis distance says how far the new transaction is, in units of the spread of the cloud.

## The chain

For each new transaction:

1. All three detectors score the same transaction.
2. Each raw score becomes its percentile in the last 500 transactions: 'this score is above 99% of what I have seen'.
3. The ensemble score is the mean of the three percentiles. Above 0.99, the transaction is flagged.

## What you get on real fraud data

Same split as the supervised `FraudPipeline`: 40,000 train, 10,000 test, 102 real frauds.

| Metric | Value |
|---|---|
| ROC-AUC, LODA | 0.966 |
| ROC-AUC, Half-Space Trees | 0.935 |
| ROC-AUC, Mahalanobis | 0.957 |
| ROC-AUC, ensemble | **0.959** |
| Training time (one core, no labels) | 32 s |

For comparison, the supervised `FraudPipeline` (Mondrian forest + SGD tree, needs every label) reaches **0.969** on the same split. **One point of AUC difference, without a single label.**

## When you need it

- A new market, corridor or product with no labelled history.
- A stream where the fraud label arrives weeks later, if at all.
- A second layer of defence next to a supervised system, to catch the frauds the supervised one was never trained on.

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

*The numbers on this page come from a single run on real historical data. Not a forecast, not financial advice.*
