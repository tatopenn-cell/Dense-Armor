# Credit scoring: will this borrower repay?

**In plain words:** A bank calls 45,000 customers a year. Two online trees learn who is likely to sign and who is not, one call at a time.

## The scene

A telemarketing desk has 1,000 calls to make tomorrow. Calling everyone is not possible. Calling nobody loses customers. Who should be called first?

The credit scoring pipeline watches the historical stream of calls and answers the question in real time. It learns from every completed call, one at a time. When a customer walks in, the model gives a score between 0 and 1.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| MondrianForest | Random rectangles over the customer features | Long calls and good past campaigns land in the same region |
| SGTClassifier | Newton-step splits | A second opinion on the same customer |

## 1. Random rectangles over the features (MondrianForest)

Imagine every past customer as a dot in space: age, balance, call duration, previous campaign outcomes. The Mondrian forest partitions this space into rectangles, randomly. New customers fall into a rectangle. If the rectangle is full of people who signed, the score is high.

## 2. A second opinion (SGTClassifier)

The SGD tree cuts the same space, but each split is chosen with the Newton step of the loss, and a t-test decides whether the split is worth making. Slower to learn, but the rectangles are chosen, not random.

## The chain

For each new customer:

1. The customer features go to **MondrianForest**. A score comes out.
2. The same features go to **SGTClassifier**. A second score comes out.
3. The two are averaged. Above 0.25, the customer is called.

## What you get on real campaign data

UCI Bank Marketing: 45,211 real telemarketing calls, 11.6% subscriptions. Split 80/20: 36,168 train, 9,043 test, 1,048 signed.

| Metric | Value |
|---|---|
| ROC-AUC, MondrianForest | 0.859 |
| ROC-AUC, SGT | 0.833 |
| ROC-AUC, ensemble | **0.875** |
| Default threshold | 0.25 |
| Calls made (out of 9,043) | 1,991 |
| True positives | 779 |
| False positives | 1,212 |
| Missed (signed but not called) | 269 |
| Recall | 74.3% |
| Precision | 39.1% |
| F1 | 51.3% |
| Training time (one core) | 259 s |

**Why threshold 0.25 and not 0.5**: with the classic threshold the model catches only 19% of the customers who would sign. Lowering the threshold to 0.25 pushes recall to 74% while precision stays at 39%. One call out of three is a signature. This is what a marketing desk wants.

## When you need it

- A bank running a telemarketing campaign.
- A lender that wants a real-time score while the customer is on the phone.
- A system that must adapt to a new product without a full retrain.

## Details

### Implementation

The pipeline is in `pipelines/finance/credit_scoring.py`.
Primitives chained, each with its own page:

- [Mondrian Forest](../tree/index.md)
- [Stochastic Gradient Trees](../tree/index.md)

### References

- Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). *Mondrian Forests: Efficient Online Random Forests.* NeurIPS.
- Gouk, H., Pfahringer, B., Frank, E. (2019). *Stochastic Gradient Trees.* ACML.
- Dataset: UCI Bank Marketing, bank-full.csv. 45,211 rows, 16 features, target y. Original source: S. Moro, P. Cortez, P. Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*.

*The numbers on this page come from a single run on real historical data. Not a forecast, not financial advice.*
