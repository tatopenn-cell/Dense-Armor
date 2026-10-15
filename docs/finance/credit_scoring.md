# Credit scoring: ensemble online tree architectures for real-time conversion optimization

This section evaluates the statistical convergence and predictive efficacy of a streaming ensemble pipeline combining a Mondrian Forest with a Stochastic Gradient Tree (SGT) classifier, optimized for online customer lifetime value and conversion scoring.

**In plain words:** A bank or a sales desk has thousands of potential customers to contact but limited time to make the calls. This tool learns one completed interaction at a time to predict exactly who is likely to subscribe to a product, combining random structural partitioning with adaptive loss-driven trees to generate an analytical ranking that prioritizes high-probability clients in real time.


## The scene

When an institutional banking desk or marketing division executes large-scale customer engagement campaigns, outbound resource allocation faces severe operational bottlenecks. Processing a high-volume client directory indiscriminately generates massive overhead, inflates customer fatigue, and yields low conversion rates, while missing critical lead windows.

To maximize customer lifetime value and optimize outbound conversion funnels, engagement strategies must utilize real-time analytical ranking engines. 

To benchmark this capability on historical data, the dual-tree streaming pipeline was run on **45,211 telemarketing interactions** from the industry-standard UCI Bank Marketing dataset. Subscriptions account for 11.6% of the continuous historical stream. The model operates within a strict predictive framework, learning incrementally from each finalized customer interaction to dynamically score and rank subsequent leads as soon as they interface with the system.


## Core Capabilities

| Tool | Strategic Function | Operational Impact |
|---|---|---|
| **MondrianForest** | Online structural partitioning | Segments the multi-dimensional customer attribute space using stochastic geometric partitions, providing immediate probability scores for incoming profiles. |
| **SGTClassifier** | Loss-driven adaptive tracking | Utilizes regularized second-order Taylor expansions to optimize decision splits, delivering an independent, high-precision evaluation layer. |


![Telemarketing on 9,043 test calls: 779 of 1,048 customers who would sign are reached, 779 of 1,991 calls end in a signature](../assets/finance/credit_scoring.png)

*Higher metric density denotes superior precision in conversion ranking. The dashed baseline represents a random classification reference (0.5 AUC).*


## 1. Stochastic spatial feature partitioning (MondrianForest)
Traditional ensemble random forests demand complete batch recalculations of structural cuts when data properties drift over time. The integrated Mondrian forest architecture bypasses this limitation by treating spatial partitions as independent stochastic processes. Client attribute vectors—including age, financial balances, historical campaign outcomes, and engagement metrics—are mapped into multidimensional blocks. This allows the model to compute localized density estimates continuously and adjust conversion probabilities without global structural degradation.


## 2. Loss-driven mathematical split optimization (SGTClassifier)
Operating as the analytical reference layer of the pipeline, the Stochastic Gradient Tree classifier guides structural splits through localized optimization dynamics. Instead of relying on random feature selection, the tree chooses division boundaries by evaluating the Newton step of the empirical loss function. A localized t-test verifies the statistical significance of each proposed partition before execution. This process ensures stable, noise-resilient structural updates on continuous transaction streams.


## Execution Workflow

For every newly processed customer profile or live communication event, the streaming conversion pipeline executes a three-stage validation protocol:

1. **Parallel Feature Ingestion:** The client attribute vector is fed simultaneously to both streaming tree layers.
2. **Dual-Model Scoring Consensus:** **MondrianForest** evaluates spatial density to assess profile affinity, while **SGTClassifier** outputs an independent score driven by loss optimization.
3. **Calibrated Threshold Enforcement:** The final conversion probability is computed as the arithmetic mean of both metrics. Profiles exceeding the customized 0.25 operational threshold are instantly routed to active engagement queues.


## Empirical Performance & ROI

The processing pipeline was verified using a chronological data split consisting of a 36,168-sample training stream and a 9,043-sample active test stream containing 1,048 verified conversions.

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
| Training time (Kaggle CPU) | 166 s |

### Strategic Decision Threshold Optimization

Traditional model deployments rely mechanically on a generic 0.5 classification cutoff. On this dataset, a 0.5 threshold restricts the system to capturing only 19% of converting customers (198 out of 1,048), introducing severe opportunity costs. 

Calibrating the operational threshold down to **0.25 pushes the Recall rate to 74.3%** with a Precision of 39.1%. This strategic calibration ensures that approximately **two out of every five automated recommendations yield a verified product signature**, which reaches 779 of the 1,048 customers who would sign with 1,991 calls.


## Target Enterprise Use Cases

This streaming predictive architecture is optimized for retail banking and enterprise sales desks:

- **Live Outbound Campaign Optimization:** Prioritizes real-time telemarketing and digital engagement streams, maximizing conversion rates for active sales desks.
- **Instant Inline Credit Verification:** Generates instantaneous conversion and risk profiles while a customer interfaces with online portals or call-center representatives.
- **Dynamic Product Adherence Tracking:** Automatically calibrates classification structures to new financial products without requiring scheduled batch retrainings or model downtime.


## Details

### Implementation

The pipeline is in `pipelines/finance/credit_scoring.py`.
Primitives chained, each with its own page:

- [Mondrian Forest](../tree/index.md#5-a-prediction-with-its-uncertainty-mondrian-forests)
- [Stochastic Gradient Trees](../tree/index.md#4-predicting-a-number-the-torque-of-a-two-link-arm)

### References

- Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). *Mondrian Forests: Efficient Online Random Forests.* NeurIPS.
- Gouk, H., Pfahringer, B., Frank, E. (2019). *Stochastic Gradient Trees.* ACML.
- Dataset: UCI Bank Marketing, bank-full.csv. 45,211 rows, 16 features, target y. Original source: S. Moro, P. Cortez, P. Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
