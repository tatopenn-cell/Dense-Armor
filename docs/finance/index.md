# Financial pipelines: multi-model streaming architectures for risk, execution, and fraud detection

This section establishes the comparative framework and analytical infrastructure of nine pipelines, evaluating automated market surveillance, pre-trade risk validation, real-time transaction classification, and incremental network security protocols run on historical financial streams.

**In plain words:** This page serves as the index for nine software pipelines that handle the daily data challenges faced by market analysts, banking infrastructure, and payment desks. Every pipeline chains together the core mathematical blocks of the package to solve practical business questions—like identifying market anomalies, filtering out fraudulent transactions, or sizing trade orders safely before execution.


## Operational Pipelines Matrix

| Pipeline Architecture | Core Analytical Objective | Reference Page |
|---|---|---|
| **Market Surveillance** | Multi-stage statistical processing and online anomaly consensus orchestration. | [Surveillance](surveillance.md) |
| **Value-at-Risk Estimation** | Sequential quantile sketch architectures for streaming tail-risk monitoring. | [Risk](risk.md) |
| **Algorithmic Execution** | High-frequency pre-trade risk validation and automated order damping. | [Execution](execution.md) |
| **Single Ticker Monitoring** | Dual-model online anomaly isolation for individual equity validation. | [Single_ticker.md](single_ticker.md) |
| **Static vs Online Analysis** | Statistical divergence benchmarking between fixed-cutoff and adaptive frameworks. | [Static vs online](static_vs_online.md) |
| **Supervised Fraud Detection** | Real-time transaction classification via sequential ensemble tree models. | [Fraud detection](fraud.md) |
| **Unsupervised Fraud Detection** | Multi-model anomalous transaction profiling deployed on unlabelled streams. | [Unsupervised fraud](unsupervised_fraud.md) |
| **Network Intrusion Detection** | Drift-adaptive streaming tree architectures for real-time security perimeters. | [Intrusion detection](intrusion.md) |
| **Predictive Credit Scoring** | Ensemble online tree models for real-time customer conversion optimization. | [Credit scoring](credit_scoring.md) |


## Architectural Integration & Layout Standard

Every operational pipeline is engineered by chaining algorithmic primitives directly from `dense_armor.utility` in a deterministic, documented execution workflow. The components introduce no additional external code dependencies, ensuring constant-memory processing and strict execution boundaries. To guarantee consistency across the enterprise documentation suite, each module enforces a standardized structural template:

- **Executive Value Proposition:** Detailed deployment scenarios ("The scene"), structured core capabilities, functional execution workflows, and empirical ROI metrics evaluated against industrial benchmarks.
- **Technical Reference Specifications:** Pipeline internals, production code implementation blocks, hyperparameter tables, and peer-reviewed algorithmic citations.


## Data Benchmark & Verification Infrastructure

To ensure rigorous validation, all empirical performance metrics and mathematical convergence trajectories are calculated using public historical datasets. Two baseline fixture suites are distributed natively within the test architecture, while three high-capacity industrial benchmarks are ingested dynamically during continuous integration execution passes:

| Target Domain | Dataset Temporal & Volumetric Coverage | System Deployment Path |
|---|---|---|
| **Equity Survoillance & Risk** | S&P 500, AAPL, MSFT, TSLA: 1,254 trading days (14 Oct 2021 to 13 Oct 2026) | `test/fixtures/finance/*.csv` |
| **High-Frequency Execution** | Coinbase BTC/USD execution stream: 1,000 real-time trades (14 Oct 2026) | `test/fixtures/finance/coinbase_btcusd_trades.csv` |
| **Supervised / Unsupervised Fraud** | OpenML `creditcard` benchmark: 284,807 transactions containing 492 verified frauds | Dynamic network ingestion |
| **Network Intrusion Security** | NSL-KDD benchmark stream: 125,973 labelled network packet flows | Dynamic network ingestion |
| **Conversion & Credit Scoring** | UCI Bank Marketing benchmark: 45,211 multi-feature telemarketing interactions | Dynamic network ingestion |

All quantitative parameters, computational latencies, and accuracy matrices can be audited and reproduced locally by executing the automated regression suite via `pytest test_slow/finance/`. Dynamic network connection capabilities are required to initialize the streaming ingestion layers for the OpenML, NSL-KDD, and UCI repositories.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
