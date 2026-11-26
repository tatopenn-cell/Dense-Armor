# Financial pipelines: multi-model streaming architectures for risk, execution, and fraud detection

## The map of finance

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder). The pages are grouped in four sections.

<object type="image/svg+xml" data="../assets/maps/finance.svg" style="width:100%"></object>

This page serves as the index for nine analytical pipelines, and for the parts of the pre-trade system around them, that handle the daily data challenges faced by market analysts, banking infrastructure, and payment desks. Every pipeline chains together the core mathematical blocks of the package to solve practical business questions—like identifying market anomalies, filtering out fraudulent transactions, or sizing trade orders safely before execution.

## A first example: three orders, five controls

An order leaves the desk only if it passes the pre-trade controls. Three orders of a stock are checked against a size limit of 0.5 and a price band of 2% around the reference price of 100.

```python
from dense_armor.core.finance import PreTradeChecker

checker = PreTradeChecker(max_notional=100_000.0, max_size=0.5, price_band=0.02, max_orders=10, window_s=60.0, duplicate_window_s=1.0)
for size, price in ((0.4, 100.5), (0.9, 100.5), (0.4, 105.0)):
    d = checker.check("acct1", "buy", size, price, 100.0, t=size)
    print(size, price, d.approved, d.reason)
```

```
0.4 100.5 True ok
0.9 100.5 False size
0.4 105.0 False price_band
```

The first order is inside the limits and goes out. The second asks for a size of 0.9, above 0.5: refused for `size`. The third is priced 5% above the reference, outside the band of 2%: refused for `price_band`. The pages below build on these checks.

## The pages, by section

### Markets and risk

What the market is doing and how much it can cost: anomalies, tail risk, volatility regime, noise against drift, a portfolio, the order book and a flash crash.

| Pipeline | What it is for | Page |
|---|---|---|
| **Market Surveillance** | Multi-stage statistical processing and online anomaly consensus orchestration. | [Surveillance](surveillance.md) |
| **Value-at-Risk Estimation** | Sequential quantile sketch architectures for streaming tail-risk monitoring. | [Risk](risk.md) |
| **Regime** | Bayesian online changepoint detection on daily returns (BOCPD, Adams and MacKay 2007). | [Regime](regime.md) |
| **Drift vs noise** | Raw tick returns against a pre-averaged statistic, with a paired Wilcoxon and an A12. | [Drift vs noise](drift_vs_noise.md) |
| **Single Ticker Monitoring** | Dual-model online anomaly isolation for individual equity validation. | [Single ticker](single_ticker.md) |
| **Static vs Online Analysis** | Statistical divergence benchmarking between fixed-cutoff and adaptive frameworks. | [Static vs online](static_vs_online.md) |
| **Order book** | Order flow imbalance and the linear price impact of Cont, Kukanov and Stoikov (2010). | [Order book](order_book.md) |
| **Portfolio** | Minimum-variance and inverse-volatility weights from an online shrunk covariance (Bodnar et al. 2014). | [Portfolio](portfolio.md) |
| **Stress** | The existing pipelines on a synthetic flash crash (Braun et al. 2017). | [Stress](stress.md) |

### Orders and execution

From a decision to an order that leaves the desk: schedules, the five pre-trade controls, a guard on the model, and the messages that come and go.

| Pipeline | What it is for | Page |
|---|---|---|
| **Algorithmic Execution** | High-frequency pre-trade risk validation and automated order damping. | [Execution](execution.md) |
| **Execution Schedules** | TWAP and VWAP split of a parent order across time buckets. | [Schedules](schedule.md) |
| **Pre-trade limits** | Five controls of SEC Rule 15c3-5 (c)(1): size, price band, duplicate, rate, notional. | [Pre-trade limits](pretrade.md) |
| **Trading guard** | An online model stopped by a guard when the last tick jumps, with the pre-trade checker on the orders that remain. | [Trading guard](trading_guard.md) |
| **FIX gateway** | FIX 4.4 initiator that sends limit orders to a test acceptor and collects the execution reports. | [FIX gateway](fix_gateway.md) |
| **ITCH messages** | Decode the seven order messages of Nasdaq TotalView-ITCH 5.0. | [ITCH messages](itch.md) |
| **Feed file** | Framed Nasdaq ITCH messages read one record at a time by `iter_itch`, decoded by `decode_message`. | [Feed file](feed.md) |
| **Synthetic feed** | ITCH messages from a seed, byte-identical on every machine, same layout, no statistics. | [Synthetic feed](synthetic.md) |

### Fraud, intrusion and credit

Models that classify transactions, network flows and customers as they arrive.

| Pipeline | What it is for | Page |
|---|---|---|
| **Supervised Fraud Detection** | Real-time transaction classification via sequential ensemble tree models. | [Fraud detection](fraud.md) |
| **Unsupervised Fraud Detection** | Multi-model anomalous transaction profiling deployed on unlabelled streams. | [Unsupervised fraud](unsupervised_fraud.md) |
| **Network Intrusion Detection** | Drift-adaptive streaming tree architectures for real-time security perimeters. | [Intrusion detection](intrusion.md) |
| **Predictive Credit Scoring** | Ensemble online tree models for real-time customer conversion optimization. | [Credit scoring](credit_scoring.md) |

### Running the desk

The parts that keep a desk running and auditable: configuration, logs, status, a dashboard and a red button, a demo, timing, queues.

| Pipeline | What it is for | Page |
|---|---|---|
| **Configuration** | One TOML file with the parameters of the four blocks; frozen dataclasses, paths resolved against the file. | [Configuration](config.md) |
| **Decision log** | JSON lines with a SHA-256 hash chain; each record carries the hash of the previous one. | [Decision log](decision_log.md) |
| **Audit table** | The decision log of F6 in a PostgreSQL table, with a migration and a trigger that forbids UPDATE and DELETE. | [Audit table](audit_db.md) |
| **Status** | One JSON object with the configuration, the emergency stop, the feed and the log state. | [Status](status.md) |
| **Dashboard** | Flask app with `GET /status`, `POST /emergency-stop`, `GET /decisions.svg`. | [Dashboard](dashboard.md) |
| **Red button** | `GET /` serves a single HTML page with a red button that creates the stop file. | [Red button](emergency.md) |
| **Demo** | A feed, a checker and a log in one command; one JSON line with counts, log state and schedule sum. | [Demo](demo.md) |
| **Timing** | Median, IQR, p99 and p99/median of the time per sample, on a declared host. | [Timing](timing.md) |
| **Ring buffer** | Fixed-capacity queue with preallocated storage; cost against `BoundedStream`. | [Ring buffer](ring.md) |
| **Streaming** | A bounded queue with three policies and two lazy broker adapters. | [Streaming](streaming.md) |
| **Primitives map** | The public names of the finance modules, with their signatures: the map that the dashboard must expose. | [Primitives map](PRIMITIVES.md) |


## Architectural Integration & Layout Standard

Every operational pipeline is engineered by chaining algorithmic primitives directly from `dense_armor.core` in a deterministic, documented execution workflow. The components introduce no additional external code dependencies, ensuring constant-memory processing and strict execution boundaries. To guarantee consistency across the enterprise documentation suite, each module enforces a standardized structural template:

- **Executive Value Proposition:** Detailed deployment scenarios ("The scene"), structured core capabilities, functional execution workflows, and measured results on public benchmarks.
- **Technical Reference Specifications:** Pipeline internals, production code implementation blocks, hyperparameter tables, and peer-reviewed algorithmic citations.


## Data Benchmark & Verification Infrastructure

To ensure rigorous validation, all empirical performance metrics and mathematical convergence trajectories are calculated using public historical datasets. Two baseline fixture suites are distributed natively within the test architecture, while three high-capacity industrial benchmarks are ingested dynamically during continuous integration execution passes:

| Target Domain | Dataset Temporal & Volumetric Coverage | System Deployment Path |
|---|---|---|
| **Equity Surveillance & Risk** | S&P 500, AAPL, MSFT, TSLA: 1,254 daily closes, 1,253 returns (14 Oct 2021 to 13 Oct 2026) | `test/fixtures/finance/*.csv` |
| **High-Frequency Execution** | Coinbase BTC/USD execution stream: 1,000 real-time trades (14 Oct 2026) | `test/fixtures/finance/coinbase_btcusd_trades.csv` |
| **Supervised / Unsupervised Fraud** | OpenML `creditcard` benchmark: 284,807 transactions containing 492 verified frauds | Dynamic network ingestion |
| **Network Intrusion Security** | NSL-KDD benchmark stream: 125,973 labelled network packet flows | Dynamic network ingestion |
| **Conversion & Credit Scoring** | UCI Bank Marketing benchmark: 45,211 multi-feature telemarketing interactions | Dynamic network ingestion |

All quantitative parameters, computational latencies, and accuracy matrices can be audited and reproduced locally by executing the automated regression suite via `pytest test_slow/finance/`. Dynamic network connection capabilities are required to initialize the streaming ingestion layers for the OpenML, NSL-KDD, and UCI repositories.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
