# Market surveillance: multi-stage statistical processing and online anomaly consensus orchestration

The Dense-Armor pipeline, composed of five tools, was tested on five years of historical market data. Following a 250-day warm-up period, the two integrated anomaly detectors isolated 43 of the remaining 1,004 days, accurately identifying the market crashes of August 2024 and April 2025.


## The scene

During systemic macro-economic shifts—such as the Tokyo stock market contraction on August 5th, 2024, or the international tariff announcements on April 2nd, 2025—traditional descriptive news-feeds generate significant operational latency, reporting market impacts only after material capital exposure has already occurred. 

An institutional surveillance framework must bypass text-based sentiment and isolate structural market anomalies directly from raw quantitative parameters. 

To benchmark this capability on historical data, the Dense-Armor multi-stage pipeline was deployed over **1,253 historical trading days** (spanning 14 October 2021 to 13 October 2026). Following a standard 250-day warm-up period, the dual-model consensus engine independently isolated **43 anomalous sessions** out of the remaining 1,004 days on the S&P 500 without prior manual rule injection. The detected sessions comprise 11 extreme price movements exceeding 2%, 12 high-volume quarterly-expiration Fridays, and 20 low-volatility structural shifts where joint price-volume distributions deviated from established historical baselines.


## Core Capabilities

| Tool | Strategic Function | Operational Impact |
|---|---|---|
| **Armatura** | Real-time data sanitization | Discards and patches anomalous price ticks caused by data-feed glitches or network dropouts before they trigger algorithmic execution loops. |
| **CUSUM** | Sub-clinical drift detection | Tracks continuous micro-movements in asset behavior, triggering alerts when cumulative variations confirm a systemic shift in market sentiment. |
| **Quantile sketch** | Bounded risk estimation | Maintains a dynamic, constant-memory summary of extreme risk tails, calculating daily Value-at-Risk parameters in under 100 microseconds. |
| **LODA** | Multi-dimensional outlier profiling | Projects joint asset attributes onto sparse sub-spaces to score complex, concurrent price and volume anomalies. |
| **Half-Space Trees** | Mass-based density monitoring | Establishes hierarchical spatial density structures to cross-validate outlier signals and ensure highly conservative alert dispatching. |


![S&P 500 daily return over five years with the 43 flagged days in red](../assets/finance/surveillance.png)

*Grey line: the daily return of the S&P 500. Red dots: the 43 flagged days. The grey band is the 250-day warm-up.*


## 1. High-speed input sanitization (Armatura)
High-frequency execution streams are highly vulnerable to corrupt data packages and network artifacts that distort nominal price spreads. Armatura operates as a real-time digital filtering layer, automatically evaluating the immediate historical envelope to intercept and sanitize absurd price points. Across the five-year S&P 500 benchmark, the filter discarded 1 tick out of 1,253 sessions.


## 2. Structural market sentiment tracking (CUSUM)
While macroeconomic culls are immediately visible, slow and systemic modifications in asset distributions frequently escape conventional threshold checks. This module continuously integrates marginal price deviations over time. When the cumulative sum surpasses statistical noise thresholds, the system flags a change in baseline market regime. Over the historical dataset, the sequential estimator isolated 22 structural regime shifts (1.8% of the stream).


## 3. Real-time value-at-risk estimation (Quantile sketch)
Calculating capital risk distribution under strict regulatory frameworks typically demands exhaustive retrospective database sweeps. The integrated quantile sketch updates dynamic percentile boundaries sample by sample while maintaining a bounded memory footprint. Run one session at a time, it generates updated 99% and 99.9% VaR percentiles within a median latency of 98 microseconds per session.


## 4. Multi-model consensus anomaly isolation (LODA + Half-Space Trees)
Price modifications cannot be evaluated accurately without conditioning them against concurrent liquidity parameters. This pipeline forces a strict consensus architecture between two independent density estimators. LODA evaluates feature vectors across sparse projections, while Half-Space Trees partitions the joint distribution space orthogonally. An anomaly is validated only upon simultaneous threshold violation by both layers.

Across the 1,004-day active monitoring window, the consensus engine successfully isolated 43 anomalous sessions, including:
- **August 5th, 2024:** Institutional liquidation ("Black Monday"), S&P 500 contraction at −3.00%
- **November 6th, 2024:** Post-election macroeconomic readjustment, +2.53%
- **April 3rd and 4th, 2025:** Tariff implementation response, −4.84% and −5.97%
- **April 9th, 2025:** Operational pause rebound, +9.52%

When scaled to individual asset monitoring, the baseline parameters automatically adapt to specific ticker volatility envelopes: Apple (AAPL) yielded 50 anomalies (including the June 11th, 2024, structural expansion following product announcements), Microsoft (MSFT) yielded 66 anomalies, and Tesla (TSLA) yielded 38 anomalies.


## 5. Algorithmic trade execution shielding (Orca)
During verified anomalous volatility windows, downstream execution strategies face heightened slippage risks. Orca monitors the final output layer, automatically applying adaptive damping constraints to incoming order sizes to mitigate market-impact costs. In the five-year run, the shield throttled 1 simulated order.


## Execution Workflow

For every arriving asset tick, the multi-stage surveillance architecture processes data linearly:

1. **Sanitization Filter:** **Armatura** checks and purges anomalous data-feed artifacts.
2. **Statistical Updates:** Clean data is dispatched to **CUSUM** for regime tracking and to the **Quantile sketch** for value-at-risk evaluation.
3. **Consensus Profiling:** Joint price-volume attributes are evaluated by **LODA** and **Half-Space Trees** to detect multi-dimensional outliers.
4. **Execution Damping:** If downstream algorithms attempt execution, **Orca** scales or clamps order sizes based on current anomaly scores.

**Compute Footprint:** The complete sequence has a median execution time of **3.85 milliseconds** and 5.27 milliseconds at the 99th percentile on standard 4-core hardware.


## Strategic Business Takeaway

- **Context:** A preprint on streaming detection of early defaults reports a 9.2-day lead over batch gradient boosting at equal false-positive rates (Li et al., 2026, not peer reviewed). It is a different task on different data, not a result of this pipeline.
- **Why it matters:** Payment and market fraud cost over **45 billion USD** in 2022 (Huang et al., 2025). On these five years a fixed 3% cutoff flags 7 days and the adaptive pipeline 43 (see [static vs online](static_vs_online.md)).


## Details

### Implementation

The pipeline is in `pipelines/finance/surveillance.py`.
Primitives chained, each with its own page:

- [Armatura](../shields/armatura.md)
- [CUSUM](../drift/cusum.md)
- [LODA](../anomaly/loda.md)
- [Half-Space Trees](../anomaly/hst.md)
- [Orca](../shields/orca.md)

### References

- Lou, B., Boland, D., Leong, P. H. W. (2024). *fSEAD.* arXiv:2406.05999. Section 2.1 (Loda).
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). *Fast anomaly detection for streaming data.* IJCAI. (Half-Space Trees.)
- Ulrich (2025). *Half-Space Trees for Novelty Detection.* arXiv:2505.08489.
- Pearson, R. K., Neuvo, Y., Astola, J., Gabbouj, M. (2016). *Generalized Hampel Filters.* EURASIP JASP 2016:87.
- Li, E., Thompson, D., Chen, M. (2026). *Real-Time Early-Default Detection Using Streaming Machine Learning with Multi-Source Behavioral Signals.* Preprints.org, doi: 10.20944/preprints202601.0866.v1 (not peer reviewed).
- Huang, A., et al. (2025). *Dynamic Calibration of Decision Thresholds for Financial Anomaly Detection.* Journal of Global Information Management 33(1), doi: 10.4018/JGIM.395852.
- Market data: Yahoo Finance, S&P 500 (^GSPC), Apple (AAPL), Microsoft (MSFT), Tesla (TSLA), daily close, 14 October 2021 to 13 October 2026.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
