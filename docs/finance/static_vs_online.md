# Static vs online: what changes when the thresholds move

**In plain words:** A fixed cutoff and an adaptive detector look at the same five years of market data and disagree on a third of the days. This page shows both, side by side.

## The scene

The simplest way to find an anomalous day is a fixed number: if the return is bigger than 3%, call it anomalous. It is one line of code, it is what every tutorial does, and it works until the market regime changes.

Five years of S&P 500 data show the limit of that approach. On the same data, the fixed threshold and the two adaptive detectors disagree on **35 days**. The next sections show which days each of them catches.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| StaticAnalysis | Flag days above a fixed cutoff | 20 days above 3% over five years |
| OnlineAnalysis | Same job, adaptive thresholds | 42 days flagged by LODA and HST together |

## 1. The static baseline (StaticAnalysis)

You pick a number. Anything above it is anomalous. On the S&P 500 over five years:

| Threshold | Days flagged |
|---|---|
| 2% | 63 out of 1,253 |
| 3% | 20 out of 1,253 |
| 4% | 6 out of 1,253 |

The number moves with the threshold. Nothing tells you which threshold is right.

## 2. The online version (OnlineAnalysis)

No threshold to pick. Two detectors (LODA and Half-Space Trees) each build their own baseline from the data they see, and a day is flagged only when **both** agree.

On the same series, after a warm-up of 250 days: **42 days flagged**, with final LODA threshold 4.0 and final HST threshold 0.53.

## 3. Where they agree

The seven days both catch point at two events:

- **August 5th 2024** — the Japanese Black Monday, S&P 500 at −4.19%.
- **April 4th to 14th 2025** — six consecutive sessions inside the tariff-announcement week.

These are the days any method sees. They are big price moves, above 3% by definition, so the fixed cutoff catches them too.

## 4. What only the static catches

Five days, all with a return between 3% and 5%:

| Date | Return |
|---|---|
| 2022-10-14 | +4.83% |
| 2022-11-03 | −3.11% |
| 2022-12-01 | +3.28% |
| 2022-12-13 | +3.30% |
| 2025-04-23 | +3.61% |

The online detectors did not flag them. That is by design: after the 2022 volatility, the online thresholds had moved up. A 4% move was no longer unusual. **The static threshold kept firing; the online one had learned.**

## 5. What only the online catches

35 days. Two patterns stand out.

**Quarterly expiration and month-end.** Many of these dates are the third Friday of March, June, September and December, or the last trading day of a month:

| Date | Return | What it is |
|---|---|---|
| 2022-12-16 | −1.70% | Quarterly expiration |
| 2023-03-17 | +2.06% | Quarterly expiration |
| 2023-06-16 | +1.73% | Quarterly expiration |
| 2023-09-15 | +0.23% | Quarterly expiration |
| 2024-03-15 | −1.00% | Quarterly expiration |
| 2024-06-21 | −0.60% | Quarterly expiration |
| 2024-09-20 | +0.12% | Quarterly expiration |

Each has a return smaller than 3%, so the fixed cutoff never sees them. But the **ratio between price and volume** was different from a normal day. Expiration Fridays move huge amounts of volume without moving the price much.

**Events the static threshold was too slow to catch.** A few more days in the same list:

| Date | Return | What it is |
|---|---|---|
| 2024-11-06 | +2.49% | The day after the US election |
| 2025-01-27 | −2.49% | The DeepSeek shock on AI stocks |
| 2025-02-03 | −2.09% | The first tariff on Canada and Mexico |

Each of these was a real market event, and each stayed below 3%. The fixed cutoff treated them as ordinary days; the online detectors saw the price-and-volume pair shift and flagged them.

## When to use which

**Static** if you want the simplest possible baseline, or if you already know what size of move matters to you. Honest, fast, wrong exactly when the market regime changes.

**Online** if the market you watch changes its behaviour over time, and if the events you care about are not always big price moves. You get the extreme days plus the structural ones, at the cost of the tool adapting to the volatility of the moment.

The two are not in competition. The static pipeline is the baseline; the online pipeline is the answer to the case the baseline cannot cover.

## Details

### Implementation

Two pipelines. `StaticAnalysis` in `pipelines/finance/static_analysis.py` applies a fixed cutoff to the daily return: one line, no dependency. `OnlineAnalysis` in `pipelines/finance/online_analysis.py` chains [LODA](../anomaly/loda.md) and [Half-Space Trees](../anomaly/hst.md) with their adaptive thresholds. See the linked pages for the algorithms; this page compares the two on the same real series.

### References

- Market data: Yahoo Finance, S&P 500 (^GSPC), 14 October 2021 to 13 October 2026.
- Lou, B., Boland, D., Leong, P. H. W. (2024). *fSEAD.* arXiv:2406.05999. Section 2.1 (Loda).
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). *Fast anomaly detection for streaming data.* IJCAI. (Half-Space Trees.)
- Ulrich (2025). *Half-Space Trees for Novelty Detection.* arXiv:2505.08489.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
