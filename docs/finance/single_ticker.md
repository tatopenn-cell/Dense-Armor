# Single ticker: is today strange?

**In plain words:** If you follow one stock, this page shows two detectors that tell you when a day is different from the usual.

## The scene

A portfolio manager follows Apple. Every morning she opens the charts and wants one thing: is today one of those days when Apple behaves differently, or is it a normal Thursday?

Two detectors answer the question together. Only when **both** agree, the day is flagged. A single detector is not enough: one can be wrong, two rarely are.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| LODA | Score how strange a day is, given several numbers | Price and volume together, not just price |
| Half-Space Trees | Same job as LODA, done another way | A second opinion on the same day |

## 1. A strangeness score (LODA)

Price and volume usually move together. When they do not, something is off. LODA looks at the pair and gives a score: low for an ordinary day, high for a strange one.

## 2. A second opinion (Half-Space Trees)

Half-Space Trees does the same job, but by cutting the space into rectangles instead of projections. When both detectors give a high score, the day is flagged.

## The chain

For each new day:

1. The return and the relative volume go to **LODA**.
2. The same pair goes to **Half-Space Trees**.
3. If both scores are above their adaptive thresholds, the day is flagged. Otherwise it is not.

## What you get on Apple over five years

Apple, 1,253 trading days. Ranges calibrated on the first 250 days.

| Metric | Value |
|---|---|
| Days tested | 1,253 |
| Days flagged anomalous | 43 |
| Time per day | about 1.1 ms |

The 43 days include June 2022 and September 2023 crashes, plus the quarterly-expiration Fridays, when volume moves without the price moving much.

**Why 43 and not a handful.** Same two detectors used on the S&P 500 would flag about 40 days too, if the input is raw. The surveillance page filters the tick with Armatura before the detectors see it, and the count drops. Both behaviours are correct: they answer different questions.

## When you need it

- A single stock that a portfolio manager follows closely.
- A pair trade where one leg is watched and the other is hedged.
- A retail investor who wants one number a day for one ticker.

## Details

### Implementation

The class is `SingleTickerPipeline`, in `pipelines/finance/single_ticker.py`. It chains [LODA](../anomaly/loda.md) and [Half-Space Trees](../anomaly/hst.md) on the (return, volume) pair of one stock. Each detector keeps its adaptive threshold; the pipeline only takes the AND of the two. See the linked pages for the algorithms.

### References

- Lou, B., Boland, D., Leong, P. H. W. (2024). *fSEAD.* arXiv:2406.05999. Section 2.1 (Loda).
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). *Fast anomaly detection for streaming data.* IJCAI. (Half-Space Trees.)
- Ulrich (2025). *Half-Space Trees for Novelty Detection.* arXiv:2505.08489.
- Market data: Yahoo Finance, Apple (AAPL), 14 October 2021 to 13 October 2026.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
