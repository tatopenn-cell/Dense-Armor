# Single ticker: is today strange?

**In plain words:** If you follow one stock, this pipeline tells you when a day is different from the others. Two independent detectors must agree before it rings.

## What it does

Every day, the pipeline receives two numbers for one stock: the **return** (how much the price moved) and the **relative volume** (how much was traded, divided by the usual amount). It gives back two scores and a flag:

- **LODA score** — a strangeness score on the (return, volume) pair.
- **HST score** — the same, computed by Half-Space Trees.
- **anomalo** — True only if **both** scores are above their own adaptive thresholds.

Two votes, not one. A day that looks strange only to LODA is not flagged.

## The tools behind it

- **LODA** ([dense_armor.utility.anomaly.loda](../anomaly/loda.md)) projects the (return, volume) point onto random 1-D directions and histograms each projection. Rare bins give a high score.
- **Half-Space Trees** ([dense_armor.utility.anomaly.hst](../anomaly/hst.md)) splits the space with an ensemble of trees and scores by how isolated the point is.

Both tools keep their own adaptive thresholds, the way they were designed. The pipeline only takes the AND of the two.

## What it does on Apple over five years

Apple, 1,253 trading days. Ranges calibrated on the first 250 days. Every number below comes from the same run on real data.

| Metric | Value |
|---|---|
| Days tested | 1,253 |
| Days flagged anomalous | 43 |
| Time per day | about 1.1 ms |

The 43 days include:

- **13 June 2022** (−5.28%) — the selloff after the May CPI print.
- **7 September 2023** (−7.02%) — the China iPhone ban.
- **16 September 2022**, **15 December 2023** and other quarterly-expiration Fridays, when volume moves without the price moving much.

The list is not dominated by price crashes. Many of the flagged days have small returns but unusual **volume**, exactly the case that a return-only threshold cannot catch.

## Why 43 and not 4

Same two tools as the surveillance pipeline, but no upstream filter. In surveillance, Armatura cleans the tick before LODA and HST see it, and the count on the S&P 500 drops from 42 to a few. On raw data, the two detectors are more sensitive. Both behaviours are correct: they answer different questions.

## Smallest example

```python
from pipelines.finance.anomalia_titolo import AnomaliaTitoloPipeline

t = AnomaliaTitoloPipeline(range_ret=(-0.05, 0.05), range_vol=(0.5, 2.0), seed=0)
for _ in range(7):
    t.process_day(0.001, 1.0)

e = t.process_day(-0.05, 3.0)
print(round(e.loda_score, 3), round(e.hst_score, 3), e.anomalo)
```

You get a LODA score well above 1, an HST score above its adaptive threshold, and `anomalo=True`.

## When you need it

- A single stock that a portfolio manager follows closely.
- A pair trade where one leg is watched and the other is hedged.
- A retail investor who wants one number a day for one ticker.

## Details

### References

- Lou, B., Boland, D., Leong, P. H. W. (2024). *fSEAD.* arXiv:2406.05999. Section 2.1 (Loda).
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). *Fast anomaly detection for streaming data.* IJCAI.
- Ulrich (2025). *Half-Space Trees for Novelty Detection.* arXiv:2505.08489.
- Market data: Yahoo Finance, Apple (AAPL), 14 October 2021 – 13 October 2026, 1,253 trading days.

*The numbers on this page come from a single run on real historical data. They are not a forecast, and not financial advice.*
