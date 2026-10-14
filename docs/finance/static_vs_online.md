# Static vs online: what changes when the thresholds move

**In plain words:** A fixed cutoff and an adaptive detector look at the same five years of market data and disagree on more than a third of the days. This page shows both, and what each one catches.

## The static baseline

The simplest way to flag an anomalous day is a fixed number. If the return is bigger than three percent, call it anomalous. One line of code, what most tutorials do, where everyone starts.

```python
from pipelines.finance import AnalisiStatica

a = AnalisiStatica(soglia=0.03)
r = a.analizza(rendimenti)
print(r.n_anomali)
```

On the S&P 500 over five years:

| Threshold | Days flagged |
|---|---|
| 2% | 63 out of 1,253 |
| 3% | 20 out of 1,253 |
| 4% | 6 out of 1,253 |

The number moves with the threshold. Nothing tells you which threshold is right.

## The online version

The same idea, no threshold to pick. Two detectors (LODA and Half-Space Trees) each build their own baseline from the data they see, and a day is flagged only when **both** agree.

```python
from pipelines.finance import AnalisiOnline

a = AnalisiOnline(range_ret=(-0.05, 0.05), range_vol=(0.5, 2.0), seed=0)
r = a.analizza(rendimenti, volumi_rel)
print(r.n_anomali)
```

On the same S&P 500, after a warm-up of 250 days:

| Metric | Value |
|---|---|
| Days flagged | 42 out of 1,004 |
| Final LODA threshold | 4.000 |
| Final HST threshold | 0.532 |

No number to pick. The two thresholds are what the tools converged to on this data.

## Where they agree, and where they do not

Same data, same period, side by side:

| Group | Static (3%) | Online (LODA+HST) |
|---|---|---|
| Flagged days | 12 | 42 |
| Flagged by both | 7 | 7 |
| Flagged only by one | 5 | 35 |

The seven days both catch point at two events:

- **August 5th 2024** — the Japanese Black Monday, S&P 500 at −4.19%.
- **April 4th to 14th 2025** — six consecutive sessions inside the tariff-announcement week.

Those are the days any method sees. They are big price moves, above 3% by definition, so the fixed cutoff catches them too.

## What only the static catches

Five days, all with a return between 3% and 5%:

| Date | Return |
|---|---|
| 2022-10-14 | +4.83% |
| 2022-11-03 | −3.11% |
| 2022-12-01 | +3.28% |
| 2022-12-13 | +3.30% |
| 2025-04-23 | +3.61% |

These are real moves, and the online detectors did not flag them. That is by design: after living through the 2022 volatility, the online thresholds had moved up. A 4% move was no longer unusual for the S&P 500 in that period. **The static threshold kept firing; the online one had learned.**

This is what *adaptive* means. If the market is chaotic for months, a 3% day stops being news, and the detector treats it as such.

## What only the online catches

35 days. Two patterns stand out.

**Quarterly expiration and month-end.** Many of these dates are the third Friday of March, June, September and December (options and futures expiration) or the last trading day of a month:

| Date | Return | What it is |
|---|---|---|
| 2022-12-16 | −1.70% | Quarterly expiration |
| 2023-03-17 | +2.06% | Quarterly expiration |
| 2023-06-16 | +1.73% | Quarterly expiration |
| 2023-09-15 | +0.23% | Quarterly expiration |
| 2024-03-15 | −1.00% | Quarterly expiration |
| 2024-06-21 | −0.60% | Quarterly expiration |
| 2024-09-20 | +0.12% | Quarterly expiration |

Every one of these has a return smaller than 3%, so the fixed cutoff never sees them. But the **ratio between price and volume** was different from a normal day. Expiration Fridays move huge amounts of volume without moving the price much. The online detectors see that, because they look at the pair (return, volume), not just the return.

**Events the static threshold was too slow to catch.** A few more days in the same list, but for a different reason:

| Date | Return | What it is |
|---|---|---|
| 2024-11-06 | +2.49% | The day after the US election |
| 2025-01-27 | −2.49% | The DeepSeek shock on AI stocks |
| 2025-02-03 | −2.09% | The first Trump tariff on Canada and Mexico |

Each of these was a real market event, and each stayed below 3%. The fixed cutoff treated them as ordinary days; the online detectors saw the price-and-volume pair shift and flagged them.

## When to use which

**Static** if you want the simplest possible baseline, or if you already know what size of move matters to you. It is honest, fast, and wrong exactly when the market regime changes.

**Online** if the market you watch changes its behaviour over time, and if the events you care about are not always big price moves. You get the extreme days plus the structural ones, at the cost of the tool adapting to the volatility of the moment.

The two are not in competition. The static pipeline is the baseline; the online pipeline is the answer to the case the baseline cannot cover.

## Details

### References

- Market data: Yahoo Finance, S&P 500 (^GSPC), 14 October 2021 – 13 October 2026, 1,254 trading days.
- Lou, B., Boland, D., Leong, P. H. W. (2024). fSEAD. arXiv:2406.05999. Section 2.1 (Loda).
- Tan, S. C., Ting, K. M., Liu, T. F. (2011). Fast anomaly detection for streaming data. IJCAI. (Half-Space Trees.)
- Ulrich (2025). Half-Space Trees for Novelty Detection. arXiv:2505.08489.

*The numbers on this page come from a single run on real historical data. They are not a forecast, and not financial advice.*
