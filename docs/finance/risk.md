# Risk: sequential quantile sketch architectures for streaming value-at-risk estimation

This section evaluates the computational efficiency and statistical precision of an online risk-monitoring pipeline, combining a fully bounded quantile sketch (DDSketch) with a sequential change-detection framework (CUSUM) to estimate dynamic Value-at-Risk (VaR) and isolate volatility drift in high-frequency streams.

**In plain words:** Every morning, a bank or investment fund needs to know exactly how much money it could lose if the market turns bad. Instead of loading decades of historical files into slow spreadsheets, this tool processes new market returns one day at a time in less than a millisecond, using compact mathematical buckets to calculate risk percentiles with 1% accuracy while instantly flagging if overall market volatility is starting to rise.

## The scene

It is 7:00 in the morning at a European fund. The risk committee meets at 9:00 and wants one number: **how much can we lose today, if the market turns ugly**. The old way is to open ten years of returns, compute percentiles in Excel, save the sheet. It takes half an hour and the market has already moved.

At 7:01 a single call to the pipeline gives the number. At 9:00 the committee has it, refreshed with yesterday's session.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| DDSketch | Remember how big the tail is | The worst 1% loss of the year, in one number |
| CUSUM | Add up small changes until a bell rings | The loss series is drifting upward, and nobody sees it yet |

![Daily loss of the S&P 500 over five years with the 99% risk line in red](../assets/finance/risk.png)

*Each grey bar is the loss of one day. The red line is the VaR at 99%: the loss that 99 days out of 100 stay below.*

## 1. Sizing the worst day (DDSketch)

Every bank, every fund, every trading desk has to know: if today is a bad day, how bad. The answer is a single number, the loss that 99 days out of 100 stay below.

The classic way is to keep every return and compute the percentile at the end. With millions of days the file does not fit anywhere. **DDSketch keeps the tail in a fixed number of buckets, and the answer is accurate to 1%.**

## 2. Noticing that losses are rising (CUSUM)

The VaR number is a snapshot. If it starts moving up week after week, the mood of the market is changing, and you want to know before the committee reads the number.

CUSUM adds up the small moves in the loss series. When the sum stops looking like noise, it rings the bell.

## The chain

For every new day:

1. The day's loss goes to **DDSketch**. The tail is updated, and the VaR at 99% and 99.9% is ready.
2. The same loss goes to **CUSUM**. If the recent losses are systematically larger than the older ones, the alarm fires.
3. The committee gets two numbers and one flag. No history re-scan, no file to open.

## What you get on five years of S&P 500

Tested on 1,253 real trading days. One call per day.

| Metric | Value |
|---|---|
| VaR 99% | 2.93% |
| VaR 99.9% | 4.28% |
| Days with a loss above the final VaR 99% | 14 out of 1,253 (1.1%) |
| Days with a loss above the final VaR 99.9% | 3 out of 1,253 (0.24%) |
| Days above a 5% loss | 1 out of 1,253 (4 April 2025, −5.97%) |
| Days with a volatility-drift alarm | 49 out of 1,253 (3.9%) |
| Time per day | 98 microseconds (median, Kaggle CPU) |

## When you need it

- Risk committee reporting: one number a day, no history re-scan.
- Intraday stress: call `process_day` on a rolling window of the last hour for an updated VaR.
- Regulatory limits: the `over_limit` flag is the first thing the desk sees.

## Details

### Implementation

The pipeline is in `pipelines/finance/risk.py`.
Primitives chained, each with its own page:

- [DDSketch](../stats/sketches.md)
- [CUSUM](../drift/cusum.md)

### References

- Masson, C., Rim, J. E., Lee, H. K. (2019). *DDSketch.* PVLDB 12(12), 2195-2205. arXiv:1908.10693.
- Reynolds, M. R. Jr., Stoumbos, Z. G. (1975). *A CUSUM for monitoring the process mean.*
- Market data: Yahoo Finance, S&P 500 (^GSPC), 14 October 2021 to 13 October 2026.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
