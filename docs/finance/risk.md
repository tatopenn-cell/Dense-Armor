# Risk: how much can we lose today?

**In plain words:** Every morning a bank has to answer one question. This pipeline answers it in a thousandth of a second, without reading a single line of history.

## What it does

You give it one number a day: how much you lost. It gives back two numbers and two flags:

- **VaR 99%** — the loss that 99 days out of 100 stay below.
- **VaR 99.9%** — the same, for the top 0.1% of days. The once-every-four-years number.
- **oltre_limite** — true when today's loss crossed the absolute limit you set.
- **deriva_volatilita** — true when the loss series is shifting upward. CUSUM on the loss series, gradual, not one big day.

## The tools behind it

- **DDSketch** ([dense_armor.utility.stats.quantiles](../stats/sketches.md)) keeps the tail of the loss distribution with a relative-error guarantee of 1%. It stores a fixed number of buckets, no matter how long the stream is.
- **CUSUM** ([dense_armor.utility.drift.cusum](../drift/cusum.md)) adds up the small moves in the loss series and rings when the sum stops looking like noise.

## What it does on five years of S&P 500

Tested on 1,253 real trading days (October 2021 to October 2026). One call per day.

| Metric | Value |
|---|---|
| VaR 99% | 2.87% |
| VaR 99.9% | 4.20% |
| Days above a 5% loss | 1 out of 1,253 |
| Days with a volatility-drift alarm | 53 out of 1,253 (4.2%) |
| Time per day | about 116 microseconds |

The one day above 5% is **7 April 2025**, the first session of the tariff-announcement week. The 5 August 2024 "Black Monday" stayed at −4.19%, below the limit: the risk pipeline reports it as a large day, not as a limit breach.

## Smallest example

```python
import numpy as np
from pipelines.finance.rischio import RischioPipeline

rp = RischioPipeline(alpha=0.01, limite_perdita=0.05, seed=0)
rng = np.random.default_rng(0)
for _ in range(200):
    rp.process_day(float(abs(rng.normal(0.01, 0.005))))

e = rp.process_day(0.10)
print(round(e.var_99, 4), round(e.var_999, 4), e.oltre_limite)
```

You get a VaR 99% around `0.02`, a VaR 99.9% slightly above it, and `oltre_limite=True` because 10% is above the 5% limit.

## When you need it

- Risk committee reporting: one number a day, no history re-scan.
- Intraday stress: call `process_day` on a rolling window of the last hour to get an updated VaR.
- Regulatory limits: the `oltre_limite` flag is the first thing the desk sees.

## References

- Masson, C., Rim, J. E., Lee, H. K. (2019). *DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees.* Proceedings of the VLDB Endowment, 12(12), 2195-2205. arXiv:1908.10693.
- Reynolds, M. R. Jr., Stoumbos, Z. G. (1975). *A CUSUM for monitoring the process mean.*

*The numbers on this page come from a single run on real historical data. They are not a forecast, and not financial advice.*
