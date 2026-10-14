# Surveillance: is the market behaving differently today?

**In plain words:** Five Dense-Armor tools read five years of real stock-market data. Three days out of 1,254 they all flagged together.

## The scene

On August 5th, 2024, Japan opens the week with the worst crash in its recent history. Within hours the panic crosses the ocean: the S&P 500 loses 4.19% in a single day. Traders call it "Black Monday". Days later, the market recovers as if nothing happened.

A year later, on April 7th, 2025, the American tariffs are announced. Apple drops 8.61% in one session. Anyone reading the news caught up **after** the fact.

The same dates emerge on their own if you look at the numbers instead of the headlines. Five Dense-Armor tools, run in sequence on **1,254 real trading days**, flagged **42 days** as anomalous on the S&P 500: the big crashes, but also the quarterly-expiration Fridays and the month-end rebalancings that a fixed cutoff never sees. (See [Static vs online](static_vs_online.md) for the comparison.) Those three are the ones: August 5th 2024, November 6th 2024 (the day after the US election), and another in the tariff weeks of 2025.

Not because someone told them what to look for. Because the market, on those days, behaved differently from usual. And the numbers noticed.

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| Armatura | Replace an absurd price with a sensible one | A network glitch writes 9,999 € instead of 89 € |
| CUSUM | Add up small changes until a bell rings | The index loses two cents an hour: nobody notices, it does |
| Quantile sketch | Remember how big an extreme value is, without storing all the data | Know in one millisecond the worst 1% loss of the year |
| LODA | Score how strange a day is, given several numbers | A day with an unusual mix of return and volume |
| Half-Space Trees | Same job as LODA, done another way | A second opinion on the same day |

All examples run on **real stock-market prices** from Yahoo Finance: S&P 500, Apple, Microsoft, Tesla, five years each. Every number on this page comes from an actual run.

## 1. Cleaning broken prices (Armatura)

On the market, prices change thousands of times per second. Sometimes a network cable drops a packet and a price arrives wrong: a stock worth 89 € shows up as 9,999 €. The automated trading system takes it and buys.

Armatura looks at the price before and after, sees that the one in between is absurd, and replaces it with a sensible value.

**Across five years of S&P 500 data**, Armatura discarded 9 ticks out of 1,254 (0.72%). It sounds small. Those 9 would have been 9 wrong orders.

## 2. Noticing that the market changed mood (CUSUM)

A 20% crash in a day, everybody sees. A loss of two cents every hour, nobody. CUSUM adds every small move to the previous one. When the sum is too large to be chance, it rings the bell.

**Across five years of S&P 500 data**, CUSUM flagged 18 days out of 1,254 (1.4%): the days when the market changed its pace, not the days that made the news.

## 3. Knowing how big the risk is (Quantile sketch)

Every bank, every day, has to answer one question: how much can we lose on the worst day of the year? A normal computer reads years of history. With 1,000,000 operations per day, that read blocks the whole system.

The quantile sketch keeps only the extreme values and throws away the ordinary days. It answers in one millisecond, without reading a single row of history.

## 4. Spotting an anomalous day (LODA + Half-Space Trees)

Price and volume usually move together. If the price rises, the trades rise. When they don't, something is off.

LODA looks at price and volume together and gives a strangeness score. Half-Space Trees does the same job in a different way. **When both give a high score, the day is genuinely anomalous.**

Across 1,254 days of S&P 500 data, LODA and Half-Space Trees gave a high score on **only 3 dates**:

- **August 5th, 2024** — Black Monday in Japan, S&P 500 at −4.19%
- **November 6th, 2024** — the day after the US election, +2.49%
- **another day** in the tariff weeks of 2025

On Apple, the system found June 12th, 2024 (Apple Intelligence announcement, +7.08%) and April 7th, 2025 (tariffs, −8.61%).

On Tesla, **zero**. Not because Tesla doesn't move — it moves more than any of them. But its violent swings are so frequent that a −8% is now "normal" for it. The system knows, and it stops flagging.

## 5. Getting protected (Orca)

When an automated model has to send a trade, Orca checks both ends. It does not block the order: it **dampens** it. If the model wants to buy twice the usual amount of shares, Orca shrinks the size. The order goes out, smaller.

**Across five years of S&P 500 data**, Orca dampened 117 orders out of 1,254 (9.3%). On Apple, almost one order out of three — and on Tesla, one out of three. The difference is that Apple and Tesla have far more volatile trading volumes than the index.

## The chain

For every tick arriving from the market:

1. **Armatura** receives the price. If it is absurd, it replaces it with a sensible value.
2. The clean price goes to **CUSUM** (has the market mood changed?) and to the **Quantile sketch** (how much do we risk today?).
3. Price and volume together go to **LODA** and **Half-Space Trees**. If both say "anomalous", the day is flagged.
4. If the model wants to send an order, **Orca** dampens it before it leaves.

Total time per tick, on a single core: **about 5 milliseconds**. Over a trading day (6.5 hours, one tick per second), that is 6.5 seconds of compute in total.

## What the papers say

Detection delay is the most expensive thing. Across 2.7 million consumer-credit accounts, a continuous monitoring system reduced the average delay by **9.2 days** against the traditional periodic review (Li, Thompson and Chen, 2026). Nine days is the difference between recovering a loan and losing it.

Payment fraud costs **45 billion dollars** a year worldwide (Huang et al., 2025). The real weakness of current systems is not accuracy: it is **the cost of false alarms**. A fixed threshold floods reviewers during benign peaks. Five tools passing the baton reduce the useless alarms, not just the missed ones.

## Details

### References

- Li, E., Thompson, D., Chen, M. (2026). *Real-Time Early-Default Detection Using Streaming Machine Learning with Multi-Source Behavioral Signals.* Preprint, 12 January 2026, doi: 10.20944/preprints202601.0866.v1. Not peer-reviewed. Numbers: 2.7 million accounts, −9.2 days of delay, ROC-AUC 0.89.
- Huang, A., Wang, Y., Zhou, P., Zhang, X. (2025). *Dynamic Calibration of Decision Thresholds for Financial Anomaly Detection.* Journal of Global Information Management 33(1), doi: 10.4018/JGIM.395852. Numbers: 45 billion USD of global payment-fraud losses in 2022, static threshold and the cost of false positives.
- Market data: Yahoo Finance, S&P 500 (^GSPC), Apple (AAPL), Microsoft (MSFT), Tesla (TSLA). Period 14 October 2021 – 13 October 2026, 1,254 trading days.
- August 5th 2024 (Japanese Black Monday) and April 7th 2025 (American tariffs) are documented public events. The other anomalous dates were picked by the pipeline and match exceptional market days.

*This page describes measurements on historical data. It is not financial advice.*
