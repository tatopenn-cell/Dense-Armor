# Finance

**In plain words:** Five Dense-Armor pipelines cover the questions a market analyst asks every day. This page is the index.

## The five pipelines

| Pipeline | Question it answers | Page |
|---|---|---|
| **Surveillance** | Is the market behaving differently today? | [Surveillance](surveillance.md) |
| **Risk** | How much can we lose today, and is volatility rising? | [Risk](risk.md) |
| **Execution** | Should this order leave, and how big? | [Execution](execution.md) |
| **Single ticker** | Is today strange for this one stock? | [Single ticker](single_ticker.md) |
| **Static vs online** | How does a fixed cutoff compare to the adaptive detectors? | [Static vs online](static_vs_online.md) |
| **Fraud detection** | Is this transaction real? | [Fraud detection](fraud.md) |
| **Unsupervised fraud** | Is this transaction strange, even without labels? | [Unsupervised fraud](unsupervised_fraud.md) |
| **Intrusion detection** | Is this network packet an attack? | [Intrusion detection](intrusion.md) |
| **Credit scoring** | Will this borrower repay? | [Credit scoring](credit_scoring.md) |

## What they have in common

Every pipeline chains primitives from `dense_armor.utility` in a documented order. None of them adds new algorithms. Each page has:

- A **commercial opening**: the scene, what is in the box, the numbered situations, the chain, the numbers on real data.
- A **technical Details** section: the pipeline internals, the code example, the parameters, the references.

## The data behind every number

All the numbers on these pages come from one of the two real datasets shipped with the tests:

| Data | Coverage | File |
|---|---|---|
| S&P 500, Apple, Microsoft, Tesla | 1,254 trading days each, 14 Oct 2021 to 13 Oct 2026 | `test/fixtures/finance/*.csv` |
| BTC/USD trades | 1,000 trades, 14 Oct 2026 | `test/fixtures/finance/coinbase_btcusd_trades.csv` |

The fixture files are checked in, so the numbers can be reproduced by running `pytest test/finance/`.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
