# Execution: should this order leave?

**In plain words:** A trading desk sends thousands of orders a day. This page shows the check that runs on each one before it goes out.

## The scene

A desk in London. The automated strategy wants to buy 700 BTC in one shot. On a normal day it buys 5. If the order goes through, the desk moves the market against itself and loses money on the way in.

The pipeline sees the size, compares it to the usual one, and either sends it smaller or refuses it. **The whole check takes a few microseconds.**

## What is in the box

| Tool | Its one job | Real example |
|---|---|---|
| Rolling mean | Remember the usual order size | Size 5 is normal, 50 is suspicious |
| Orca | Dampen a suspicious order before it leaves | A 6x order goes out at 3x, smaller but valid |

## 1. Refusing the enormous orders (rolling mean)

The first check is the simplest: is this order ten times bigger than the usual one? If yes, it is refused. No legitimate model sends an order that big by accident, and if it does, the desk wants to know before it goes out.

## 2. Dampening the suspicious ones (Orca)

Between 5x and 10x the rolling mean there is a middle ground. The order is not absurd, but not ordinary either. Orca is called here, and only here: it dampens the size, the order still goes out, but smaller.

Orca costs about 20 milliseconds per call. On a natural distribution of sizes, only 1 to 2 percent of orders fall in this range, so the average check stays below 10 microseconds.

## The chain

For every order:

1. The order size is compared to the **rolling mean**.
2. Above 10x: refused outright.
3. Between 5x and 10x: goes through **Orca**, comes out smaller.
4. Below 5x: passes without further check.

## What you get on 1,000 real BTC/USD trades

One thousand trades from the Coinbase public API on 14 October 2026. Sizes from 0.00000001 to 0.7 BTC.

| Metric | Value |
|---|---|
| Passed directly (no Orca) | 968 (96.8%) |
| Went through Orca | 12 |
| Refused (10x rule) | 20 (2.0%) |
| Dampened by Orca | 0 |
| Time per order | a few microseconds |

Zero dampened is not a bug: on a natural distribution the 5x-10x range is almost empty. Orca stays as a safety catch for the day a runaway model sends 50x.

## When you need it

- Any desk that executes automated strategies.
- A broker that wants a last line of defence before the order reaches the exchange.
- Compliance: every refused or dampened order is logged.

## Details

### The pipeline

The class is `ExecutionPipeline`, in `pipelines/finance/execution.py`. Two primitives:

- `RunningMoments` from `dense_armor.utility.stats.moments`, for the rolling mean and standard deviation of order sizes.
- `Orca` from `dense_armor.utility.protect.orca`, for the damping step.

```python
from pipelines.finance.execution import ExecutionPipeline

ep = ExecutionPipeline(seed=0)
for _ in range(30):
    ep.process_order(5.0)

print(ep.process_order(6.0).approved)
print(ep.process_order(500.0).approved)
print(ep.process_order(500.0).reason)
```

### Parameters you can tune

| Name | Default | Meaning |
|---|---|---|
| `damping_threshold` | 0.30 | reduction above which Orca counts as dampened |
| `max_size_multiple` | 10.0 | size above this times the mean: refused |
| `orca_pre_threshold` | 5.0 | size above this times the mean: goes through Orca |
| `warmup` | 20 | orders before the rolling mean is trusted |

### References

- Orca: see [docs/shields/orca.md](../shields/orca.md).
- Running mean: Welford, B. P. (1962). *Note on a method for calculating corrected sums of squares and products.* Technometrics 4(3), 419-420.
- Trade data: Coinbase Exchange public API, endpoint `/products/BTC-USD/trades`, 14 October 2026, 1,000 rows.

*The numbers on this page come from a single run on real trade data. Not a forecast, not financial advice.*
