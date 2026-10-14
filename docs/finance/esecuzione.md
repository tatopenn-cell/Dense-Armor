# Execution: should this order leave?

**In plain words:** A trading desk sends thousands of orders a day. This pipeline looks at each one before it goes out, and answers two questions: is it safe, and how big should it be?

## What it does

Every order passes three checks, in order:

1. **Is it enormous?** If the size is more than ten times the rolling mean, the order is refused outright.
2. **Is it ordinary?** If the size is below five times the rolling mean, the order goes out unchanged. This is the case for almost every order.
3. **Is it suspicious but not absurd?** Only this middle ground goes through Orca, which dampens it. Orca costs about 20 milliseconds per call, so it must stay rare.

## Why the two thresholds

A trading desk measures time in microseconds. A check that costs 20 ms is unusable on every order. But an order that is 5x or 10x the usual size is exactly the case where the check is worth the cost. The pipeline pays for it only where it would hurt most to be wrong.

## What it does on 1,000 real BTC/USD trades

One thousand trades from the Coinbase public API on 14 October 2026. Sizes in BTC, from 0.00000001 to 0.7.

| Metric | Value |
|---|---|
| Orders processed | 1,000 |
| Passed directly (no Orca) | 968 (96.8%) |
| Went through Orca | 12 |
| Dampened by Orca | 0 |
| Refused (10x rule) | 20 (2.0%) |
| Time per order | a few microseconds |

The twenty refused orders are the largest trades of the sample. Half of them sit in short bursts (trades 43-45, 127-132, 985-987): when the rolling mean is still small, each new large trade in a burst looks huge. This is the expected behaviour with a slow-adapting mean and a fast market.

## Smallest example

```python
from pipelines.finance.esecuzione import EsecuzionePipeline

ep = EsecuzionePipeline(seed=0)
for _ in range(30):
    ep.process_order(5.0)

print(ep.process_order(6.0).approvato)
print(ep.process_order(500.0).approvato)
print(ep.process_order(500.0).motivo)
```

You get `True`, `False`, and the reason `size > max_size_molteplice * mean`.

## When you need it

- Any desk that executes automated strategies.
- A broker that wants a last line of defence against a runaway model.
- A compliance team that needs a log of every refused or dampened order.

## References

- Orca: see [docs/shields/orca.md](../../shields/orca.md) for the damping algorithm.
- Running mean: Welford, B. P. (1962). *Note on a method for calculating corrected sums of squares and products.* Technometrics 4(3), 419-420.
- Trade data: Coinbase Exchange public API, endpoint `/products/BTC-USD/trades`, 14 October 2026, 1,000 rows.

*The numbers on this page come from a single run on real trade data. They are not a forecast, and not financial advice.*
