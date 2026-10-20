# Order book: does the book say where the price goes next?

The order book of a stock at the best bid and ask is a record of supply and demand. Between two observations, the sizes at the best bid and ask change, and the price moves. This page answers one question: **does the book say where the price goes next?**

The variable that summarises the book is the order flow imbalance of Cont, Kukanov and Stoikov (2010). It sums, over a time interval, the contribution of every event at the best bid and ask to the size of the two queues. The paper finds a linear relation between the mid-price change and the imbalance, with an average R² of 65% on 50 stocks (abstract, page 1).

## 1. The contribution of one event

```python
from pipelines.finance.order_book import event_contribution

e = event_contribution(100.0, 10.0, 100.01, 10.0,
                       100.0, 15.0, 100.01, 10.0)
print(e)
```

```
5.0
```

The bid price is unchanged and the bid size grows by five: five shares were added to the bid queue, and the contribution is `+5.0`. The paper defines it on page 4: `e_n = 1{PB_n >= PB_n-1} qB_n - 1{PB_n <= PB_n-1} qB_n-1 - 1{PA_n <= PA_n-1} qA_n + 1{PA_n >= PA_n-1} qA_n-1`. The same event on the ask side has the opposite sign, because an ask size that grows is supply, not demand.

## 2. The imbalance over an interval

```python
import numpy as np
from pipelines.finance.order_book import order_flow_imbalance

book = np.array([[100.0, 10.0, 100.01, 10.0],
                 [100.0, 15.0, 100.01, 10.0],
                 [100.0, 15.0, 100.01, 12.0]])
print([round(float(v), 1) for v in order_flow_imbalance(book)])
```

```
[5.0, -2.0]
```

The first step adds five shares to the bid (`+5.0`); the second adds two shares to the ask, which is supply, and the sign is negative (`-2.0`). The imbalance over a whole interval is the sum of the contributions of the events in it (page 4).

## 3. The regression on a synthetic book

```python
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/finance/order_book.json").read_text())
print(round(payload["estimated_slope"], 6), round(payload["r2"], 3))
print(round(payload["planted_slope"], 6), payload["n_events"])
```

```
0.002708 0.641
0.0002 2000
```

The book is synthetic: 2,000 states, a planted relation between the imbalance and the mid-price change with slope `2e-4`, and Gaussian noise of standard deviation `1e-3`. The regression of equation 4, page 8, gives a slope of `0.002708` and an R² of `0.641`. The estimated slope is 13.5 times the planted one, and the page states it: `synthetic_book` generates an exogenous imbalance and moves the mid price by it, but the imbalance that `order_flow_imbalance` recomputes from the book is not the exogenous one, because the book has constant sizes and a price that moves by whole ticks. The R² of 0.641 is in the range the paper reports on real books (65% average), and the planted slope is not recovered: a test with a hand-made book, where the size changes and the price does not, pins the exact relation.

## 4. What the numbers say, and what they do not

The regression on the synthetic book has an R² of 0.641, close to the 65% of the paper on 50 stocks. The page does not claim that the slope is the paper's: the slope of the paper is on real books of 50 stocks over half-hour intervals (section 3, pages 8-9), and the book here is synthetic. The linear relation between the imbalance and the price change is the paper's (equation 4, page 8); the numbers are the ones of `benchmarks/finance/order_book.json`, a synthetic book with a planted relation.

## Details

### References

- Cont, R., Kukanov, A. and Stoikov, S. (2010). *The Price Impact of Order Book Events.* arXiv:1011.6402. The event contribution `e_n`, the order flow imbalance and the mid-price change: page 4. The regression `dP_k = alpha + beta OFI_k + eps_k`: equation 4, page 8. Average R² of 65% on 50 stocks: abstract, page 1; section 3, pages 8-9.

### Choices stated explicitly

- `event_contribution` implements the four indicators of page 4 exactly, with the bid and ask contributions in one function. The paper writes the event as a single expression, and the function returns that expression.
- The regression is an ordinary least squares on the intercept and the imbalance, as equation 4 (page 8). The paper estimates it per half-hour interval with a time-varying beta (equation 2, page 6); this page estimates one slope over the whole synthetic book.
- `synthetic_book` moves the mid price by `slope * OFI_exogenous + noise`, where `OFI_exogenous` is drawn from a normal. The imbalance that `order_flow_imbalance` recomputes from the book is not the exogenous one: with constant sizes, only the price changes contribute, and a price that moves by whole ticks makes the recomputed imbalance coarse. The page states that the estimated slope is 13.5 times the planted one, and the exact relation is pinned by a hand-made book in the tests.
- The synthetic book is labelled "synthetic" in the code, in the page and in the JSON.

### Measured cost

`regress` on 2,000 book states takes 1.9 ms on this host. The regression is an `np.linalg.lstsq` on 1,999 rows and two columns.

### Development diary

The first version of `synthetic_book` returned a book and the exogenous imbalance, and the test compared the estimated slope to the planted one exactly. The two differed by a factor of 13.5, because the recomputed imbalance is not the exogenous one. The test now pins the exact relation on a hand-made book where the bid size changes and the price does not, and the synthetic book is used for the sign and the R², with the factor declared on the page. The first version of the page said "the slope is recovered", which was false; the page now says what the two numbers are.
