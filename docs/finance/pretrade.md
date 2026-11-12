# Pre-trade limits: the five controls before an order leaves

Before an order leaves the desk it passes five checks. Each one answers a specific question: is it too big, is the price too far from the reference, is it a duplicate, has the account sent too many orders in a short time, is the open notional above the credit line. The five checks are the ones that paragraph (c)(1) of SEC Rule 15c3-5 describes.

This page builds the checker, shows one order that each check refuses, and runs the checker on 1,000 BTC/USD trades from a session of the Coinbase BTC/USD market.

## 1. Build the checker and send one order

The checker is configured with five limits, and one order is sent.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=50_000.0, max_size=0.2, price_band=0.001,
    max_orders=3, window_s=10.0, duplicate_window_s=1.0,
)
print(c.check("acct", "buy", 0.1, 83_000.0, 83_000.0, t=0.0))
~~~

~~~
Decision(approved=True, reason='ok')
~~~

The `check` returns a `Decision`, a frozen dataclass with `approved` and `reason`. An approved order adds `size * price` to the open notional of the account and appends to the rate and duplicate lists. A rejected order changes nothing.

## 2. One order for each reason

Each of the five controls is shown with a single order that triggers it. The other limits are loosened so that only the reason under test can fire.

**Size.** `size > max_size`.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=1e9, max_size=0.2, price_band=1.0,
    max_orders=100, window_s=10.0, duplicate_window_s=0.001,
)
print(c.check("a", "buy", 0.5, 83_000.0, 83_000.0, 0.0))
~~~

~~~
Decision(approved=False, reason='size')
~~~

**Price band.** `abs(price / ref_price - 1) > price_band`.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=1e9, max_size=1.0, price_band=0.001,
    max_orders=100, window_s=10.0, duplicate_window_s=0.001,
)
print(c.check("a", "buy", 0.1, 83_500.0, 83_000.0, 0.0))
~~~

~~~
Decision(approved=False, reason='price_band')
~~~

**Duplicate.** An approved order with the same account, side, size and price in the last `duplicate_window_s` seconds.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=1e9, max_size=1.0, price_band=1.0,
    max_orders=100, window_s=10.0, duplicate_window_s=1.0,
)
print(c.check("a", "buy", 0.1, 83_000.0, 83_000.0, 0.0))
print(c.check("a", "buy", 0.1, 83_000.0, 83_000.0, 0.5))
~~~

~~~
Decision(approved=True, reason='ok')
Decision(approved=False, reason='duplicate')
~~~

**Rate.** `max_orders` or more approved orders of the account in the last `window_s` seconds.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=1e9, max_size=1.0, price_band=1.0,
    max_orders=1, window_s=10.0, duplicate_window_s=0.001,
)
print(c.check("a", "buy", 0.1, 83_000.0, 83_000.0, 0.0))
print(c.check("a", "buy", 0.1, 83_000.0, 83_000.0, 0.5))
~~~

~~~
Decision(approved=True, reason='ok')
Decision(approved=False, reason='rate')
~~~

**Notional.** The open notional of the account plus `size * price` above `max_notional`.

~~~python
from dense_armor.core.finance import PreTradeChecker

c = PreTradeChecker(
    max_notional=10_000.0, max_size=1.0, price_band=1.0,
    max_orders=100, window_s=10.0, duplicate_window_s=0.001,
)
print(c.check("a", "buy", 0.2, 83_000.0, 83_000.0, 0.0))
~~~

~~~
Decision(approved=False, reason='notional')
~~~

A value exactly at a limit is approved; one tick above is rejected. The order of the checks is fixed: the first one that fails gives the reason, and an order that breaks two limits reports only the first.

## 3. The 1,000 BTC/USD trades of one session

The benchmark in `benchmarks/finance_pretrade.py` runs the checker on the 1,000 trades of `test/fixtures/finance/coinbase_btcusd_trades.csv`, each trade as an order with `ref_price` the price of the previous trade, three sessions shifted in time so the rate and duplicate windows do not carry over. The block reads the JSON the script wrote and prints the count of decisions per reason in each session.

~~~python
import json
from pathlib import Path

j = json.loads(Path("benchmarks/finance_pretrade.json").read_text())
print("sessions:")
for s in j["sessions"]:
    print(" ", s)
~~~

~~~
sessions:
  {'duplicate': 57, 'ok': 500, 'rate': 424, 'size': 19}
  {'duplicate': 57, 'ok': 500, 'rate': 424, 'size': 19}
  {'duplicate': 57, 'ok': 500, 'rate': 424, 'size': 19}
~~~

The three sessions give the same counts. The time per call is recorded in the JSON; the median on the machine that ran the benchmark is 30.23 microseconds, with a minimum of 5.27 and a maximum of 486.36 over the 3,000 calls.

## Details

**References.**
- SEC Release 34-63241, Rule 15c3-5 (c)(1)(i): pre-set credit or capital thresholds, in the aggregate for each customer, PDF page 128 (label 128 on the page).
- SEC Release 34-63241, Rule 15c3-5 (c)(1)(ii): price or size parameters, on an order-by-order basis or over a short period of time, duplicative orders, PDF page 128 (label 128 on the page).

**Choices stated explicitly.**
- The order of the five controls (size, price band, duplicate, rate, notional) is our choice. The rule lists the two categories of controls in (c)(1)(i) and (c)(1)(ii) but does not impose an order between them.
- Only approved orders count for the rate window and for the duplicate window. The rule says "prevent the entry of erroneous orders" and does not say whether a rejected order should count. We chose no, so a spike of rejected orders does not lock the account.
- The duplicate key is `(account, side, size, price)` inside `duplicate_window_s`. The rule names "duplicative orders" without a key; a different key is a different block.
- `release` subtracts a notional after a fill or a cancel and raises `ValueError` if the release is larger than the open notional. The rule does not name a release step; it is the caller's job to call it.

**Measured cost.** The JSON records the median, the minimum and the maximum of the time per `check` call over 3,000 calls, on the machine that ran the benchmark (platform and Python version are in the JSON under `machine`). The page does not claim a percentile other than the median.

**Parameters of the benchmark.** The limits in `benchmarks/finance_pretrade.py` (`max_notional=200_000.0`, `max_size=0.1`, `price_band=0.0005`, `max_orders=500`) were set after a first run with stricter values, which rejected 872 of the 1,000 orders for notional. The first run confirmed that the script works; the parameters were then chosen to produce a mix of all five reasons on the same fixture. The claim of the script ("the counts per reason are identical in the 3 sessions") holds with the values that are in the code, and is the only claim on the numbers.

**Declared limits.**
- It is not a compliance system and does not replace the controls of a broker.
- Paragraph (c)(2) of the rule (regulatory controls) and the restricted lists are not implemented.
- The limits are set by the caller; the class does not choose them.
- One process, no shared state across processes. The five controls are in memory.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
