# Schedules: TWAP and VWAP for a parent order

A trader who has to sell a large block of shares does not send one order: the order would eat the book and move the price against the seller. The trade is split into slices over time. Two schedules are common: **TWAP** (time weighted average price) and **VWAP** (volume weighted average price). The first spreads the parent order equally in time; the second weights each slice by the volume the market traded in the same period in the past.

## 1. Equal slices in time (TWAP)

The simplest schedule splits the parent order into `n` equal slices. If the parent order is 3 shares over 3 buckets, each bucket gets 1 share.

~~~python
from dense_armor.utility.finance import twap_schedule

print(repr(twap_schedule(3.0, 3)))
~~~

~~~
array([1., 1., 1.])
~~~

The function returns a float64 array whose sum is the parent order (the last slice absorbs the floating point residue). This is the optimal static strategy when the traded volume is constant: it sells at constant speed, `total / n` per bucket (Remark 3, PDF page 5 of Kato 2014).

## 2. Weight by past volume (VWAP)

Markets do not trade at constant volume: the open and the close are busier than the middle of the session. The **VWAP** schedule weights each bucket by the volume the market traded in that bucket in the past sessions, and it uses the **harmonic mean** of those volumes, not the arithmetic mean.

~~~python
from dense_armor.utility.finance import vwap_schedule

h = [[1.0, 2.0], [4.0, 2.0]]
print([round(float(x), 4) for x in vwap_schedule(3.6, h)])
~~~

~~~
[1.6, 2.0]
~~~

Bucket 1 gets `1 / mean(1/1, 1/4) = 1.6`; bucket 2 gets `1 / mean(1/2, 1/2) = 2.0`. The slices sum to 3.6, the parent order (the sell-off condition, equation (2.3), PDF page 2 of Kato 2014). When the history is constant across buckets, the harmonic mean equals the constant volume and the VWAP schedule equals the TWAP schedule (Remark 3, PDF page 5).

## 3. One session of the Coinbase BTC/USD fixture

The fixture `test/fixtures/finance/coinbase_btcusd_trades.csv` has 1,000 BTC/USD trades from a short session of the Coinbase BTC/USD market. The block sorts the trades by time, splits the session into 10 consecutive buckets of equal duration, sums the traded size per bucket, and prints the TWAP and VWAP schedules of 1.0 BTC.

~~~python
import numpy as np
from datetime import datetime
from pathlib import Path
from dense_armor.utility.finance import twap_schedule, vwap_schedule

rows = Path("test/fixtures/finance/coinbase_btcusd_trades.csv").read_text().splitlines()
head = rows[0].split(",")
trades = [dict(zip(head, ln.split(","))) for ln in rows[1:] if ln]
times = [datetime.fromisoformat(t["time"].replace("Z", "+00:00")) for t in trades]
sizes = np.array([float(t["size"]) for t in trades])
start, end = min(times), max(times)
width = (end - start).total_seconds() / 10
volume = np.zeros(10)
for ti, si in zip(times, sizes):
    k = min(int((ti - start).total_seconds() / width), 9)
    volume[k] += si
history = volume.reshape(1, 10)
tw = twap_schedule(1.0, 10)
vw = vwap_schedule(1.0, history)
print("volume:", [round(float(v), 4) for v in volume])
print("twap:  ", [round(float(v), 4) for v in tw])
print("vwap:  ", [round(float(v), 4) for v in vw])
~~~

~~~
volume: [0.6066, 2.713, 0.2406, 0.5194, 0.5931, 0.1485, 1.145, 0.4535, 1.0825, 1.2485]
twap:   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
vwap:   [0.0693, 0.31, 0.0275, 0.0594, 0.0678, 0.017, 0.1308, 0.0518, 0.1237, 0.1427]
~~~

The fixture is one session, so the history has one row and the harmonic mean of one value is that value: the weights are the volume of the session itself. The TWAP schedule is flat at `0.1` BTC per bucket; the VWAP schedule gives the busy buckets (bucket 2 at 2.713 BTC and bucket 7 at 1.145 BTC) more shares and the quiet ones (bucket 6 at 0.1485 BTC) fewer.

## Details

**References.**
- Kato, T. (2014). *An optimal execution problem with market impact and stochastic volume.* Sell-off condition equation (2.3), PDF page 2; Definition 1 (VWAP strategy), PDF page 3; temporary market impact equation (3.2), PDF page 3; static strategy `A_stat`, PDF page 4; Theorem 2 and Remark 3, PDF page 5.
- Almgren, R. and Chriss, N. (2001). *Optimal execution of portfolio transactions.* Journal of Risk 3(2), 5-39. Not read: the PDF was not available in the session.

**Choices stated explicitly.**
- The harmonic mean is the expectation `u_t = 1 / E[1 / v_t]` of Theorem 2, computed from the past sessions of the input array. The paper writes it as an expectation; the function takes a finite sample and uses its mean.
- The last slice of `twap_schedule` absorbs the floating point residue, so the array sums to `total` exactly. A plain `total / n` per slice would lose the last digit for values that are not representable in binary.
- The bucket boundaries in step 3 are defined by `(t - t_start) / width` with `t_start` the first trade and `width = (t_end - t_start) / 10`; a trade at the right edge of the session falls in the last bucket.

**Measured cost.** Not measured. The page reports no timing, because no script of the repository printed a time for these functions.

**Declared limits.**
- The optimality of the schedules rests on the model of Theorem 2: temporary impact `kappa_tilde * zeta / v` (equation (3.2)), static strategy, risk-neutral trader, and an unaffected price independent of the volume process. A trader who is not risk-neutral, or who uses an adaptive strategy, has a different optimum.
- The cumulative volume of the day is not observable in advance: the paper's Theorem 2 says the optimal static strategy weights the buckets by the expected harmonic mean of the volume, and the expected value is estimated from the past sessions. The fixture is one session, so the estimate in step 3 is the volume of that session itself, not an estimate from several sessions.
- No market impact was measured. The page compares two schedules on one session; it does not claim that either one is faster or cheaper to execute.
- No execution was simulated. The schedules are vectors of sizes, not orders sent to a venue.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
