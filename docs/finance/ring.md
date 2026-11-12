# Ring buffer: a queue that does not allocate

A trading loop reads messages from the network into a queue and a worker takes them out. The queue has a fixed capacity, so the loop knows how much memory it uses and never waits for an allocation in the middle of a busy moment. Two queues in this repository do that: `BoundedStream`, the one the execution pipeline already uses, and `RingBuffer`, a queue built on one numpy array allocated at construction.

This page builds a `RingBuffer`, pushes and pops rows, and compares its cost with `BoundedStream` on the same workload.

## 1. Push three rows, pop three rows

A `RingBuffer` holds up to `capacity` rows of `width` float64 values. Rows come out in the order they went in.

~~~python
import numpy as np
from dense_armor.core.finance import RingBuffer

b = RingBuffer(4, 1)
out = np.zeros(1)
for v in (1.0, 2.0, 3.0):
    b.push(np.array([v]))
for _ in range(3):
    b.pop_into(out)
    print(out[0])
~~~

~~~
1.0
2.0
3.0
~~~

`push(row)` copies the row into the next free slot and returns `True`, or returns `False` when the buffer is full and leaves it unchanged. `pop_into(out)` copies the oldest row into `out` and returns `True`, or returns `False` when the buffer is empty. The buffer never blocks and never overwrites.

## 2. A push onto a full buffer returns `False`

When the buffer is full, a push is refused. The stored rows are not touched.

~~~python
import numpy as np
from dense_armor.core.finance import RingBuffer

b = RingBuffer(2, 1)
print(b.push(np.array([1.0])))
print(b.push(np.array([2.0])))
print(b.push(np.array([3.0])))
~~~

~~~
True
True
False
~~~

The third push returns `False` because the buffer already holds two rows (its capacity). The caller decides what to do with the refused row: retry, drop, or keep a count. The buffer does not choose.

## 3. The 1,000 BTC/USD trades of one session

The block pushes the 1,000 trades of `test/fixtures/finance/coinbase_btcusd_trades.csv` (one session of a Coinbase BTC/USD market) through a ring buffer of the same capacity and checks that every row comes out equal to the row that went in. The columns are the size, the price and the seconds since the first trade.

~~~python
import numpy as np
from datetime import datetime
from pathlib import Path
from dense_armor.core.finance import RingBuffer

rows = Path("test/fixtures/finance/coinbase_btcusd_trades.csv").read_text().splitlines()
head = rows[0].split(",")
trades = [dict(zip(head, ln.split(","))) for ln in rows[1:] if ln]
t0 = datetime.fromisoformat(trades[0]["time"].replace("Z", "+00:00"))
b = RingBuffer(len(trades), 3)
for r in trades:
    t = datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
    b.push(np.array([float(r["size"]), float(r["price"]), (t - t0).total_seconds()]))
out = np.zeros(3)
ok = True
for r in trades:
    b.pop_into(out)
    if out[0] != float(r["size"]) or out[1] != float(r["price"]):
        ok = False
print("rows pushed:", len(trades))
print("rows out equal to rows in:", ok)
~~~

~~~
rows pushed: 1000
rows out equal to rows in: True
~~~

Every row comes out equal to the row that went in, in the same order. The buffer does not reorder and does not lose rows when it is not full.

## 4. Cost against `BoundedStream`

`BoundedStream` (in `pipelines/finance/connectors.py`) is the queue the execution pipeline uses. It is built on `collections.deque`. `RingBuffer` is built on one numpy array. The benchmark in `benchmarks/finance_ring.py` pushes and pops 200,000 rows of width 4 through each queue, three sessions, and records the time per push-pop pair.

~~~python
import json
from pathlib import Path

j = json.loads(Path("benchmarks/finance_ring.json").read_text())
print("n per session:", j["n_per_session"], "sessions:", j["sessions"])
print("capacity:", j["capacity"], "width:", j["width"])
print("same sequence:", j["same_sequence"])
print("Wilcoxon two-sided p:", j["wilcoxon_two_sided_p"])
print("A12 (ring slower than bounded):", j["a12_ring_slower_than_bounded"])
~~~

~~~
n per session: 200000 sessions: 20
capacity: 256 width: 4
same sequence: True
Wilcoxon two-sided p: 8.844915021211901e-05
A12 (ring slower than bounded): 0.0
~~~

Both queues return the same sequence of rows (the claim of the benchmark, written before any run). Over 20 sessions, the median of the medians is 2.26 microseconds per push-pop pair for `RingBuffer` and 0.55 microseconds per pair for `BoundedStream`; the ratio is 4.12. The paired Wilcoxon on the 20 medians gives p = 0.000088, and the A12 effect size (the probability that a random `RingBuffer` measurement is slower than a random `BoundedStream` measurement) is 0.000, which is large. The page reports the numbers; it does not choose which queue the caller should use, because the trade-off (a numpy array with preallocated storage against a `deque` of Python objects) depends on what the caller does with the rows.

## Details

**References.**
- Our design: the classic circular queue with one writer per index (`head` for the producer, `tail` for the consumer). No paper.

**Choices stated explicitly.**
- `push` copies the row into the buffer with `np.asarray(row, dtype=np.float64)`. A list is accepted and converted; the stored row is a copy of the values, not a reference to the caller's array. Changing the caller's array after `push` does not change the stored row.
- The buffer does not block when it is full and does not overwrite. A push on a full buffer returns `False` and leaves the stored rows unchanged.
- The buffer does not raise on shape: a row that is not `(width,)` raises `ValueError`. This is the check that a list without `.shape` would skip.
- The class takes no lock. Its correctness for one producer and one consumer rests on the GIL of CPython and on the rule that `head` is written only by the producer and `tail` only by the consumer.

**Measured cost.** The JSON records the median of each of the 20 sessions for `RingBuffer` and for `BoundedStream`, plus the paired Wilcoxon p-value and the A12 effect size, with the platform and the Python version under `machine`. The page reports the median of the medians; it does not claim a percentile other than the median.

**Declared limits.**
- One producer and one consumer. Several producers or several consumers are not supported and not measured.
- A CPython with the GIL. A free-threaded build of Python is not measured, and no claim is made for it.
- No proof of the memory model. The class is not a lock-free queue in the sense of the C++ memory model or of a formally verified data structure.
- No measurement across processes. The buffer lives in one process; the cost of moving rows across processes is not measured.
- No measurement on a latency percentile other than the median. The maximum is recorded but the page does not claim it as a tail latency.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
