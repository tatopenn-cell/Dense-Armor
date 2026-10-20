# Streaming: what happens when the feed is faster than the pipeline?

A market feed can deliver more ticks per second than the pipeline consumes. The queue between the producer and the consumer has a fixed size, and when it is full something must happen: the oldest item is dropped, the new one is dropped, or the producer waits. This page gives the queue with the three policies and the counters of what it did, and two adapters to a broker.

The adapters are **written, not run against a broker**: no broker is in this environment, and the tests use a client injected by the test. The page says it and does not claim a broker test.

## 1. The three policies

```python
import json
from pathlib import Path

rows = json.loads(Path("benchmarks/finance/streaming.json").read_text())["rows"]
print([(r["policy"], r["consumed"], r["dropped_oldest"],
        r["dropped_newest"], r["blocked"]) for r in rows])
```

```
[('drop_oldest', 500, 450, 0, 0), ('drop_newest', 500, 0, 450, 0), ('block', 500, 0, 0, 450)]
```

The measurement is a queue of 50 items, 1,000 pushes and one pop every two pushes, so the producer is twice as fast as the consumer. With `drop_oldest` the queue drops the oldest item 450 times and the consumer sees 500 items; with `drop_newest` it drops the newest 450 times and the consumer sees 500; with `block` the producer is refused 450 times and the queue stays full. The counters are exact: 1,000 pushes, 500 consumed, 450 dropped or blocked.

## 2. The queue, one push at a time

```python
from pipelines.finance.connectors import BoundedStream

s = BoundedStream(capacity=2, policy="drop_oldest")
for v in (1, 2, 3):
    s.push(v)
print(len(s), s.counters.dropped_oldest, s.pop(), s.pop())
```

```
2 1 2 3
```

The queue holds two items; the third push drops the oldest (1) and keeps (2, 3). The counters count the drop, and the two pops return what survived.

## 3. The adapters

```python
from pipelines.finance.connectors import ZmqSubscriber

class FakeClient:
    def __init__(self):
        self.calls = []
    def publish(self, topic, payload):
        self.calls.append((topic, payload))

c = FakeClient()
ZmqSubscriber(client=c, topic="ticks").publish(1.5)
print(c.calls)
```

```
[('ticks', 1.5)]
```

The adapter takes a client with a `publish(topic, payload)` method. When the client is `None`, it imports `zmq` (or `confluent_kafka`) lazily, and a missing package gives an `ImportError` that names the `finance-stream` extra. The test covers both branches with a fake client and with a fake module in `sys.modules`.

## Details

### References

- No paper in the folder is cited for this page. `ISTRUZIONI.md` cites none for item F, and the folder has no paper on queues, backpressure or load shedding. The three policies and the counters are this implementation's choice, and the page says so.

### Choices stated explicitly

- The three policies (`drop_oldest`, `drop_newest`, `block`) and the five counters are this implementation's choice. A real broker client has its own backpressure; the queue here is the one between the feed and the pipeline, and the page does not claim that a broker behaves this way.
- The adapters are written and not run against a broker. No broker is in this environment, and the tests inject a client. The page does not report a broker latency or a broker test.
- The measurement uses a queue of 50 items, 1,000 pushes and a 2:1 producer-to-consumer ratio. The numbers are the ones of `benchmarks/finance/streaming.json`, and the page prints the counters, not a latency.

### Measured cost

The measurement of section 1 takes 0.01 s on this host. The queue is a `collections.deque`, and a push or a pop is O(1).

### Development diary

The first version of the test for the missing package used `pytest.importorskip("zmq")`, which skips the test when `zmq` is installed: the `ImportError` branch was never covered and the coverage stayed at 98%. The test now forces a package name that does not exist, and a second test uses a fake module in `sys.modules` to cover the branch where the import succeeds. The two branches are covered and the coverage is 100%.
