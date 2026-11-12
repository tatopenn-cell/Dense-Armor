# Decision log: a hash chain anyone can verify

A decision log records what a desk decided and in what order. A plain log is
easy to edit after the fact: a line changes, nobody notices. This log writes
one JSON record per decision, and each record carries the SHA-256 of the
previous one. A single edited, removed or reordered record breaks the chain,
and `verify()` says so.

## 1. Append three decisions and read the file

Each line is a JSON object with five keys: `seq` (0, 1, 2, ...), `t` (a
timestamp in seconds), `payload` (the decision), `prev` (the hash of the
previous record, 64 zeros for the first) and `hash` (the SHA-256 of the
canonical text of the other four keys).

~~~python
import os
from pathlib import Path
from dense_armor.core.finance import DecisionLog
os.makedirs("/tmp/f6_page", exist_ok=True)
for f in ["log.jsonl", "log_tampered.jsonl"]:
    p = Path("/tmp/f6_page") / f
    if p.exists(): p.unlink()
log = DecisionLog("/tmp/f6_page/log.jsonl")
log.append(0.5, {"account": "AAPL", "approved": True, "reason": "ok"})
log.append(0.5, {"account": "AAPL", "approved": False, "reason": "size"})
log.append(1.0, {"account": "MSFT", "approved": True, "reason": "ok"})
print("records:", len(log))
print("last_hash:", log.last_hash[:16] + "...")
print("verify:", log.verify())
print("--- file ---")
for line in Path("/tmp/f6_page/log.jsonl").read_text().splitlines():
    print(line)
~~~

~~~
records: 3
last_hash: a32c79e2366ba73b...
verify: True
--- file ---
{"hash":"71d2427067c5dfb0225bc70074a6ee4a942844b611b44f2aeab04a8302f2292f","payload":{"account":"AAPL","approved":true,"reason":"ok"},"prev":"0000000000000000000000000000000000000000000000000000000000000000","seq":0,"t":0.5}
{"hash":"2179aeaaafc73810971d33da231845098f925abaa174b59895141637c9aaa5df","payload":{"account":"AAPL","approved":false,"reason":"size"},"prev":"71d2427067c5dfb0225bc70074a6ee4a942844b611b44f2aeab04a8302f2292f","seq":1,"t":0.5}
{"hash":"a32c79e2366ba73ba90582800b25cf5bd938aae68bc63a0da450f0d812c6a21f","payload":{"account":"MSFT","approved":true,"reason":"ok"},"prev":"2179aeaaafc73810971d33da231845098f925abaa174b59895141637c9aaa5df","seq":2,"t":1.0}
~~~

The file appears at the first `append`, not when the object is created. Each
record's `prev` is the previous record's `hash`, so changing any record and
recomputing its own `hash` still breaks the next record's `prev`.

## 2. Verify the chain

`verify()` re-reads the file from disk and recomputes every hash. It never
raises for the content of the file: a malformed line returns `False`.

~~~python
from dense_armor.core.finance import DecisionLog
log = DecisionLog("/tmp/f6_page/log.jsonl")
print("verify:", log.verify())
print("records:", len(log))
print("last_hash:", log.last_hash[:16] + "...")
~~~

~~~
verify: True
records: 3
last_hash: a32c79e2366ba73b...
~~~

Opening a `DecisionLog` on an existing file continues the chain: the next
`append` uses the last `hash` as `prev` and the next `seq`. `append` on a
chain that no longer verifies raises `ValueError` and writes nothing.

## 3. Edit one payload and see the chain break

The canonical text is JSON with sorted keys, no spaces and `allow_nan=False`.
Changing `"size"` to `"ok"` in the second record changes its payload, so its
own `hash` no longer matches and the third record's `prev` no longer matches.

~~~python
from pathlib import Path
from dense_armor.core.finance import DecisionLog
src = Path("/tmp/f6_page/log.jsonl")
lines = src.read_text().splitlines(keepends=True)
tampered = Path("/tmp/f6_page/log_tampered.jsonl")
tampered.write_text(lines[0] + lines[1].replace('"reason":"size"', '"reason":"ok"') + lines[2])
log = DecisionLog(tampered)
print("verify tampered:", log.verify())
~~~

~~~
verify tampered: False
~~~

## Details

**References.**
- NIST. FIPS 180-4, *Secure Hash Standard*. August 2015. SHA-256 is
  section 6.2 (pages 27 and 28 of the PDF); message padding is section
  5.1.1 (page 18).
- JSON Lines, jsonlines.org, version 1.0: UTF-8, one JSON value per
  line, line terminator `\n`, no byte order mark.

**Choices stated explicitly.**
- The canonical text is `json.dumps(..., sort_keys=True,
  separators=(",", ":"), ensure_ascii=True, allow_nan=False)`, encoded in
  ASCII. Without sorted keys the hash would depend on the insertion order of
  the payload dict.
- The genesis value is 64 zeros. Any other first `prev` would also be
  consistent, but 64 zeros are the same length as a real hash, so a
  structural check on `prev` is the same for the first record and the rest.
- The record is written with a single `open` / `write` / `close` per call.
  There is no `fsync`: a crash of the process between the write and the flush
  of the OS buffer could lose the last record. This is a declared limit, not
  a guarantee of durability.
- A relative path is not resolved against anything: `path` is used as given,
  so the caller decides where the log lives.

**Declared limits.**
- This is not a signature: the file has no cryptographic key and no public
  key. Anyone who can write the file can also recompute the whole chain.
- Deleting the last records of the file leaves a valid chain. `verify()`
  cannot see it. The only defence is to keep `last_hash` somewhere else; the
  test `test_8_deleting_tail_is_not_detected` makes this explicit.
- No retention policy and no legal value: the page describes what the code
  does with the file, not what a regulator accepts.
- `append` re-reads and re-hashes the whole file at every call: the cost is
  O(n) per record and O(n**2) for n records, and the demo with a large `--n`
  grows accordingly. This is declared, not optimised.
- One writer at a time. The file is opened, written and closed for each
  record, but two processes appending at the same time are not protected by a
  lock.
- Not a substitute for an execution report or an order book.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
