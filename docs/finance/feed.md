# Feed file: read Nasdaq ITCH messages one by one

Nasdaq's TotalView-ITCH feed is a stream of small binary order messages: an
`A` adds an order, a `D` deletes it, an `E` says it executed. A file of these
messages is useful to replay a session, test a checker, or hand a fixed set of
messages to a colleague. This module reads such a file, one record at a time,
and decodes each message with the `decode_message` of the ITCH block.

## 1. Frame two messages

A record in the file is a 2-byte big-endian length followed by the bytes of one
message. This container is our choice, not part of the ITCH specification: the
standard describes the bytes of one message, not how a file of them is
delimited.

~~~python
import os
from pathlib import Path
from dense_armor.utility.finance.feed import frame

os.makedirs("/tmp/f7_page", exist_ok=True)

def _A(order_ref, side, shares, stock, price_raw):
    b = bytearray(36)
    b[0] = ord("A")
    b[1:3] = (1).to_bytes(2, "big")
    b[3:5] = (2).to_bytes(2, "big")
    b[5:11] = (3).to_bytes(6, "big")
    b[11:19] = order_ref.to_bytes(8, "big")
    b[19] = ord(side)
    b[20:24] = shares.to_bytes(4, "big")
    b[24:32] = stock.encode("ascii").ljust(8)
    b[32:36] = price_raw.to_bytes(4, "big")
    return bytes(b)

def _D(order_ref):
    b = bytearray(19)
    b[0] = ord("D")
    b[1:3] = (1).to_bytes(2, "big")
    b[3:5] = (2).to_bytes(2, "big")
    b[5:11] = (3).to_bytes(6, "big")
    b[11:19] = order_ref.to_bytes(8, "big")
    return bytes(b)

a = _A(11, "B", 100, "AAPL", 1234567)
d = _D(11)
Path("/tmp/f7_page/feed.bin").write_bytes(frame(a) + frame(d))
print("A bytes:", len(a))
print("D bytes:", len(d))
print("A order_ref:", int.from_bytes(a[11:19], "big"))
print("A side:", chr(a[19]))
print("A shares:", int.from_bytes(a[20:24], "big"))
print("A stock:", a[24:32].decode("ascii").rstrip())
print("A price:", int.from_bytes(a[32:36], "big") / 10000.0)
~~~

~~~
A bytes: 36
D bytes: 19
A order_ref: 11
A side: B
A shares: 100
A stock: AAPL
A price: 123.4567
~~~

The `A` message is 36 bytes (offset 0 to 35), the `D` message is 19 bytes
(offset 0 to 18). `frame` adds two bytes in front of each, so the file is 59
bytes long in total.

## 2. Read the messages back

`iter_itch` is a generator: it opens the file at the first `next()`, reads one
record at a time (`read(2)` for the length, then `read(L)` for the body), and
closes the file at the end or when the generator is closed.

~~~python
from dense_armor.utility.finance.feed import iter_itch
msgs = list(iter_itch("/tmp/f7_page/feed.bin"))
print("count:", len(msgs))
for m in msgs:
    print(m["type"], m["order_ref"], m.get("side"), m.get("shares"), m.get("stock"), m.get("price"))
~~~

~~~
count: 2
A 11 B 100 AAPL 123.4567
D 11 None None None None
~~~

Each message is the dict returned by `decode_message`: `type` first, then one
key per field of the layout on PDF pages 13 to 16. A field that does not
belong to that type is not in the dict, so `m.get(...)` is `None` for a `D`.

## 3. A truncated file stops at the bad record

The length prefix says how many bytes the body must have. If the file ends
before the body, `iter_itch` raises `ValueError` with the byte offset of the
start of the record and how many messages were already yielded. This is not a
crash: the messages before the bad record have already been yielded, and the
error names the exact place.

~~~python
from pathlib import Path
from dense_armor.utility.finance.feed import iter_itch
src = Path("/tmp/f7_page/feed.bin").read_bytes()
truncated = src[:-10]
Path("/tmp/f7_page/bad.bin").write_bytes(truncated)
try:
    list(iter_itch("/tmp/f7_page/bad.bin"))
except ValueError as e:
    print("ValueError:", e)
~~~

~~~
ValueError: offset 38: body is 9 bytes, expected 19 (after 1 messages)
~~~

`offset 38` is 36 + 2 (the `A` message plus its length prefix); `body is 9
bytes` is what is left after the `D`'s length prefix; `after 1 messages` is
the count of messages that `iter_itch` already yielded.

## Details

**References.**
- Nasdaq. *TotalView-ITCH 5.0 specification*,
  `papers/nasdaq_totalview_itch_50.pdf`. Data types on PDF page 4; Add Order
  `A` and `F` on PDF page 13; Order Executed `E` and Order Executed With Price
  `C` on PDF page 14; Order Cancel `X` and Order Delete `D` on PDF page 15;
  Order Replace `U` on PDF page 16.
- The container (a 2-byte length in front of every message) is **our
  choice**, stated in `CONTESTO.md` and in this page. It is not in the ITCH
  specification: the standard describes the bytes of a message, not a file.

**Choices stated explicitly.**
- The length prefix is 2 bytes, big endian, unsigned. A message longer than
  65,535 bytes is refused by `frame`, so a file the reader accepts always has
  a length that fits in 2 bytes.
- A prefix of one byte (a file that ends after one byte) is refused with
  `offset <n>: length prefix is 1 byte, not 2`. The reader does not treat it
  as "end of file": an unfinished record is an error.
- A body shorter than its length is refused with `offset <n>: body is X
  bytes, expected Y`. A longer body is not read as one message: the reader
  reads exactly `L` bytes and the next prefix starts there.
- A length of 0 is refused with `offset <n>: record has length 0`: the
  ITCH spec has no zero-length message.
- `iter_itch` is lazy: it does not read the whole file into memory, and the
  error for a bad record comes after the good records before it were yielded.
- The file is closed with `with`, so a `KeyboardInterrupt` mid-read leaves
  the file closed.

**Declared limits.**
- The container is ours. A file of Nasdaq's own feed (SoupBinTCP or
  MoldUDP64 framing) does not have a 2-byte length in front of each
  message, and `iter_itch` cannot read it. This is a declared limit, not
  a bug: the framing of the real feed is a separate protocol.
- Only the seven order messages (`A F E C X D U`) are decoded. System
  events, trade messages, cross trades and stock directory messages are
  not.
- No order book is built. The reader produces one dict per message, in the
  order of the file. A book would need to track which orders are alive,
  which is the job of a different module.
- No real Nasdaq file was read for this page. The bytes in the examples are
  built by hand from the table of offsets on PDF pages 13 to 16. The feed is
  ours and the check is the table.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
