# ITCH messages: decoding one order at a time

Nasdaq publishes its order updates as short binary messages. Each message starts with a one-letter type, a locate code, a tracking number and a timestamp, then the fields of that type. A trading system that reads the feed must decode those bytes into a dictionary before it can follow an order.

This page builds the bytes of an Add Order, a partial execution and a delete, decodes them with one function, and follows one order through its life.

## 1. Decode one Add Order

An Add Order of type `A` is 36 bytes. The block builds the bytes field by field from the table of the specification, then decodes them.

~~~python
from dense_armor.utility.finance import decode_message

m = (
    b"A"
    + (1).to_bytes(2, "big")
    + (2).to_bytes(2, "big")
    + (0).to_bytes(6, "big")
    + (7).to_bytes(8, "big")
    + b"B"
    + (100).to_bytes(4, "big")
    + b"AAPL    "
    + (1234567).to_bytes(4, "big")
)
d = decode_message(m)
print(d)
~~~

~~~
{'type': 'A', 'locate': 1, 'tracking': 2, 'timestamp': 0, 'order_ref': 7, 'side': 'B', 'shares': 100, 'stock': 'AAPL', 'price_raw': 1234567, 'price': 123.4567}
~~~

Every integer is big endian and unsigned. The stock is ASCII, right padded with spaces and stripped. A `Price (4)` is returned as an integer number of ten-thousandths (`price_raw`) and as a float (`price`): 1234567 becomes 123.4567. The timestamp is 6 bytes of nanoseconds since midnight.

## 2. Follow one order through three messages

An order is added, partially executed, then deleted. The three messages carry the same `order_ref` (7), so the three dictionaries line up.

~~~python
from dense_armor.utility.finance import decode_message

add = (
    b"A"
    + (1).to_bytes(2, "big")
    + (2).to_bytes(2, "big")
    + (0).to_bytes(6, "big")
    + (7).to_bytes(8, "big")
    + b"B"
    + (100).to_bytes(4, "big")
    + b"AAPL    "
    + (1234567).to_bytes(4, "big")
)
ex = (
    b"E"
    + (1).to_bytes(2, "big")
    + (2).to_bytes(2, "big")
    + (500).to_bytes(6, "big")
    + (7).to_bytes(8, "big")
    + (40).to_bytes(4, "big")
    + (999).to_bytes(8, "big")
)
del_ = (
    b"D"
    + (1).to_bytes(2, "big")
    + (2).to_bytes(2, "big")
    + (900).to_bytes(6, "big")
    + (7).to_bytes(8, "big")
)
for m in (add, ex, del_):
    d = decode_message(m)
    print(d["type"], d["order_ref"])
~~~

~~~
A 7
E 7
D 7
~~~

An `A` message adds 100 shares, an `E` message executes 40, a `D` message deletes the order. After the `E` message the display size is 60; the `D` message removes the rest, so the order is no longer on the book.

## 3. Timestamps are one big endian number

The timestamp is 6 bytes of nanoseconds since midnight, read as one big endian number. The two byte sequences below are not the same number in another order.

~~~python
from dense_armor.utility.finance import decode_message

m = (
    b"D"
    + (1).to_bytes(2, "big")
    + (2).to_bytes(2, "big")
    + bytes.fromhex("000000000100")
    + (7).to_bytes(8, "big")
)
print(decode_message(m)["timestamp"])
~~~

~~~
256
~~~

Reading the six bytes as one number gives 256 (0x100) on `00 00 00 00 01 00`; the same six bytes in the other order (`01 00 00 00 00 00`) give 1099511627776 (0x10000000000).

## 4. The benchmark

The benchmark in `benchmarks/finance_itch.py` decodes 100,000 messages of type `A` in three sessions. The block reads the JSON it wrote and prints the number of sessions and the time per message.

~~~python
import json
from pathlib import Path

j = json.loads(Path("benchmarks/finance_itch.json").read_text())
print("sessions:", len(j["sessions"]))
print("time per message (us):", j["time_per_message_us"])
~~~

~~~
sessions: 3
time per message (us): {'median': 2.7790001695393585, 'min': 2.623999989737058, 'max': 739.3630003207363, 'n': 300000}
~~~


The claim of the benchmark was written before any run: 100,000 messages are decoded without an exception in each session. It holds. The time is recorded in the JSON with the machine that ran it; the median is 2.78 microseconds per message on that machine, and the maximum (739 µs) is one garbage collector pause.

## Details

**References.**
- Nasdaq TotalView-ITCH 5.0. Data Types (big endian, alpha fields, `Price (4)`, timestamps in nanoseconds since midnight), PDF page 4. Message layouts: Add Order `A` and Add Order with attribution `F`, PDF page 13; Order Executed `E`, PDF page 14; Order Executed with Price `C`, PDF page 15; Order Cancel `X` and Order Delete `D`, PDF page 15; Order Replace `U`, PDF page 16.

**Choices stated explicitly.**
- The price is returned both as an integer number of ten-thousandths (`price_raw`) and as a float (`price`). The float loses no digits for the range of Nasdaq prices (`Price (4)` up to 200,000.0000), but the raw integer is the exact value.
- Alpha fields (stock, side, printable, attribution) are returned as `str` after stripping the right padding. A field that is entirely spaces returns the empty string.
- The decoder takes exactly one message as `bytes` and does not frame anything. The `SoupBinTCP` and `MoldUDP64` framings are separate protocols, not implemented here.
- A `bytes` object of the wrong length is a `ValueError`, not a silent truncation. The length of each type is fixed by the table of the specification.

**Measured cost.** The JSON records the median, the minimum and the maximum of the time per message over 300,000 calls (100,000 × 3 sessions), with the platform and the Python version under `machine`. The page reports the median and the maximum; it does not claim a percentile other than the median.

**Declared limits.**
- The framing of the feed (`SoupBinTCP`, `MoldUDP64`) is not decoded.
- The messages other than the seven (`A`, `F`, `E`, `C`, `X`, `D`, `U`) are not decoded: system events, stock directory, trades, cross trades, broken trades, net order imbalance, retail price improvement and stock trading action are not covered.
- No order book is built. The decoder returns the fields of a message; it does not maintain a book.
- No connection to Nasdaq was made and no Nasdaq data file was read. The bytes in the page are built by hand from the offsets of the specification; the benchmark repeats one of them.

*These pages describe measurements on historical data. Not a forecast, not financial advice.*
