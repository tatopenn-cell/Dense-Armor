# Synthetic feed: ITCH messages you can reproduce on any machine

A replay of a real Nasdaq session is useful, but a fixed session has two
drawbacks: it is tied to one day, and it does not test the edge cases a desk
worries about. A synthetic feed gives exactly the same bytes on every machine
for the same seed, so a test, a page or a colleague can reproduce it without a
file, a network, a clock or a random number generator.

## 1. Build five messages and print their types

`synthetic_itch(n, seed)` returns `n` framed messages in the container of F7.
The bytes come from SHA-256 of `(seed, block_index)`, so nothing depends on the
Python version, on NumPy or on `random`.

~~~python
import os
from pathlib import Path
from dense_armor.core.finance.synthetic import synthetic_itch
from dense_armor.core.finance.feed import iter_itch

os.makedirs("/tmp/f9_page", exist_ok=True)
Path("/tmp/f9_page/feed.bin").write_bytes(synthetic_itch(5, 0))
for m in iter_itch("/tmp/f9_page/feed.bin"):
    print(m["type"])
~~~

~~~
A
E
F
F
D
~~~

The first message is always `A`: no order is live at the start, so the rule
"a type other than A or F with no live order becomes A" fires.

## 2. The same seed gives the same bytes

Two calls with the same `(n, seed)` produce byte-identical output. A different
seed produces a different stream.

~~~python
import hashlib
from dense_armor.core.finance.synthetic import synthetic_itch

a = synthetic_itch(20, 0)
b = synthetic_itch(20, 0)
print("len:", len(a))
print("sha256 a:", hashlib.sha256(a).hexdigest()[:16] + "...")
print("sha256 b:", hashlib.sha256(b).hexdigest()[:16] + "...")
print("equal:", a == b)
c = synthetic_itch(20, 1)
print("sha256 c (seed=1):", hashlib.sha256(c).hexdigest()[:16] + "...")
print("a == c:", a == c)
~~~

~~~
len: 649
sha256 a: f900bd1d7f8c4af2...
sha256 b: f900bd1d7f8c4af2...
equal: True
sha256 c (seed=1): 9b8b683d8132856a...
a == c: False
~~~

The first 16 hex digits of the two SHA-256 hashes are the same, and `equal`
is `True`. With `seed=1` the hash is different and `a == c` is `False`.

## 3. Read 1,000 messages and count the types

The feed has the byte layout of ITCH and none of its statistics. The
distribution of the seven types is set by the byte `<` cuts in the
documentation, not by a model of a market.

~~~python
from collections import Counter
from pathlib import Path
from dense_armor.core.finance.synthetic import synthetic_itch
from dense_armor.core.finance.feed import iter_itch

Path("/tmp/f9_page/big.bin").write_bytes(synthetic_itch(1000, 0))
msgs = list(iter_itch("/tmp/f9_page/big.bin"))
print("count:", len(msgs))
counts = Counter(m["type"] for m in msgs)
for t in ("A", "F", "E", "C", "X", "D", "U"):
    print(t, counts[t])
~~~

~~~
count: 1000
A 386
F 107
E 133
C 72
X 89
D 107
U 106
~~~

## Details

**References.**
- Nasdaq. *TotalView-ITCH 5.0 specification*, 2018. Data types on
  page 4 of the PDF; Add Order `A` and `F` on page 13; Order Executed `E`
  and Order Executed With Price `C` on page 14; Order Cancel `X` and Order
  Delete `D` on page 15; Order Replace `U` on page 16.

**Choices stated explicitly.**
- The byte stream is SHA-256 of `(seed, block_index)`. A user can reproduce
  any byte of the stream without calling `synthetic_itch`, using any SHA-256
  implementation.
- The stock list `["AAPL", "MSFT", "NVDA", "AMZN"]` and the price formula
  `1_000_000 + locate * 100_000 + (2 bytes % 2001) - 1000` are ours. The
  ITCH spec says `Price (4)` is an integer with four implied decimals; it
  does not say how a synthetic price should be chosen.
- The type cuts (`< 96 A`, `< 128 F`, `< 160 E`, `< 176 C`, `< 200 X`, `< 232 D`,
  else `U`) are chosen to give all seven types a visible share in 1,000
  messages. The ITCH specification does not prescribe a distribution.
- A message type of `E`, `C`, `X`, `D` or `U` with no live order at that
  point is turned into `A`. A real feed never does that; a synthetic feed
  needs the rule to keep the stream consistent.
- The feed does not track the shares left on an order. A live order can be
  executed or cancelled by an `E`/`C`/`X` for more shares than were on the
  book, and the feed does not refuse it. This is a limit of the model, not a
  bug of the bytes.

**Declared limits.**
- The layout is real (a message decoded by `decode_message` returns the same
  keys as a real ITCH message). The statistics are not: the shares, the
  prices and the type distribution are formulas of the byte stream.
- The `locate` is drawn for every message, including `E`, `C`, `X`, `D` and
  `U`. A real feed keeps the locate of the order an update refers to; the
  synthetic feed does not, so the `locate` of an update can differ from the
  one of the `A` it updates.
- The feed is not a model of a market. There is no matching engine, no book,
  no price process, no cross. It is a source of ITCH messages with the right
  bytes.
- No order book is built and no shares are tracked. The caller has to check
  the book if it needs one.
- No real Nasdaq file was read for this page. The bytes are the output of
  `synthetic_itch`, and the check is against the table of offsets on PDF
  pages 13 to 16.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
