# Configuration: one file for the finance blocks

A desk that runs schedules, pre-trade limits and a ring buffer does not want the
same numbers typed in five places. One TOML file holds the parameters, and the
blocks read it. A configuration that loads is a configuration the blocks accept:
`load_config` builds each block once and raises the block's own error, prefixed
with the table name.

## 1. Write a configuration and load it

The file has four tables. The values are the ones a small desk would use: 100
shares split over 4 slices, a 10-share pre-trade cap, a 3-row ring, three
relative paths.

~~~python
from dense_armor.utility.finance.config import load_config

c = load_config("/tmp/finance_page/config.toml")
print("schedule:", c.schedule.total, "in", c.schedule.n, "slices")
print("pretrade max_size:", c.pretrade.max_size)
print("ring:", c.ring.capacity, "rows x", c.ring.width)
print("log name:", c.paths.log.name)
print("feed name:", c.paths.feed.name)
print("stop name:", c.paths.emergency_stop.name)
~~~

~~~
schedule: 100.0 in 4 slices
pretrade max_size: 10.0
ring: 3 rows x 2
log name: log.jsonl
feed name: feed.bin
stop name: STOP
~~~

The three paths are `pathlib.Path` objects. A relative path is resolved against
the folder of the TOML file, so the file can be moved with its data. Each table
is a frozen dataclass: assigning to a field raises
`dataclasses.FrozenInstanceError`, so a loaded configuration cannot change under
the blocks.

## 2. Build a checker and send one order

`FinanceConfig.make_checker()` returns a new `PreTradeChecker` from the
`pretrade` table. `make_ring()` does the same for the ring. The two methods do
not cache: two calls return two independent objects.

~~~python
from dense_armor.utility.finance.config import load_config

c = load_config("/tmp/finance_page/config.toml")
checker = c.make_checker()
d = checker.check("AAPL", "buy", size=5.0, price=100.0, ref_price=100.0, t=0.0)
print(d)
~~~

~~~
Decision(approved=True, reason='ok')
~~~

A 5-share buy at the reference price is inside the 10-share cap and the 1 %
band, so the checker approves it.

## 3. A configuration the blocks refuse

The structure is checked first, then the ranges. A `max_orders` of 0 is an
integer, so the structure passes, and the `PreTradeChecker` refuses it. The
message names the table.

~~~python
from dense_armor.utility.finance.config import load_config

load_config("/tmp/finance_page/bad.toml")
~~~

~~~
ValueError: [pretrade]: max_orders must be >= 1, got 0
~~~

The prefix `[pretrade]:` is the rule of `load_config`: an error of a block is
raised again with the table name in front, so a caller that reads only the
first token knows which table is wrong.

## Details

**References.**
- TOML v1.0.0 specification, `papers/toml_v1_0_0_spec.html`, sections
  [Table](https://toml.io/en/v1.0.0#table),
  [Integer](https://toml.io/en/v1.0.0#integer),
  [Float](https://toml.io/en/v1.0.0#float),
  [String](https://toml.io/en/v1.0.0#string). The Float section says `nan`
  and `inf` are valid floats; a `nan` limit is refused by the block, not by
  the parser.

**Choices stated explicitly.**
- An integer is accepted where a number is expected, and stored as `float`.
  The TOML spec has both types, and a hand-written file often writes `5`
  where the block wants `5.0`; refusing it would be a rule of ours, not of the
  spec.
- A relative path is resolved against the folder of the TOML file. The spec
  says nothing about paths, so this is our choice, stated here and tested in
  `test_relative_path_resolved_against_file_folder`.
- `load_config` builds each block once, so the ranges are validated by the
  blocks of F1, F2 and F4 and not by a second copy of their rules. The
  removal test in the delivery shows that dropping the dry run makes a test
  fail.
- `tomllib` is imported inside `load_config`, so a Python 3.10 session raises
  `RuntimeError("tomllib needs Python 3.11 or later")` only when a
  configuration is read, not at import time.

**Measured cost.** Not measured. The page reports no timing, because no script
of the repository printed a time for these functions.

**Declared limits.**
- No environment variables, no includes, no hot reload. The file is read once
  per call; a change on disk is visible only on the next call.
- One file. A configuration that spans several files is not supported.
- The reading needs Python 3.11 or later, because `tomllib` is in the standard
  library from 3.11. On Python 3.10 the call raises `RuntimeError`; the tests
  that read a TOML file are skipped there with the reason written.
- The three paths are not created by `load_config`. A log path that does not
  exist is not an error here; F6 creates the log at the first `append`.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
