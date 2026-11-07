# Demo: a feed, a checker, a log, in one command

The five blocks of F5 to F9 (a configuration, a decision log, a feed reader, a
status command, a synthetic feed) are useful one at a time. A desk wants them
in one command: write a feed, send the orders through the pre-trade checker,
record every decision in the log, print one JSON line with what happened.

## 1. Run the demo on the fixture

The command takes `--config`, `--n` (default 1000) and `--seed` (default 0).
The fixture is `test/fixtures/finance/demo.toml`: it points at three relative
paths in the folder of the configuration, and its limits are chosen so that the
1,000-message run of seed 0 shows at least three different reasons.

~~~python
import json, os, subprocess, sys
from pathlib import Path
work = Path("/tmp/f10_page")
work.mkdir(parents=True, exist_ok=True)
for f in ("log.jsonl", "feed.bin", "STOP"):
    (work / f).unlink(missing_ok=True)
import shutil
shutil.copy("/content/Dense-Armor/test/fixtures/finance/demo.toml", work / "demo.toml")
env = os.environ.copy()
env["PYTHONPATH"] = "/content/Dense-Armor"
r = subprocess.run(
    [sys.executable, "-m", "dense_armor.utility.finance.demo",
     "--config", "demo.toml", "--n", "1000"],
    capture_output=True, text=True, cwd=str(work), env=env,
)
payload = json.loads(r.stdout)
payload["elapsed_s"] = "<float>"
print("exit:", r.returncode)
print("stdout:", json.dumps(payload, sort_keys=True))
~~~

~~~
exit: 0
stdout: {"counts": {"notional": 116, "ok": 32, "size": 345}, "elapsed_s": "<float>", "log": {"last_hash": "5573286aec2926ff070501e3092c25f6a22db2e638db7a47b553fdaa2343f482", "records": 493, "verified": true}, "n_messages": 1000, "n_orders": 493, "ok": true, "python": "3.13.18", "schedule_sum": 100.0}
~~~

`counts` has three reasons (`notional`, `ok`, `size`); the sum is 493, which is
`n_orders`. `n_messages` is 1,000. `verified` is `true`: the log written by the
run is a valid chain. `schedule_sum` is 100.0, the total of the schedule.
`elapsed_s` is one run of this page, on Python 3.13.18 on Linux, in a Colab session. It is not a measurement of speed.

## 2. Verify the log with the status command

`status` of F8 reads the configuration and reports the log: records, last hash,
and `verify()`.

~~~python
import os, subprocess, sys
from pathlib import Path
work = Path("/tmp/f10_page")
env = os.environ.copy()
env["PYTHONPATH"] = "/content/Dense-Armor"
r = subprocess.run(
    [sys.executable, "-m", "dense_armor.utility.finance.status",
     "--config", "demo.toml"],
    capture_output=True, text=True, cwd=str(work), env=env,
)
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "demo.toml", "emergency_stop": false, "feed": {"exists": true, "path": "/tmp/f10_page/feed.bin"}, "log": {"exists": true, "last_hash": "5573286aec2926ff070501e3092c25f6a22db2e638db7a47b553fdaa2343f482", "path": "/tmp/f10_page/log.jsonl", "records": 493, "verified": true}, "ok": true}
~~~

The `last_hash` is the same string as the demo printed. `records` is 493, the
number of records the demo appended. `verified` is `true`.

## 3. Break one record and run the status command again

The log is a hash chain (F6). Edit one payload with plain file operations and
the chain breaks.

~~~python
import os, subprocess, sys
from pathlib import Path
work = Path("/tmp/f10_page")
log = work / "log.jsonl"
lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
log.write_text(
    lines[0] + lines[1].replace('"reason":"size"', '"reason":"ok"') + "".join(lines[2:]),
    encoding="utf-8", newline="\n",
)
env = os.environ.copy()
env["PYTHONPATH"] = "/content/Dense-Armor"
r = subprocess.run(
    [sys.executable, "-m", "dense_armor.utility.finance.status",
     "--config", "demo.toml"],
    capture_output=True, text=True, cwd=str(work), env=env,
)
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "demo.toml", "emergency_stop": false, "feed": {"exists": true, "path": "/tmp/f10_page/feed.bin"}, "log": {"exists": true, "last_hash": "5573286aec2926ff070501e3092c25f6a22db2e638db7a47b553fdaa2343f482", "path": "/tmp/f10_page/log.jsonl", "records": 493, "verified": false}, "ok": true}
~~~

`verified` is now `false`. The exit code is still 0: `status` reports, it does
not decide. The `last_hash` is unchanged because the last record is not the one
that was edited; the chain is broken at the second record.

## Details

**References.**
- Nasdaq. *TotalView-ITCH 5.0 specification*, 2018. Message layouts on
  pages 13 to 16 of the PDF.
- NIST. FIPS 180-4, *Secure Hash Standard*. August 2015. SHA-256 is
  section 6.2, pages 27 and 28 of the PDF.

**Choices stated explicitly.**
- The `account` passed to the checker is the stock symbol. A real desk would
  use a customer or a strategy account; the demo uses the symbol because it is
  the only identifier the feed carries.
- The `ref_price` of a stock is the price of the first `A` or `F` of that stock
  in the file. A second message of the same stock compares against that price,
  not against the previous one, so a 10 % move fails `price_band`.
- The `ring.width` must be 4 for the demo. The demo writes the four values
  `[order_ref, shares, price, t]` into the ring; a width other than 4 is
  refused before anything is written.
- The log is never extended by a second run. If `paths.log` exists, the demo
  exits 2 with `{"error": "log exists"}`. A desk that wants a second run
  removes the log, or uses a different configuration.
- The stop is checked once, at the start, and never again. A stop file created
  during the run does not interrupt it.

**Measured cost.** One run on the machine named above, with the Python version
in the JSON (`python: 3.13.18`). `elapsed_s` is one wall-clock measurement of
this run, not a benchmark. It varies between runs and machines, and no page,
docstring or test says it is fast or fixed.

**Declared limits.**
- The feed is synthetic: it has the byte layout of ITCH and none of its
  statistics. There is no Nasdaq connection, no real order and no execution
  report. The demo sends the orders to a checker
  that decides on the numbers of the configuration, and records the decisions.
- No execution is simulated. An `A` is an order that arrived; an approved
  decision is a decision, not a fill.
- The log has no signature and no retention policy. It is a chain, not a legal
  record.
- `elapsed_s` is one run on one machine, not a measurement of speed.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
