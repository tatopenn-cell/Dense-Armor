# FIX gateway: send limit orders to a test acceptor

FIX is the language most brokers and exchanges speak. A desk sends a
`NewOrderSingle` and reads an `ExecutionReport` back. This module is the
initiator side: it logs on, sends limit orders, waits for the reports, and
logs out. The acceptor is a small program of the repository, not a broker: it
fills every limit order at its price.

## 1. Start the acceptor and send one order

The fixture `test/fixtures/finance/fix_executor.py` is adapted from the
QuickFIX executor example (The QuickFIX Software License, Version 1.0,
Copyright (c) 2001-2020 Oren Miller). Two settings files are written in
`<tmp>`: one for the acceptor, one for the initiator.

~~~python
import socket, subprocess, sys, tempfile, time
from pathlib import Path

work = Path(tempfile.mkdtemp())
root = Path("/content/Dense-Armor")
executor = root / "test" / "fixtures" / "finance" / "fix_executor.py"
dictionary = root / "test" / "fixtures" / "finance" / "FIX44.xml"

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]

acceptor_cfg = work / "acceptor.cfg"
acceptor_cfg.write_text(
    "[DEFAULT]\nConnectionType=acceptor\nBeginString=FIX.4.4\n"
    "SenderCompID=SERVER\nTargetCompID=CLIENT\n"
    f"SocketAcceptPort={port}\nStartTime=00:00:00\nEndTime=00:00:00\n"
    "HeartBtInt=5\n"
    f"FileStorePath={work / 'store_acceptor'}\n"
    f"FileLogPath={work / 'log_acceptor'}\n"
    f"DataDictionary={dictionary}\nUseDataDictionary=Y\n\n"
    "[SESSION]\nBeginString=FIX.4.4\nSenderCompID=SERVER\n"
    "TargetCompID=CLIENT\nConnectionType=acceptor\n",
    encoding="utf-8", newline="\n",
)
initiator_cfg = work / "initiator.cfg"
initiator_cfg.write_text(
    "[DEFAULT]\nConnectionType=initiator\nBeginString=FIX.4.4\n"
    "SenderCompID=CLIENT\nTargetCompID=SERVER\n"
    f"SocketConnectPort={port}\nSocketConnectHost=127.0.0.1\n"
    "StartTime=00:00:00\nEndTime=00:00:00\nHeartBtInt=5\n"
    "ReconnectInterval=1\n"
    f"FileStorePath={work / 'store_init'}\n"
    f"FileLogPath={work / 'log_init'}\n"
    f"DataDictionary={dictionary}\nUseDataDictionary=Y\n\n"
    "[SESSION]\nBeginString=FIX.4.4\nSenderCompID=CLIENT\n"
    "TargetCompID=SERVER\nConnectionType=initiator\n",
    encoding="utf-8", newline="\n",
)

proc = subprocess.Popen(
    [sys.executable, str(executor), str(acceptor_cfg)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
for _ in range(100):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            break
    except OSError:
        time.sleep(0.1)

from dense_armor.utility.finance.fix_gateway import send_orders

reports = send_orders(
    initiator_cfg,
    [{"cl_ord_id": "S1", "symbol": "AAPL", "side": "buy",
      "qty": 100, "price": 100.0}],
    timeout_s=5.0,
)
r = reports[0]
print("cl_ord_id:", r["cl_ord_id"])
print("ord_status:", r["ord_status"])
print("exec_type:", r["exec_type"])
print("cum_qty:", r["cum_qty"])
print("avg_px:", r["avg_px"])
print("last_qty:", r["last_qty"])
print("last_px:", r["last_px"])
proc.terminate()
proc.wait(timeout=5)
~~~

~~~
cl_ord_id: S1
ord_status: 2
exec_type: F
cum_qty: 100.0
avg_px: 100.0
last_qty: 100.0
last_px: 100.0
~~~

`ord_status: 2` is `OrdStatus=FILLED` and `exec_type: F` is `ExecType=TRADE`,
the two values of FIX 4.4 for a completed execution. `cum_qty` and `last_qty`
are 100, the quantity of the order, and `avg_px` and `last_px` are 100.0, the
limit price: the test acceptor fills at the price of the order.

## 2. The invalid order fails before the session

Every order is validated before `fix.SocketInitiator` starts: a wrong one
raises `ValueError` and no connection is made. `cl_ord_id` must be non-empty
and unique in the list, `symbol` non-empty, `side` `buy` or `sell`, `qty` and
`price` finite numbers greater than 0.

~~~python
from dense_armor.utility.finance.fix_gateway import send_orders

try:
    send_orders(
        "/nonexistent.cfg",
        [{"cl_ord_id": "S1", "symbol": "AAPL", "side": "hold",
          "qty": 100, "price": 100.0}],
    )
except ValueError as e:
    print("ValueError:", e)
~~~

~~~
ValueError: orders[0].side must be 'buy' or 'sell', got 'hold'
~~~

The path `/nonexistent.cfg` is never opened: the validation raises first.

## 3. No acceptor listening: a timeout, not a hang

A `SocketConnectPort` with nothing behind it makes `quickfix` retry forever.
`send_orders` waits at most `timeout_s`, then raises `TimeoutError` listing the
`ClOrdID` still missing and stops the initiator.

~~~python
from dense_armor.utility.finance.fix_gateway import send_orders

try:
    send_orders(
        "/tmp/no_acceptor.cfg",
        [{"cl_ord_id": "S1", "symbol": "AAPL", "side": "buy",
          "qty": 100, "price": 100.0}],
        timeout_s=2.0,
    )
except TimeoutError as e:
    print(f"TimeoutError: {e}")
~~~

~~~
TimeoutError: logon did not arrive within 2.0 s
~~~

The wall time is 3.0-3.3 s, not 2.0 s: QuickFIX retries every second, and the
wait is checked at the next retry. The contract is "at most `timeout_s` plus
one retry interval", and the delivery of G4 states it.

## Details

**References.**
- QuickFIX. *FIX 4.4 XML data dictionary*
  (`test/fixtures/finance/FIX44.xml`, copied from the QuickFIX distribution,
  The QuickFIX Software License, Version 1.0, Copyright (c) 2001-2020 Oren
  Miller). The tags in this page are checked against the dictionary:
  `ClOrdID` (11), `Side` (54), `TransactTime` (60), `OrdType` (40),
  `OrderID` (37), `ExecID` (17), `ExecType` (150), `OrdStatus` (39),
  `LeavesQty` (151), `CumQty` (14), `AvgPx` (6), `LastQty` (32), `LastPx` (31).
  The enums: `Side` 1 = BUY, 2 = SELL; `OrdType` 2 = LIMIT; `OrdStatus` 2 =
  FILLED, 8 = REJECTED; `ExecType` F = TRADE.
- QuickFIX. *Executor example*, adapted for Python 3 in
  `test/fixtures/finance/fix_executor.py`.

**Choices stated explicitly.**
- FIX 4.4, limit orders only, `HandlInst` not sent. A limit order is what the
  test acceptor fills; a market order has no price and the acceptor would
  reject it with `OrdType` wrong.
- `ExecType=TRADE` (F), not `FILL` (2). In FIX 4.4 `ExecType` values are
  `0` NEW, `3` DONE_FOR_DAY, `4` CANCELED, `5` REPLACED, `8` REJECTED,
  `F` TRADE. `2` is not in the enum of FIX 4.4, and a report with `150=2` is
  rejected by the initiator with a `Reject` (35=3).
- The `session_id` is taken from `onLogon`, and every `NewOrderSingle` is sent
  with `Session.sendToTarget(msg, session_id)`. A message without the Sender
  and Target in the header does not find the session, and
  `Session.sendToTarget(msg)` raises `SessionNotFound`.
- The acceptor is a process of the repository. It is a test fixture, not a
  broker. The orders are not sent to any real venue.
- `timeout_s` is checked with `math.isfinite(timeout_s) and timeout_s > 0`, so
  a NaN or a negative number is `ValueError` before the session.

**Measured cost.** Not measured. The page reports the two times the code shows
(`TimeoutError after 3.0 s`), not a latency claim.

**Declared limits.**
- A test acceptor, not a broker. No certification with any venue, no
  production FIX session, no sequence number recovery beyond what QuickFIX
  does.
- No cancel, no replace, no resend request. Only `NewOrderSingle` with
  `OrdType=2`.
- One session. The settings file has one `[SESSION]` block; a multi-session
  settings file would need a change of the initiator.
- The sequence numbers are in the file store of the run
  (`FileStorePath`). A different run with the same store continues from where
  the previous one stopped; the fixture uses a fresh `tmp_path`, so each test
  starts at 1.
- One report per `ClOrdID` is required, with `OrdStatus` `2` or `8`. A
  partial fill (`OrdStatus` `1`) would not close the `ClOrdID`, and
  `send_orders` would wait until the timeout. The test acceptor fills in full;
  a real acceptor that sends partials would need a longer contract.
- No signature, no encryption, no TLS. FIX over plain TCP on `127.0.0.1`.
- The timeout wall clock is `timeout_s` plus one retry interval (1 s in the
  fixture). The page says so.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
