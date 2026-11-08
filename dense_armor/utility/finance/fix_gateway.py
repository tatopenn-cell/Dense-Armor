"""A FIX 4.4 initiator that sends limit orders and collects the reports."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any


def _validate(orders: list[dict[str, Any]], timeout_s: float) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a finite number > 0, got {timeout_s}")
    seen: set[str] = set()
    for i, o in enumerate(orders):
        if not isinstance(o, dict):
            raise ValueError(f"orders[{i}] must be a dict, got {type(o).__name__}")
        for key in ("cl_ord_id", "symbol", "side", "qty", "price"):
            if key not in o:
                raise ValueError(f"orders[{i}] missing key {key}")
        cid = o["cl_ord_id"]
        if not isinstance(cid, str) or not cid:
            raise ValueError(
                f"orders[{i}].cl_ord_id must be a non-empty str, got {cid!r}"
            )
        if cid in seen:
            raise ValueError(f"orders[{i}].cl_ord_id is not unique: {cid!r}")
        seen.add(cid)
        sym = o["symbol"]
        if not isinstance(sym, str) or not sym:
            raise ValueError(f"orders[{i}].symbol must be a non-empty str, got {sym!r}")
        side = o["side"]
        if side not in ("buy", "sell"):
            raise ValueError(f"orders[{i}].side must be 'buy' or 'sell', got {side!r}")
        for key in ("qty", "price"):
            v = o[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"orders[{i}].{key} must be a number, got {type(v).__name__}"
                )
            if not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(
                    f"orders[{i}].{key} must be a finite number > 0, got {v}"
                )


class _Collector:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.logged_on = threading.Event()
        self.session_id: Any = None
        self.done: set[str] = set()

    def add(self, r: dict[str, Any]) -> None:
        with self.lock:
            self.reports.append(r)
            if r["ord_status"] in ("2", "8"):
                self.done.add(r["cl_ord_id"])


def _make_application(collector: _Collector) -> Any:
    import quickfix as fix

    class _App(fix.Application):
        def onCreate(self, sessionID) -> None:
            return

        def onLogon(self, sessionID) -> None:
            collector.session_id = sessionID
            collector.logged_on.set()

        def onLogout(self, sessionID) -> None:
            return

        def toAdmin(self, sessionID, message) -> None:
            return

        def fromAdmin(self, sessionID, message) -> None:
            return

        def toApp(self, sessionID, message) -> None:
            return

        def fromApp(self, message, sessionID) -> None:
            msg_type = fix.MsgType()
            message.getHeader().getField(msg_type)
            if msg_type.getValue() != fix.MsgType_ExecutionReport:
                return
            r: dict[str, Any] = {
                "cl_ord_id": None,
                "ord_status": None,
                "exec_type": None,
                "cum_qty": None,
                "last_qty": None,
                "last_px": None,
                "avg_px": None,
            }
            f = fix.ClOrdID()
            if message.isSetField(f):
                message.getField(f)
                r["cl_ord_id"] = f.getValue()
            f = fix.OrdStatus()
            if message.isSetField(f):
                message.getField(f)
                r["ord_status"] = f.getValue()
            f = fix.ExecType()
            if message.isSetField(f):
                message.getField(f)
                r["exec_type"] = f.getValue()
            f = fix.CumQty()
            if message.isSetField(f):
                message.getField(f)
                r["cum_qty"] = float(f.getValue())
            f = fix.LastQty()
            if message.isSetField(f):
                message.getField(f)
                r["last_qty"] = float(f.getValue())
            f = fix.LastPx()
            if message.isSetField(f):
                message.getField(f)
                r["last_px"] = float(f.getValue())
            f = fix.AvgPx()
            if message.isSetField(f):
                message.getField(f)
                r["avg_px"] = float(f.getValue())
            collector.add(r)

    return _App()


def send_orders(
    settings_path: str | Path, orders: list[dict[str, Any]], timeout_s: float = 10.0
) -> list[dict[str, Any]]:
    """Send limit orders over FIX 4.4 and return the execution reports.

    Every order is validated before the session starts: a wrong one raises
    ``ValueError`` and no connection is made. The function starts a
    ``quickfix.SocketInitiator`` with the settings file, waits for the
    logon, sends one ``NewOrderSingle`` (``OrdType=2``, limit) per order,
    collects the execution reports until each ``ClOrdID`` has an
    ``OrdStatus`` of ``2`` (filled) or ``8`` (rejected), then logs out and
    stops the initiator. The initiator is stopped also on an exception.

    Args:
        settings_path: the QuickFIX settings file of the initiator.
        orders: a list of dicts, each with ``cl_ord_id`` (non-empty str,
            unique), ``symbol`` (non-empty str), ``side`` (``"buy"`` or
            ``"sell"``), ``qty`` and ``price`` (finite numbers > 0).
        timeout_s: the maximum time to wait, in seconds; finite, > 0.
            Default 10.0.

    Returns:
        A list of dicts, one per report, with ``cl_ord_id``,
        ``ord_status``, ``exec_type``, ``cum_qty``, ``last_qty`` (or
        ``None``), ``last_px`` (or ``None``), ``avg_px``.

    Raises:
        ValueError: if an order or ``timeout_s`` is wrong, before any
            connection.
        TimeoutError: if the logon or the reports do not arrive within
            ``timeout_s``; the message lists the ``ClOrdID`` still missing.
    """
    _validate(orders, timeout_s)
    import quickfix as fix

    collector = _Collector()
    app = _make_application(collector)
    settings = fix.SessionSettings(str(settings_path))
    store = fix.FileStoreFactory(settings)
    log = fix.FileLogFactory(settings)
    initiator = fix.SocketInitiator(app, store, settings, log)
    started = False
    try:
        initiator.start()
        started = True
        if not collector.logged_on.wait(timeout=timeout_s):
            raise TimeoutError(f"logon did not arrive within {timeout_s} s")
        for o in orders:
            msg = fix.Message()
            msg.getHeader().setField(fix.BeginString(fix.BeginString_FIX44))
            msg.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
            msg.setField(fix.ClOrdID(o["cl_ord_id"]))
            msg.setField(fix.Symbol(o["symbol"]))
            msg.setField(
                fix.Side(fix.Side_BUY if o["side"] == "buy" else fix.Side_SELL)
            )
            msg.setField(fix.TransactTime())
            msg.setField(fix.OrdType(fix.OrdType_LIMIT))
            msg.setField(fix.OrderQty(float(o["qty"])))
            msg.setField(fix.Price(float(o["price"])))
            fix.Session.sendToTarget(msg, collector.session_id)
        deadline = time.monotonic() + timeout_s
        wanted = {o["cl_ord_id"] for o in orders}
        while time.monotonic() < deadline:
            with collector.lock:
                if wanted <= collector.done:
                    break
            time.sleep(0.05)
        with collector.lock:
            missing = wanted - collector.done
            reports = list(collector.reports)
        if missing:
            raise TimeoutError(
                f"reports did not arrive within {timeout_s} s, missing ClOrdID: {sorted(missing)}"
            )
        return reports
    finally:
        if started:
            try:
                initiator.stop()
            except Exception:
                pass
