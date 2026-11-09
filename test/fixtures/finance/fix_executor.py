"""A test FIX 4.4 acceptor that fills every limit order at its price.

Adapted for Python 3 from the executor example of QuickFIX
(``papers/quickfix_executor_example.py``, The QuickFIX Software License,
Version 1.0, Copyright (c) 2001-2020 Oren Miller). This file is a test
fixture: it starts with a settings file, waits for a logon, and answers
each ``NewOrderSingle`` with an ``ExecutionReport`` of ``OrdStatus=FILLED``
at the price of the order. It exits when the standard input closes or when
SIGTERM arrives.

Usage: ``python fix_executor.py <settings.cfg>``
"""

from __future__ import annotations

import signal
import sys
import threading
import time

import quickfix as fix


class Application(fix.Application):
    order_id = 0
    exec_id = 0

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()

    def onCreate(self, sessionID) -> None:
        return

    def onLogon(self, sessionID) -> None:
        return

    def onLogout(self, sessionID) -> None:
        return

    def toAdmin(self, sessionID, message) -> None:
        return

    def fromAdmin(self, sessionID, message) -> None:
        return

    def toApp(self, sessionID, message) -> None:
        return

    def fromApp(self, message, sessionID) -> None:
        begin_string = fix.BeginString()
        msg_type = fix.MsgType()
        message.getHeader().getField(begin_string)
        message.getHeader().getField(msg_type)
        if msg_type.getValue() != fix.MsgType_NewOrderSingle:
            return
        ord_type = fix.OrdType()
        symbol = fix.Symbol()
        side = fix.Side()
        order_qty = fix.OrderQty()
        price = fix.Price()
        cl_ord_id = fix.ClOrdID()
        message.getField(ord_type)
        if ord_type.getValue() != fix.OrdType_LIMIT:
            raise fix.IncorrectTagValue(ord_type.getField())
        message.getField(symbol)
        message.getField(side)
        message.getField(order_qty)
        message.getField(price)
        message.getField(cl_ord_id)
        report = fix.Message()
        report.getHeader().setField(begin_string)
        report.getHeader().setField(fix.MsgType(fix.MsgType_ExecutionReport))
        report.setField(fix.OrderID(self.gen_order_id()))
        report.setField(fix.ExecID(self.gen_exec_id()))
        report.setField(fix.OrdStatus(fix.OrdStatus_FILLED))
        report.setField(symbol)
        report.setField(side)
        report.setField(fix.CumQty(order_qty.getValue()))
        report.setField(fix.AvgPx(price.getValue()))
        report.setField(fix.LastQty(order_qty.getValue()))
        report.setField(fix.LastPx(price.getValue()))
        report.setField(cl_ord_id)
        report.setField(order_qty)
        report.setField(fix.ExecType(fix.ExecType_TRADE))
        report.setField(fix.LeavesQty(0))
        try:
            fix.Session.sendToTarget(report, sessionID)
        except fix.SessionNotFound:
            return

    def gen_order_id(self) -> str:
        self.order_id += 1
        return str(self.order_id)

    def gen_exec_id(self) -> str:
        self.exec_id += 1
        return str(self.exec_id)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fix_executor.py <settings.cfg>", file=sys.stderr)
        return 2
    settings = fix.SessionSettings(sys.argv[1])
    application = Application()
    store_factory = fix.FileStoreFactory(settings)
    log_factory = fix.FileLogFactory(settings)
    acceptor = fix.SocketAcceptor(application, store_factory, settings, log_factory)

    _stop_flag = threading.Event()

    def _stop(signum, frame):
        _stop_flag.set()
        acceptor.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    acceptor.start()
    try:
        while not _stop_flag.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        acceptor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
