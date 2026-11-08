"""Tests for send_orders against the test FIX acceptor."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.needs_fix


def _order(
    cl_ord_id: str,
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 100.0,
    price: float = 100.0,
) -> dict:
    return {
        "cl_ord_id": cl_ord_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
    }


def test_1_one_order_one_report(fix_acceptor):
    from dense_armor.utility.finance.fix_gateway import send_orders

    reports = send_orders(fix_acceptor, [_order("S1")], timeout_s=5.0)
    assert len(reports) == 1
    r = reports[0]
    assert r["cl_ord_id"] == "S1"
    assert r["ord_status"] == "2"
    assert r["exec_type"] == "F"
    assert r["cum_qty"] == 100.0
    assert r["avg_px"] == 100.0
    assert r["last_qty"] == 100.0
    assert r["last_px"] == 100.0


def test_2_two_orders_two_reports(fix_acceptor):
    from dense_armor.utility.finance.fix_gateway import send_orders

    reports = send_orders(
        fix_acceptor,
        [
            _order("S1", symbol="AAPL", side="buy"),
            _order("S2", symbol="MSFT", side="sell"),
        ],
        timeout_s=5.0,
    )
    assert len(reports) == 2
    by_id = {r["cl_ord_id"]: r for r in reports}
    assert set(by_id) == {"S1", "S2"}
    assert by_id["S1"]["ord_status"] == "2"
    assert by_id["S2"]["ord_status"] == "2"


def test_3_acceptor_log_has_logon_order_report_logout(fix_acceptor):
    from dense_armor.utility.finance.fix_gateway import send_orders

    send_orders(fix_acceptor, [_order("S1")], timeout_s=5.0)
    log_dir = fix_acceptor.parent / "log_acceptor"
    assert log_dir.exists()
    text = ""
    for f in log_dir.iterdir():
        if f.is_file():
            text += f.read_text(encoding="utf-8", errors="replace")
    assert "35=A" in text
    assert "35=D" in text
    assert "11=S1" in text
    assert "54=1" in text
    assert "40=2" in text
    assert "35=8" in text
    assert "35=5" in text


def test_4_invalid_order_no_acceptor_needed():
    from dense_armor.utility.finance.fix_gateway import send_orders

    with pytest.raises(ValueError, match="qty"):
        send_orders("/nonexistent.cfg", [_order("S1", qty=float("nan"))])
    with pytest.raises(ValueError, match="side"):
        send_orders("/nonexistent.cfg", [_order("S1", side="hold")])
    with pytest.raises(ValueError, match="not unique"):
        send_orders(
            "/nonexistent.cfg",
            [_order("S1"), _order("S1")],
        )


def test_5_no_acceptor_timeout(tmp_path):
    pytest.importorskip("quickfix", reason="quickfix not installed")
    from dense_armor.utility.finance.fix_gateway import send_orders

    root = Path(__file__).resolve().parents[2]
    dictionary = root / "fixtures" / "finance" / "FIX44.xml"
    cfg = tmp_path / "no_acceptor.cfg"
    cfg.write_text(
        "[DEFAULT]\n"
        "ConnectionType=initiator\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=CLIENT\n"
        "TargetCompID=SERVER\n"
        "SocketConnectPort=1\n"
        "SocketConnectHost=127.0.0.1\n"
        "StartTime=00:00:00\n"
        "EndTime=00:00:00\n"
        "HeartBtInt=5\n"
        "ReconnectInterval=1\n"
        f"FileStorePath={tmp_path / 'store'}\n"
        f"FileLogPath={tmp_path / 'log'}\n"
        f"DataDictionary={dictionary}\n"
        "UseDataDictionary=Y\n"
        "\n"
        "[SESSION]\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=CLIENT\n"
        "TargetCompID=SERVER\n"
        "ConnectionType=initiator\n",
        encoding="utf-8",
        newline="\n",
    )
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="logon"):
        send_orders(cfg, [_order("S1")], timeout_s=2.0)
    assert time.monotonic() - t0 < 5.0


# --- coverage of the non-session branches --------------------------------


def test_6_validate_timeout_s_zero():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match="timeout_s"):
        _validate([], 0.0)
    with pytest.raises(ValueError, match="timeout_s"):
        _validate([], float("nan"))


def test_7_validate_not_a_dict():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match=r"orders\[0\] must be a dict"):
        _validate([42], 1.0)  # type: ignore[list-item]


def test_8_validate_missing_key():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match="missing key price"):
        _validate([{"cl_ord_id": "S1", "symbol": "AAPL", "side": "buy", "qty": 1}], 1.0)


def test_9_validate_cl_ord_id_empty():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match="cl_ord_id must be a non-empty str"):
        _validate([_order("")], 1.0)


def test_10_validate_symbol_empty():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match="symbol must be a non-empty str"):
        _validate([_order("S1", symbol="")], 1.0)


def test_11_validate_qty_not_number():
    from dense_armor.utility.finance.fix_gateway import _validate

    with pytest.raises(ValueError, match="qty must be a number"):
        _validate([_order("S1", qty="100")], 1.0)  # type: ignore[arg-type]


def test_12_collector_add_done():
    from dense_armor.utility.finance.fix_gateway import _Collector

    c = _Collector()
    c.add(
        {
            "cl_ord_id": "S1",
            "ord_status": "0",
            "exec_type": "0",
            "cum_qty": None,
            "last_qty": None,
            "last_px": None,
            "avg_px": None,
        }
    )
    assert c.done == set()
    c.add(
        {
            "cl_ord_id": "S1",
            "ord_status": "2",
            "exec_type": "F",
            "cum_qty": 100.0,
            "last_qty": 100.0,
            "last_px": 100.0,
            "avg_px": 100.0,
        }
    )
    assert c.done == {"S1"}
    c.add(
        {
            "cl_ord_id": "S2",
            "ord_status": "8",
            "exec_type": "8",
            "cum_qty": 0.0,
            "last_qty": None,
            "last_px": None,
            "avg_px": 0.0,
        }
    )
    assert c.done == {"S1", "S2"}


def test_13_application_methods_directly():
    """Call the _App methods without a session, to cover them."""
    pytest.importorskip("quickfix", reason="quickfix not installed")
    from dense_armor.utility.finance.fix_gateway import _Collector, _make_application

    c = _Collector()
    app = _make_application(c)
    # onCreate, onLogout, toAdmin, fromAdmin, toApp
    app.onCreate(None)
    app.onLogout(None)
    app.toAdmin(None, None)
    app.fromAdmin(None, None)
    app.toApp(None, None)
    # onLogon sets session_id and the event
    app.onLogon("session-1")
    assert c.session_id == "session-1"
    assert c.logged_on.is_set()


def test_14_from_app_ignores_non_execution_report():
    """A heartbeat in fromApp returns without a report."""
    pytest.importorskip("quickfix", reason="quickfix not installed")
    import quickfix as fix

    from dense_armor.utility.finance.fix_gateway import _Collector, _make_application

    c = _Collector()
    app = _make_application(c)
    msg = fix.Message()
    msg.getHeader().setField(fix.BeginString(fix.BeginString_FIX44))
    msg.getHeader().setField(fix.MsgType(fix.MsgType_Heartbeat))
    app.fromApp(msg, None)
    assert c.reports == []


def test_15_from_app_reads_execution_report():
    """An ExecutionReport in fromApp fills the collector."""
    pytest.importorskip("quickfix", reason="quickfix not installed")
    import quickfix as fix

    from dense_armor.utility.finance.fix_gateway import _Collector, _make_application

    c = _Collector()
    app = _make_application(c)
    msg = fix.Message()
    msg.getHeader().setField(fix.BeginString(fix.BeginString_FIX44))
    msg.getHeader().setField(fix.MsgType(fix.MsgType_ExecutionReport))
    msg.setField(fix.ClOrdID("S1"))
    msg.setField(fix.OrdStatus(fix.OrdStatus_FILLED))
    msg.setField(fix.ExecType(fix.ExecType_TRADE))
    msg.setField(fix.CumQty(100.0))
    msg.setField(fix.LastQty(100.0))
    msg.setField(fix.LastPx(100.0))
    msg.setField(fix.AvgPx(100.0))
    app.fromApp(msg, None)
    assert len(c.reports) == 1
    r = c.reports[0]
    assert r["cl_ord_id"] == "S1"
    assert r["ord_status"] == "2"
    assert r["exec_type"] == "F"
    assert r["cum_qty"] == 100.0
    assert r["last_qty"] == 100.0
    assert r["last_px"] == 100.0
    assert r["avg_px"] == 100.0
    assert c.done == {"S1"}
