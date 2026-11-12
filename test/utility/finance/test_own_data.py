"""The files that a visitor brings to the dashboard: prices, orders, uploads and reports."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from dashboard import own_data as od
from dashboard.paths import FIXTURES


def _prices_csv(n=40, delim=",", decimal=".", base=100.0):
    rows = [delim.join(["Date", "AAA", "BBB"])]
    for i in range(n):
        a = f"{base + i * 0.5:.2f}".replace(".", decimal)
        b = f"{50 + i * 0.1:.2f}".replace(".", decimal)
        rows.append(delim.join([f"2024-01-{1 + i % 28:02d}", a, b]))
    return "\n".join(rows)


def test_prices_one_column_per_asset_with_dates():
    p = od.parse_prices(_prices_csv())
    assert p["columns"] == ["AAA", "BBB"]
    assert p["prices"].shape == (40, 2) and p["dropped"] == 0
    assert p["dates"][0] == "2024-01-01" and len(p["dates"]) == 40
    assert od.log_returns(p["prices"]).shape == (39, 2)


def test_prices_with_semicolons_and_decimal_commas():
    p = od.parse_prices(_prices_csv(delim=";", decimal=","))
    assert p["prices"][0].tolist() == [100.0, 50.0]


def test_prices_without_a_date_column():
    text = "AAA,BBB\n" + "\n".join(f"{100 + i},{50 + i}" for i in range(35))
    p = od.parse_prices(text)
    assert p["dates"] is None and p["columns"] == ["AAA", "BBB"]


def test_prices_of_a_yahoo_file_use_only_the_close_and_the_ticker_name():
    p = od.parse_prices((FIXTURES / "GSPC.csv").read_text(encoding="utf-8"))
    assert p["columns"] == ["^GSPC"]
    n_lines = len((FIXTURES / "GSPC.csv").read_text(encoding="utf-8").splitlines())
    assert p["prices"].shape == (n_lines - 3, 1)
    assert p["prices"][0, 0] == pytest.approx(4438.259765625)


def test_prices_with_a_missing_value_drop_the_row_and_count_it():
    lines = _prices_csv().splitlines()
    lines[5] = "2024-01-05,,50.4"
    lines[9] = "2024-01-09,abc,50.8"
    p = od.parse_prices("\n".join(lines))
    assert p["prices"].shape == (38, 2) and p["dropped"] == 2


def test_prices_errors_name_the_problem():
    with pytest.raises(od.DataError, match="empty"):
        od.parse_prices("  \n ")
    with pytest.raises(od.DataError, match="only 5 usable rows; at least 30"):
        od.parse_prices(_prices_csv(5))
    half = "Date,AAA\n" + "\n".join(
        f"2024-01-{i % 28 + 1:02d},{'x' if i % 2 else 100 + i}" for i in range(40)
    )
    with pytest.raises(od.DataError, match="no column of numbers"):
        od.parse_prices(half)
    with pytest.raises(od.DataError, match="no rows of numbers"):
        od.parse_prices("Date,AAA\nfoo,bar")
    with pytest.raises(od.DataError, match="no rows of numbers"):
        od.parse_prices(
            "Date,Name\n" + "\n".join(f"2024-01-{i % 28 + 1:02d},x" for i in range(40))
        )
    lines = _prices_csv().splitlines()
    lines[7] = "2024-01-07,0,50.6"
    with pytest.raises(
        od.DataError, match=r"column 'AAA' has a price of 0 in row 7 of the data"
    ):
        od.parse_prices("\n".join(lines))


def test_decode_handles_a_bom_and_latin1_and_refuses_a_big_file():
    assert od.decode("﻿A,B".encode()) == "A,B"
    assert od.decode("caf\xe9".encode("latin-1")) == "caf\xe9"
    with pytest.raises(od.DataError, match="KB"):
        od.decode(b"x" * (od.MAX_BYTES + 1))


ORDERS = "account,side,size,price\nAAPL,buy,10,100\nAAPL,S,5,101\nMSFT,sell,1,300\n"


def test_orders_get_a_reference_price_and_a_time_by_default():
    o = od.parse_orders(ORDERS)
    assert [x["row"] for x in o] == [2, 3, 4]
    assert [x["side"] for x in o] == ["buy", "sell", "sell"]
    assert [x["ref_price"] for x in o] == [100.0, 100.0, 300.0]
    assert [x["t"] for x in o] == [0.0, 1.0, 2.0]


def test_orders_with_aliases_a_reference_and_iso_times():
    text = (
        "symbol,direction,qty,limit,ref_price,timestamp\n"
        "AAPL,bid,10,100,99,2024-01-02T09:30:00+00:00\n"
        "AAPL,ask,5,101,99,2024-01-02T09:30:01+00:00\n"
    )
    o = od.parse_orders(text)
    assert o[0]["ref_price"] == 99.0 and o[1]["t"] - o[0]["t"] == pytest.approx(1.0)
    assert o[0]["account"] == "AAPL" and o[0]["size"] == 10.0


def test_orders_errors_name_the_row_and_the_column():
    with pytest.raises(od.DataError, match="missing: side, size"):
        od.parse_orders("account,price\nA,1")
    with pytest.raises(od.DataError, match="row 3: the side 'hold'"):
        od.parse_orders("account,side,size,price\nA,buy,1,1\nA,hold,1,1")
    with pytest.raises(od.DataError, match="row 2: the size '0'"):
        od.parse_orders("account,side,size,price\nA,buy,0,1")
    with pytest.raises(od.DataError, match="row 2: the price 'x'"):
        od.parse_orders("account,side,size,price\nA,buy,1,x")
    with pytest.raises(od.DataError, match="row 2: the account is empty"):
        od.parse_orders("account,side,size,price\n,buy,1,1")
    with pytest.raises(od.DataError, match="row 2: the time 'soon'"):
        od.parse_orders("account,side,size,price,t\nA,buy,1,1,soon")
    with pytest.raises(od.DataError, match="row 2: the ref_price '-1'"):
        od.parse_orders("account,side,size,price,ref_price\nA,buy,1,1,-1")
    with pytest.raises(od.DataError, match="no orders"):
        od.parse_orders("account,side,size,price")
    big = "account,side,size,price\n" + "A,buy,1,1\n" * (od.MAX_ORDERS + 1)
    with pytest.raises(od.DataError, match="limit is 50000"):
        od.parse_orders(big)


def test_an_upload_is_kept_listed_and_loaded_again(tmp_path):
    raw = _prices_csv().encode()
    meta = od.store_prices(tmp_path, "my prices.csv", raw)
    assert meta["id"].startswith("up_") and meta["n_rows"] == 40
    assert meta["columns"] == ["AAA", "BBB"] and meta["first_date"] == "2024-01-01"
    assert od.store_prices(tmp_path, "again.csv", raw)["id"] == meta["id"]
    parsed, m2 = od.load_prices(tmp_path, meta["id"])
    assert parsed["prices"].shape == (40, 2) and m2["name"] in (
        "my prices.csv",
        "again.csv",
    )
    assert [m["id"] for m in od.list_uploads(tmp_path)] == [meta["id"]]


def test_an_upload_id_that_is_not_ours_is_refused(tmp_path):
    for bad in ("../x", "up_zz", "", None, "up_" + "a" * 11):
        with pytest.raises(od.DataError, match="not an uploaded file"):
            od.load_prices(tmp_path, bad)
    with pytest.raises(od.DataError, match="not there any more"):
        od.load_prices(tmp_path, "up_" + "0" * 10)
    (tmp_path / "uploads" / "up_broken.json").write_text("{", encoding="utf-8")
    assert od.list_uploads(tmp_path) == []


RESULT = {
    "chain": {"data": "up_x", "signal": "hoeffding", "n": 100},
    "data_meta": {"kind": "upload", "source": "p.csv", "n": 99},
    "series": {
        "accuracy": [{"i": 0, "v": 0.5}, {"i": 10, "v": 0.6}],
        "risk": [],
        "features": [],
    },
    "alarms": [3, 7],
    "metrics": {
        "accuracy": 0.55,
        "portfolio": {"weights": [0.25, 0.75], "error": None},
        "role_attributes": {"signal": {"budget_s": 1.0}},
    },
}


def test_the_report_is_a_long_csv_that_a_spreadsheet_opens():
    rows = list(csv.reader(io.StringIO(od.report_csv(RESULT))))
    assert rows[0] == ["section", "name", "index", "value"]
    assert ["chain", "signal", "", "hoeffding"] in rows
    assert ["data", "source", "", "p.csv"] in rows
    assert ["metric", "accuracy", "", "0.55"] in rows
    assert ["metric", "portfolio.weights[1]", "", "0.75"] in rows
    assert ["series_accuracy", "accuracy", "10", "0.6"] in rows
    assert ["alarm", "risk", "7", ""] in rows
    assert not any("role_attributes" in r[1] for r in rows)


def test_a_report_is_saved_under_a_run_id_and_read_back(tmp_path):
    rid = od.save_report(tmp_path, RESULT)
    assert len(rid) == 12 and od.load_report(tmp_path, rid) == RESULT
    for bad in ("../x", "zz", "", None):
        with pytest.raises(od.DataError, match="not a run id"):
            od.load_report(tmp_path, bad)
    with pytest.raises(od.DataError, match="is not there"):
        od.load_report(tmp_path, "0" * 12)


def test_the_checked_orders_csv():
    res = [
        {
            "row": 2,
            "account": "A",
            "side": "buy",
            "size": 1.0,
            "price": 2.0,
            "ref_price": 2.0,
            "t": 0.0,
            "approved": True,
            "reason": "ok",
        },
        {
            "row": 3,
            "account": "A",
            "side": "sell",
            "size": 9.0,
            "price": 2.0,
            "ref_price": 2.0,
            "t": 1.0,
            "approved": False,
            "reason": "size",
        },
    ]
    rows = list(csv.reader(io.StringIO(od.orders_csv(res))))
    assert (
        rows[0][-2:] == ["approved", "reason"]
        and rows[1][-2:] == ["1", "ok"]
        and rows[2][-2:] == ["0", "size"]
    )
    assert Path(od.__file__).name == "own_data.py"
    assert np.isclose(float(rows[2][3]), 9.0)
