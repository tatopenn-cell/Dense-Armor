"""The dashboard with the visitor's own files: prices into the chain, orders into the checker, plans."""

from __future__ import annotations

import csv
import io
import json

import pytest

pytest.importorskip("flask", reason="flask not installed")
pytest.importorskip("tomllib", reason="load_config needs tomllib, Python 3.11 or later")

from dashboard import own_data as od  # noqa: E402
from dashboard.app import create_app  # noqa: E402
from dashboard.paths import FIXTURES  # noqa: E402

TOML = """\
[schedule]
total = 100.0
n = 4

[pretrade]
max_notional = 100000.0
max_size = 300.0
price_band = 0.05
max_orders = 10
window_s = 1.0
duplicate_window_s = 0.05

[ring]
capacity = 3
width = 2

[paths]
log = "log.jsonl"
feed = "feed.bin"
emergency_stop = "STOP"
"""


@pytest.fixture
def client(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML, encoding="utf-8", newline="\n")
    return create_app(str(cfg)).test_client()


def _upload(client, text, name="p.csv"):
    return client.post(
        "/api/data/upload",
        data={"file": (io.BytesIO(text.encode()), name)},
        content_type="multipart/form-data",
    )


PRICES4 = od.sample_prices_csv(FIXTURES, 300)
PICKS = {
    "data": "",
    "features": "none",
    "signal": "hoeffding",
    "portfolio": "none",
    "risk": "none",
    "execution": "none",
}


def test_the_chain_runs_from_any_folder_not_only_from_the_root(
    client, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    r = client.post("/api/chain/run", json={**PICKS, "data": "gspc", "n": 200})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["metrics"]["n"] > 100, j


def test_an_upload_is_checked_and_described(client):
    r = _upload(client, PRICES4)
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["id"].startswith("up_")
    assert j["columns"] == ["GSPC", "AAPL", "MSFT", "TSLA"] and j["n_rows"] == 300
    assert j["first_date"] < j["last_date"]
    listed = client.get("/api/data/uploads").get_json()
    assert [m["id"] for m in listed] == [j["id"]]


def test_a_bad_upload_says_why(client):
    assert (
        client.post("/api/data/upload").get_json()["error"].startswith("send the CSV")
    )
    r = _upload(client, "Date,AAA\n2024-01-01,5\n")
    assert (
        r.status_code == 400
        and "only 1 usable rows; at least 30" in r.get_json()["error"]
    )
    big = client.post(
        "/api/data/upload",
        data={"file": (io.BytesIO(b"x" * (od.MAX_BYTES + 5)), "b.csv")},
        content_type="multipart/form-data",
    )
    assert big.status_code == 400 and "KB" in big.get_json()["error"]


def test_the_chain_on_my_file_gives_a_run_id_and_a_report_to_download(client):
    up = _upload(client, PRICES4).get_json()["id"]
    r = client.post(
        "/api/chain/run",
        json={
            **PICKS,
            "data": up,
            "portfolio": "min_variance",
            "n": 250,
            "column": "AAPL",
        },
    )
    j = r.get_json()
    assert r.status_code == 200 and j["ok"], j
    assert j["data_meta"]["kind"] == "upload" and j["data_meta"]["column"] == "AAPL"
    assert j["data_meta"]["columns"] == ["GSPC", "AAPL", "MSFT", "TSLA"]
    weights = j["metrics"]["portfolio"]["weights"]
    assert len(weights) == 4 and abs(sum(weights) - 1.0) < 1e-3
    rid = j["run_id"]
    rows = list(
        csv.reader(
            io.StringIO(client.get(f"/api/report/{rid}.csv").get_data(as_text=True))
        )
    )
    assert rows[0] == ["section", "name", "index", "value"]
    assert ["data", "column", "", "AAPL"] in rows
    assert any(r[0] == "metric" and r[1] == "accuracy" for r in rows)
    assert any(r[0] == "series_accuracy" for r in rows)
    again = client.get(f"/api/report/{rid}.json").get_json()
    assert again["run_id"] == rid and again["metrics"]["n"] == j["metrics"]["n"]
    csv_r = client.get(f"/api/report/{rid}.csv")
    assert (
        "attachment" in csv_r.headers["Content-Disposition"]
        and csv_r.mimetype == "text/csv"
    )


def test_a_file_with_one_price_column_runs_the_chain_but_not_the_portfolio(client):
    text = "Date,AAA\n" + "\n".join(
        f"2024-01-{i % 28 + 1:02d},{100 + i * 0.3 + (i % 5)}" for i in range(80)
    )
    up = _upload(client, text).get_json()["id"]
    j = client.post(
        "/api/chain/run", json={**PICKS, "data": up, "portfolio": "min_variance"}
    ).get_json()
    assert (
        j["ok"] and "at least two price columns" in j["metrics"]["portfolio"]["error"]
    )


def test_the_chain_refuses_what_is_not_an_upload_or_a_column(client):
    up = _upload(client, PRICES4).get_json()["id"]
    gone = client.post(
        "/api/chain/run", json={**PICKS, "data": "up_" + "0" * 10}
    ).get_json()
    assert gone["ok"] is False and "not there any more" in gone["error"]
    col = client.post(
        "/api/chain/run", json={**PICKS, "data": up, "column": "NOPE"}
    ).get_json()
    assert col["ok"] is False and "has no column 'NOPE'" in col["error"]
    bad = client.post("/api/chain/run", json={**PICKS, "data": "up_../x"}).get_json()
    assert bad["ok"] is False and "not an uploaded file" in bad["error"]
    assert (
        client.post("/api/chain/run", json={**PICKS, "data": "nope"})
        .get_json()["error"]
        .startswith("unknown data")
    )


def test_reports_that_do_not_exist_or_are_not_ids(client):
    for name in ("zz.csv", "0" * 12 + ".csv", "../etc.csv"):
        assert client.get(f"/api/report/{name}").status_code == 404
    up = _upload(client, PRICES4).get_json()["id"]
    rid = client.post("/api/chain/run", json={**PICKS, "data": up}).get_json()["run_id"]
    assert client.get(f"/api/report/{rid}.xml").status_code == 404


def test_the_sample_files_are_valid_inputs_of_the_same_pages(client):
    p = client.get("/sample/prices.csv")
    assert od.parse_prices(p.get_data(as_text=True))["prices"].shape == (500, 4)
    o = od.parse_orders(client.get("/sample/orders.csv").get_data(as_text=True))
    assert len(o) == 60
    v = od.parse_history(client.get("/sample/volumes.csv").get_data(as_text=True))
    assert len(v) == 20 and len(v[0]) == 10
    assert _upload(client, p.get_data(as_text=True)).get_json()["ok"]


def _orders(client, text=None, **form):
    text = (
        client.get("/sample/orders.csv").get_data(as_text=True)
        if text is None
        else text
    )
    return client.post(
        "/api/pretrade/check",
        data={"file": (io.BytesIO(text.encode()), "o.csv"), **form},
        content_type="multipart/form-data",
    )


def test_my_orders_are_checked_and_the_log_verifies(client):
    r = _orders(client)
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["n"] == 60
    assert sum(j["counts"].values()) == 60 and j["approved"] == j["counts"]["ok"]
    assert (
        len(j["counts"]) >= 3
        and j["log"]["records"] == 60
        and j["log"]["verified"] is True
    )
    assert (
        j["limits"]["max_size"] == 300.0
        and len(j["results"]) == 60
        and not j["truncated"]
    )
    rid = j["run_id"]
    rows = list(
        csv.reader(
            io.StringIO(client.get(f"/api/pretrade/{rid}.csv").get_data(as_text=True))
        )
    )
    assert rows[0][-2:] == ["approved", "reason"] and len(rows) == 61
    log = client.get(f"/api/pretrade/{rid}.jsonl")
    lines = log.get_data(as_text=True).splitlines()
    assert len(lines) == 60 and json.loads(lines[-1])["hash"] == j["log"]["last_hash"]
    assert "attachment" in log.headers["Content-Disposition"]


def test_the_limits_of_the_form_replace_those_of_the_config(client):
    text = "account,side,size,price\nA,buy,50,10\nA,sell,20,10\n"
    base = _orders(client, text).get_json()
    assert base["counts"] == {"ok": 2}
    tight = _orders(client, text, max_size="30").get_json()
    assert tight["limits"]["max_size"] == 30.0 and tight["counts"] == {
        "size": 1,
        "ok": 1,
    }
    js = client.post(
        "/api/pretrade/check", json={"csv": text, "limits": {"max_size": 30}}
    ).get_json()
    assert js["counts"] == {"size": 1, "ok": 1} and js["name"] == "pasted.csv"


def test_orders_that_cannot_be_used_say_which_row_or_limit(client):
    r = _orders(client, "account,side,size,price\nA,buy,1,1\nA,hold,1,1\n")
    assert r.status_code == 400 and "row 3: the side 'hold'" in r.get_json()["error"]
    assert (
        "unknown limit"
        in client.post(
            "/api/pretrade/check",
            json={"csv": "account,side,size,price\nA,buy,1,1", "limits": {"nope": 1}},
        ).get_json()["error"]
    )
    assert "above 0" in _orders(client, None, max_size="-1").get_json()["error"]
    assert "whole number" in _orders(client, None, max_orders="2.5").get_json()["error"]
    assert (
        client.post("/api/pretrade/check")
        .get_json()["error"]
        .startswith("send a CSV file")
    )


def test_long_results_are_cut_in_the_answer_but_not_in_the_download(client):
    n = 620
    text = "account,side,size,price\n" + "\n".join(f"A{i},buy,1,10" for i in range(n))
    j = _orders(client, text).get_json()
    assert j["n"] == n and len(j["results"]) == 500 and j["truncated"] is True
    rows = list(
        csv.reader(
            io.StringIO(
                client.get(f"/api/pretrade/{j['run_id']}.csv").get_data(as_text=True)
            )
        )
    )
    assert len(rows) == n + 1


def test_downloads_of_a_check_that_does_not_exist(client):
    assert client.get("/api/pretrade/zz.csv").status_code == 404
    assert client.get("/api/pretrade/" + "0" * 12 + ".jsonl").status_code == 404
    assert client.get("/api/pretrade/" + "0" * 12 + ".txt").status_code == 404


def test_the_schedule_of_the_config_and_of_my_own_numbers(client):
    flat = client.post("/api/schedule", json={}).get_json()
    assert (
        flat["ok"]
        and flat["total"] == 100.0
        and flat["n"] == 4
        and flat["flat_history"] is True
    )
    assert flat["twap"] == pytest.approx([25.0] * 4) and flat["vwap"] == pytest.approx(
        [25.0] * 4
    )
    mine = client.post(
        "/api/schedule",
        json={"total": "1000", "n": 4, "history": "10,20,30,40\n10,20,30,40"},
    ).get_json()
    assert mine["twap"] == pytest.approx([250.0] * 4) and mine["vwap"] == pytest.approx(
        [100, 200, 300, 400]
    )
    assert mine["sessions"] == 2 and mine["flat_history"] is False
    rows = list(csv.reader(io.StringIO(mine["csv"])))
    assert rows[0] == ["bucket", "twap", "vwap"] and rows[4] == [
        "3",
        "250.000000",
        "400.000000",
    ]
    lists = client.post(
        "/api/schedule", json={"total": 10, "n": 2, "history": [[1, 3], [1, 3]]}
    ).get_json()
    assert lists["vwap"] == pytest.approx([2.5, 7.5])


def test_a_schedule_that_cannot_be_made_says_why(client):
    def bad(body):
        r = client.post("/api/schedule", json=body)
        assert r.status_code == 400
        return r.get_json()["error"]

    assert "from 1 to 1000" in bad({"n": 0})
    assert "from 1 to 1000" in bad({"n": 2.5})
    assert "from 1 to 1000" in bad({"n": "x"})
    assert "0 or more" in bad({"total": -5})
    assert "must be a number" in bad({"total": "abc"})
    assert "session 2 of the history has 3 buckets; the plan has 4" in bad(
        {"history": "1,2,3,4\n1,2,3"}
    )
    assert "must be a number" in bad({"history": "1,2,3,4\n1,x,3,4"})
    assert "above 0" in bad({"history": [[1, 0, 1, 1]]})
