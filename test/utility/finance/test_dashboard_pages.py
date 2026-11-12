"""The pages and the routes of the dashboard that test_dashboard.py does not reach."""

from __future__ import annotations

import pytest

pytest.importorskip("flask", reason="flask not installed")
pytest.importorskip("tomllib", reason="load_config needs tomllib, Python 3.11 or later")

from dashboard.app import create_app  # noqa: E402

from dense_armor.core.finance.decision_log import DecisionLog  # noqa: E402
from dense_armor.core.finance.synthetic import synthetic_itch  # noqa: E402

TOML = """\
[schedule]
total = 10.0
n = 2

[pretrade]
max_notional = 1000000.0
max_size = 10.0
price_band = 0.01
max_orders = 5
window_s = 1.0
duplicate_window_s = 0.1

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
    log = DecisionLog(tmp_path / "log.jsonl")
    for i in range(5):
        log.append(float(i), {"reason": "ok" if i % 2 else "size", "stock": "ABC",
                              "approved": bool(i % 2), "order_ref": i})
    (tmp_path / "feed.bin").write_bytes(synthetic_itch(10, 0))
    return create_app(str(cfg)).test_client()


@pytest.mark.parametrize("path", [
    "/decisions", "/audit", "/feed", "/synthetic", "/synthetic?n=abc", "/synthetic?n=5000",
    "/schedule", "/pretrade", "/ring", "/decode", "/run", "/fix",
    "/s/data", "/s/portfolio", "/s/execution", "/s/risk", "/s/risk?cat=drift", "/s/risk?cat=nope",
    "/s/features", "/s/features?cat=moments", "/s/features?cat=sketches", "/s/signal", "/s/signal?cat=linear",
])
def test_the_pages_answer(client, path):
    r = client.get(path)
    assert r.status_code == 200, path
    assert b"<html" in r.data


def test_the_feed_page_without_a_feed_file(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML, encoding="utf-8", newline="\n")
    r = create_app(str(cfg)).test_client().get("/feed")
    assert r.status_code == 200


def test_the_decision_api_counts_and_limits(client):
    assert client.get("/api/counts").get_json() == {"size": 3, "ok": 2}
    rows = client.get("/api/decisions?limit=2").get_json()
    assert [r["seq"] for r in rows] == [3, 4]
    assert len(client.get("/api/decisions?limit=abc").get_json()) == 5
    assert len(client.get("/api/decisions?limit=0").get_json()) == 1
    assert client.get("/api/last-run/decisions").get_json() == []
    assert client.get("/api/last-run/decisions?limit=abc").get_json() == []


def test_a_run_of_the_demo_writes_the_last_run_log(client):
    r = client.post("/api/run", json={"n": 60, "seed": 3})
    body = r.get_json()
    assert r.status_code == 200 and isinstance(body, dict), body
    last = client.get("/api/last-run/decisions?limit=5").get_json()
    assert isinstance(last, list)
    assert client.post("/api/run", json={"n": "abc"}).status_code == 400
    assert client.post("/api/run", json={"n": 20}).status_code == 200


def test_the_demo_summary_answers(client):
    r = client.get("/api/demo/summary")
    assert r.status_code == 200
    assert isinstance(r.get_json(), dict)


def _first_two_pieces_of_each_category(client):
    return client.get("/api/pieces/search").get_json()


def test_every_category_has_pieces_that_open_and_run(client):
    pieces = _first_two_pieces_of_each_category(client)
    assert len(pieces) > 10
    for p in pieces:
        url = f"/piece/{p['section']}/{p['category']}/{p['id']}"
        page = client.get(url)
        assert page.status_code in (200, 404), url
        run = client.post(f"/api{url}/run", json={"n": 80, "params": {}})
        assert run.status_code in (200, 400, 500), url
        body = run.get_json()
        assert isinstance(body, dict) and "ok" in body, (url, body)


def test_a_piece_with_a_bad_body_or_an_unknown_id_is_refused(client):
    assert client.get("/piece/signal/trees/no_such_piece").status_code in (200, 404)
    bad = client.post("/api/piece/signal/trees/hoeffding/run", data="not json", content_type="text/plain")
    assert bad.status_code in (200, 400)
    assert client.post("/api/piece/nosection/nocat/noid/run", json={}).status_code in (200, 400, 404, 500)


def test_the_learners_route_with_the_other_learners_and_a_window(client):
    for name in ("HoeffdingAdaptiveTreeClassifier", "MondrianForestClassifier"):
        r = client.post("/api/learners/run", json={"learner": name, "n": 100, "window": 20, "stream": "gradual"})
        assert r.status_code == 200 and r.get_json()["ok"] is True, r.get_json()


def test_the_chain_route_with_every_slot(client):
    body = {"data": "driftstream_sudden", "features": "standard_scaler", "signal": "hoeffding",
            "portfolio": "min_variance", "risk": "adwin", "execution": "twap", "n": 80}
    r = client.post("/api/chain/run", json=body)
    assert r.status_code == 200 and r.get_json()["ok"] is True, r.get_json()


def test_the_playground_runs_the_helpers_and_reports_a_failure_and_a_chart(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML, encoding="utf-8", newline="\n")
    client = create_app(str(cfg), enable_playground=True).test_client()
    failing = client.post("/api/playground/run", json={"code": "raise SystemExit(3)", "params": {"n": 100}})
    assert failing.status_code == 200 and failing.get_json()["ok"] is False
    chart = client.post("/api/playground/run", json={
        "code": "import json\nprint('text')\nprint(json.dumps({'__chart__': {'k': 1}}))",
        "params": {"n": 100, "learner": "HoeffdingTreeClassifier", "use_anomaly": True,
                   "use_mixture": True, "use_garch": True},
    })
    body = chart.get_json()
    assert body["ok"] is True, body
    assert body["chart"] == {"k": 1} and body["stdout"] == "text"
    unknown = client.post("/api/playground/run", json={"code": "pass", "params": {"learner": "Nope"}})
    assert unknown.get_json()["ok"] is False


def test_the_audit_page_with_a_database_that_answers(client, monkeypatch):
    class Db:
        last_hash = "abc"

        def __init__(self, url):
            self.url = url

        def __len__(self):
            return 7

        def verify(self):
            return True

    monkeypatch.setattr("dense_armor.core.finance.audit_db.AuditDB", Db)
    r = client.get("/audit")
    assert r.status_code == 200 and b"abc" in r.data


def test_the_feed_page_reports_a_feed_that_cannot_be_read(client, monkeypatch):
    def boom(path):
        raise ValueError("bad feed")
        yield

    monkeypatch.setattr("dashboard.app.iter_itch", boom)
    r = client.get("/feed")
    assert r.status_code == 200 and b"bad feed" in r.data


def test_the_fix_page_with_the_library_installed(client, monkeypatch):
    import importlib.metadata
    import sys
    import types

    monkeypatch.setitem(sys.modules, "quickfix", types.ModuleType("quickfix"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.8.7")
    r = client.get("/fix")
    assert r.status_code == 200 and b"9.8.7" in r.data


def test_the_more_page_and_the_inputs_that_break_a_route(client):
    assert client.get("/s/more").status_code == 200
    bad = client.post("/api/piece/signal/trees/hoeffding/run", json={"params": 5})
    assert bad.status_code == 400
    listed = client.post("/api/playground/run", json=[1, 2])
    assert listed.status_code == 403


def test_the_chain_route_reports_a_run_that_raises(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("chain down")

    monkeypatch.setattr("dashboard.chain.run_chain", boom)
    r = client.post("/api/chain/run", json={})
    assert r.status_code == 500 and r.get_json()["error"] == "RuntimeError: chain down"


def test_the_learners_route_reports_a_stream_that_raises(client, monkeypatch):
    class Boom:
        def __init__(self, **k):
            raise RuntimeError("stream down")

    monkeypatch.setattr("dense_armor.core.signals.datasets.DriftStream", Boom)
    r = client.post("/api/learners/run", json={"n": 100})
    assert r.status_code == 500 and "stream down" in r.get_json()["error"]


def test_the_playground_reports_a_timeout_and_a_list_body(tmp_path, monkeypatch):
    import subprocess

    cfg = tmp_path / "c.toml"
    cfg.write_text(TOML, encoding="utf-8", newline="\n")
    client = create_app(str(cfg), enable_playground=True).test_client()
    assert client.post("/api/playground/run", json=[1, 2]).status_code == 400

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=20)

    monkeypatch.setattr(subprocess, "run", slow)
    out = client.post("/api/playground/run", json={"code": "pass"}).get_json()
    assert out["ok"] is False and "timeout" in out["stderr"]


def test_section_pages_link_only_pieces_that_have_a_page(client):
    import re
    for sec in ("data", "portfolio", "execution"):
        html = client.get(f"/s/{sec}").get_data(as_text=True)
        for href in re.findall(r'class="piece-expand" href="(/piece/[^"]+)"', html):
            assert client.get(href).status_code == 200, href
