"""Tests for the Flask dashboard: status, emergency stop, SVG."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from dashboard.app import create_app, main
from dense_armor.utility.finance.decision_log import DecisionLog


@pytest.fixture(autouse=True)
def _require_flask():
    pytest.importorskip("flask", reason="flask not installed")
    pytest.importorskip("tomllib", reason="load_config needs tomllib, Python 3.11 or later")

BASE_TOML = """\
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


def _write(tmp_path: Path, body: str = BASE_TOML) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(body, encoding="utf-8", newline="\n")
    return p


def test_1_get_status(tmp_path):
    cfg = _write(tmp_path)
    app = create_app(str(cfg))
    client = app.test_client()
    r = client.get("/status")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    payload = json.loads(r.data)
    assert payload["ok"] is True
    assert payload["emergency_stop"] is False
    assert payload["config"] == str(cfg)


def test_2_post_emergency_stop(tmp_path):
    cfg = _write(tmp_path)
    app = create_app(str(cfg))
    client = app.test_client()
    r = client.post("/emergency-stop")
    assert r.status_code == 200
    assert (tmp_path / "STOP").exists()
    r2 = client.post("/emergency-stop")
    assert r2.status_code == 200
    files = list(tmp_path.glob("STOP*"))
    assert len(files) == 1


def test_3_status_after_post(tmp_path):
    cfg = _write(tmp_path)
    app = create_app(str(cfg))
    client = app.test_client()
    client.post("/emergency-stop")
    r = client.get("/status")
    payload = json.loads(r.data)
    assert payload["emergency_stop"] is True


def test_4_get_emergency_stop_is_405(tmp_path):
    cfg = _write(tmp_path)
    app = create_app(str(cfg))
    client = app.test_client()
    r = client.get("/emergency-stop")
    assert r.status_code == 405


def test_5_decisions_svg(tmp_path):
    cfg = _write(tmp_path)
    log = DecisionLog(tmp_path / "log.jsonl")
    for _ in range(5):
        log.append(1.0, {"reason": "ok"})
    for _ in range(2):
        log.append(2.0, {"reason": "size"})
    log.append(3.0, {"reason": "rate"})
    app = create_app(str(cfg))
    client = app.test_client()
    r = client.get("/decisions.svg")
    assert r.status_code == 200
    assert r.mimetype == "image/svg+xml"
    root = ElementTree.fromstring(r.data)
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    assert len(rects) == 3
    counts = {r.get("data-reason"): int(r.get("data-count")) for r in rects}
    assert counts == {"ok": 5, "size": 2, "rate": 1}


def test_6_garbage_and_no_log(tmp_path):
    cfg = _write(tmp_path)
    log = tmp_path / "log.jsonl"
    log.write_text(
        "not json\n"
        '{"seq":0,"t":1.0,"payload":{"no":"reason"},"prev":"0"*0,"hash":"x"}\n',
        encoding="utf-8",
    )
    app = create_app(str(cfg))
    client = app.test_client()
    r = client.get("/decisions.svg")
    assert r.status_code == 200
    root = ElementTree.fromstring(r.data)
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    assert rects == []

    other = tmp_path / "other"
    other.mkdir()
    cfg2 = _write(other)
    app2 = create_app(str(cfg2))
    r2 = app2.test_client().get("/decisions.svg")
    root2 = ElementTree.fromstring(r2.data)
    assert root2.findall(".//{http://www.w3.org/2000/svg}rect") == []


def test_7_invalid_config(tmp_path):
    body = BASE_TOML.replace("max_size = 10.0\n", "")
    cfg = _write(tmp_path, body=body)
    with pytest.raises(ValueError):
        create_app(str(cfg))

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["--config", str(cfg)])
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is False


def test_8_blank_lines_and_bad_payload(tmp_path):
    cfg = _write(tmp_path)
    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n"
        '{"payload":{"reason":"ok"}}\n'
        '{"payload":"not a dict"}\n'
        '{"payload":{"reason":42}}\n'
        '{"payload":{"reason":"size"}}\n',
        encoding="utf-8",
    )
    app = create_app(str(cfg))
    r = app.test_client().get("/decisions.svg")
    root = ElementTree.fromstring(r.data)
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    counts = {r.get("data-reason"): int(r.get("data-count")) for r in rects}
    assert counts == {"ok": 1, "size": 1}


def test_9_main_without_config_exits_two():
    code = main([])
    assert code == 2


def test_10_main_with_valid_config_and_mocked_run(tmp_path, monkeypatch):
    cfg = _write(tmp_path)
    app = create_app(str(cfg))
    monkeypatch.setattr(app, "run", lambda host, port: None)
    monkeypatch.setattr(
        "dashboard.app.create_app", lambda p, **kw: app
    )
    code = main(["--config", str(cfg), "--port", "9999"])
    assert code == 0


def test_11_main_guard():
    import runpy

    try:
        runpy.run_module("dashboard.app", run_name="__main__")
    except SystemExit as e:
        assert e.code == 2


def test_12_stop_file_in_missing_subfolder(tmp_path):
    """paths.emergency_stop with a folder that does not exist yet."""
    body = BASE_TOML.replace(
        'emergency_stop = "STOP"', 'emergency_stop = "deep/nested/STOP"'
    )
    cfg = _write(tmp_path, body=body)
    app = create_app(str(cfg))
    r = app.test_client().post("/emergency-stop")
    assert r.status_code == 200
    assert (tmp_path / "deep" / "nested" / "STOP").exists()

def test_13_decisions_svg_escapes_the_reason(tmp_path):
    cfg = _write(tmp_path)
    DecisionLog(tmp_path / "log.jsonl").append(1.0, {"reason": 'a"<b>&'})
    r = create_app(str(cfg)).test_client().get("/decisions.svg")
    root = ElementTree.fromstring(r.data)
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    assert [x.get("data-reason") for x in rects] == ['a"<b>&']
    assert b"<b>" not in r.data


def test_14_emergency_stop_refuses_another_origin(tmp_path):
    cfg = _write(tmp_path)
    client = create_app(str(cfg)).test_client()
    r = client.post("/emergency-stop", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert not (tmp_path / "STOP").exists()
    r2 = client.post("/emergency-stop", headers={"Origin": "http://localhost"})
    assert r2.status_code == 200
    assert (tmp_path / "STOP").exists()


def test_15_stop_and_status_read_the_configuration_at_each_request(tmp_path):
    cfg = _write(tmp_path)
    client = create_app(str(cfg)).test_client()
    cfg.write_text(BASE_TOML.replace('emergency_stop = "STOP"', 'emergency_stop = "STOP2"'), encoding="utf-8")
    assert client.post("/emergency-stop").status_code == 200
    assert (tmp_path / "STOP2").exists()
    assert not (tmp_path / "STOP").exists()
    assert client.get("/status").get_json()["emergency_stop"] is True


def test_16_playground_is_disabled_unless_asked(tmp_path):
    client = create_app(str(_write(tmp_path))).test_client()
    r = client.post("/api/playground/run", json={"code": "print(1)"})
    assert r.status_code == 403
    assert r.get_json()["ok"] is False and "disabled" in r.get_json()["error"]


def test_17_playground_runs_code_without_the_secrets_of_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSE_ARMOR_SECRET_TEST", "s3cr3t")
    client = create_app(str(_write(tmp_path)), enable_playground=True).test_client()
    code = "import os\nprint('seen=' + str(os.environ.get('DENSE_ARMOR_SECRET_TEST')))"
    r = client.post("/api/playground/run", json={"code": code, "params": {"n": 100}})
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["ok"] is True, body
    assert "seen=None" in body["stdout"]


def test_18_playground_refuses_empty_long_and_bad_input(tmp_path):
    client = create_app(str(_write(tmp_path)), enable_playground=True).test_client()
    assert client.post("/api/playground/run", json={"code": "  "}).status_code == 400
    assert client.post("/api/playground/run", json={"code": "x" * 20001}).status_code == 400
    assert client.post("/api/playground/run", json={"code": "1", "params": {"n": "abc"}}).status_code == 400
    assert client.post("/api/playground/run", json={"code": "1", "params": {"stream": "zigzag"}}).status_code == 400


def test_19_playground_environment_keeps_only_the_allowed_keys(monkeypatch):
    from dashboard.app import _playground_env

    monkeypatch.setenv("DENSE_ARMOR_PG_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _playground_env("/repo")
    assert "DENSE_ARMOR_PG_URL" not in env
    assert env["PATH"] == "/usr/bin" and env["PYTHONPATH"] == "/repo"


def test_20_main_refuses_the_playground_off_the_loopback(tmp_path, capsys):
    cfg = _write(tmp_path)
    code = main(["--config", str(cfg), "--host", "0.0.0.0", "--enable-playground"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize("path", ["/", "/chain", "/playground", "/learners", "/emergency"])
def test_21_the_pages_answer(tmp_path, path):
    client = create_app(str(_write(tmp_path))).test_client()
    r = client.get(path)
    assert r.status_code == 200
    assert b"<html" in r.data


def test_22_the_chain_runs_and_reports_the_errors_of_its_steps(tmp_path):
    client = create_app(str(_write(tmp_path))).test_client()
    cat = client.get("/api/chain/catalog").get_json()
    assert set(cat) == {"data", "features", "signal", "portfolio", "risk", "execution"}
    r = client.post("/api/chain/run", json={"n": 100})
    out = r.get_json()
    assert r.status_code == 200 and out["ok"] is True, out
    assert set(out["metrics"]["errors"]) == {"features", "signal", "risk"}
    assert all(e["n"] == 0 for e in out["metrics"]["errors"].values()), out["metrics"]["errors"]
    assert out["series"]["accuracy"]
    bad = client.post("/api/chain/run", json={"n": 100, "features": "no_such_piece"}).get_json()
    assert bad["ok"] is False and "no_such_piece" in bad["error"]


def test_23_the_learners_route_runs_and_refuses(tmp_path):
    client = create_app(str(_write(tmp_path))).test_client()
    ok = client.post("/api/learners/run", json={"learner": "HoeffdingTreeClassifier", "n": 100})
    assert ok.status_code == 200 and ok.get_json()["ok"] is True, ok.get_json()
    assert client.post("/api/learners/run", json={"learner": "Nope"}).status_code == 400
    assert client.post("/api/learners/run", json={"stream": "zigzag"}).status_code == 400
    assert client.post("/api/learners/run", json={"n": "abc"}).status_code == 400


def test_24_the_search_finds_a_piece_by_name(tmp_path):
    client = create_app(str(_write(tmp_path))).test_client()
    found = client.get("/api/pieces/search?q=hoeffding").get_json()
    assert any("hoeffding" in (p["name"] + p["id"]).lower() for p in found)
    none = client.get("/api/pieces/search?q=zzzz-no-such-piece").get_json()
    assert none == []
    everything = client.get("/api/pieces/search").get_json()
    assert len(everything) > len(found) > 0
    assert all({"section", "category", "id", "name"} <= set(p) for p in everything)
