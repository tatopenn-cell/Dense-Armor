"""Tests for the Flask dashboard: status, emergency stop, SVG."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from dense_armor.utility.finance.dashboard import create_app, main
from dense_armor.utility.finance.decision_log import DecisionLog

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
        "dense_armor.utility.finance.dashboard.create_app", lambda p: app
    )
    code = main(["--config", str(cfg), "--port", "9999"])
    assert code == 0


def test_11_main_guard():
    import runpy

    try:
        runpy.run_module("dense_armor.utility.finance.dashboard", run_name="__main__")
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
