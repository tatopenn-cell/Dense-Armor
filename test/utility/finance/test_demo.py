"""Tests for the demo command."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "finance" / "demo.toml"


def _run(tmp_path: Path, n: int = 200) -> subprocess.CompletedProcess:
    cfg = tmp_path / "demo.toml"
    if not cfg.exists():
        shutil.copy(FIXTURE, cfg)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
            "--n",
            str(n),
        ],
        capture_output=True,
        text=True,
    )


def test_demo_run(tmp_path):
    r = _run(tmp_path, n=200)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["n_messages"] == 200
    assert payload["log"]["verified"] is True
    assert sum(payload["counts"].values()) == payload["n_orders"]
    log_file = tmp_path / "log.jsonl"
    assert log_file.exists()
    n_log_lines = len(log_file.read_text(encoding="utf-8").splitlines())
    assert n_log_lines == payload["n_orders"]

    r2 = _run(tmp_path, n=200)
    assert r2.returncode == 2
    payload2 = json.loads(r2.stdout)
    assert payload2["ok"] is False
    assert payload2["error"] == "log exists"


def test_demo_n_not_integer(tmp_path):
    cfg = tmp_path / "demo.toml"
    shutil.copy(FIXTURE, cfg)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
            "--n",
            "-1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    payload = json.loads(r.stdout.strip())
    assert payload["ok"] is False
    assert "non-negative integer" in payload["error"]


def test_demo_seed_not_integer(tmp_path):
    cfg = tmp_path / "demo.toml"
    shutil.copy(FIXTURE, cfg)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
            "--seed",
            "-1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    payload = json.loads(r.stdout.strip())
    assert payload["ok"] is False


def test_demo_missing_config(tmp_path):
    cfg = tmp_path / "missing.toml"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    payload = json.loads(r.stdout.strip())
    assert payload["ok"] is False


def test_demo_ring_width_must_be_4(tmp_path):
    cfg = tmp_path / "bad_width.toml"
    body = FIXTURE.read_text(encoding="utf-8").replace("width = 4", "width = 2")
    cfg.write_text(body, encoding="utf-8", newline="\n")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
            "--n",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    payload = json.loads(r.stdout.strip())
    assert "ring.width" in payload["error"]


def test_demo_emergency_stop(tmp_path):
    cfg = tmp_path / "demo.toml"
    shutil.copy(FIXTURE, cfg)
    (tmp_path / "STOP").write_bytes(b"")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_armor.utility.finance.demo",
            "--config",
            str(cfg),
            "--n",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 3
    payload = json.loads(r.stdout.strip())
    assert payload["error"] == "emergency stop"
    assert not (tmp_path / "log.jsonl").exists()


def test_demo_main_guard():
    import runpy
    import sys as _sys

    orig = _sys.argv
    try:
        _sys.argv = ["demo", "--config", "/nonexistent/x.toml"]
        try:
            runpy.run_module("dense_armor.utility.finance.demo", run_name="__main__")
        except SystemExit as e:
            assert e.code == 2
    finally:
        _sys.argv = orig


def _fake_itch_two_orders_same_stock():
    """Two A messages of AAPL at 100.0 then 110.0, framed in the F7 container."""

    def _A(order_ref, shares, price_raw):
        b = bytearray(36)
        b[0] = ord("A")
        b[1:3] = (1).to_bytes(2, "big")
        b[3:5] = (0).to_bytes(2, "big")
        b[5:11] = (0).to_bytes(6, "big")
        b[11:19] = order_ref.to_bytes(8, "big")
        b[19] = ord("B")
        b[20:24] = shares.to_bytes(4, "big")
        b[24:32] = b"AAPL    "
        b[32:36] = price_raw.to_bytes(4, "big")
        return bytes(b)

    a = _A(1, 10, 1_000_000)
    d = _A(2, 10, 1_100_000)
    return len(a).to_bytes(2, "big") + a + len(d).to_bytes(2, "big") + d


def test_demo_ref_price_is_first_of_stock(tmp_path, monkeypatch):
    """The ref_price of the second A of AAPL is the price of the first,
    so a 10% move exceeds the price_band and is refused."""
    import contextlib
    import io

    from dense_armor.utility.finance import demo as demo_mod

    cfg = tmp_path / "demo.toml"
    shutil.copy(FIXTURE, cfg)
    fake = _fake_itch_two_orders_same_stock()
    monkeypatch.setattr(demo_mod, "synthetic_itch", lambda n, s: fake)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = demo_mod.main(["--config", str(cfg), "--n", "2"])
    assert code == 0
    payload = json.loads(buf.getvalue().strip())
    # dovrebbe cadere nel price_band se ref_price = 100.0, non se ref_price = 110.0
    counts = payload["counts"]
    reasons = set(counts.keys())
    assert "price_band" in reasons, (
        f"expected price_band in {reasons}: "
        f"the second A must be compared with the price of the first"
    )


def test_demo_verify_call_is_used(tmp_path, monkeypatch):
    """The demo returns DecisionLog.verify()'s result, not a hardcoded True."""
    import contextlib
    import io

    from dense_armor.utility.finance import decision_log as dl_mod
    from dense_armor.utility.finance import demo as demo_mod

    cfg = tmp_path / "demo.toml"
    shutil.copy(FIXTURE, cfg)
    monkeypatch.setattr(dl_mod.DecisionLog, "verify", lambda self: False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = demo_mod.main(["--config", str(cfg), "--n", "20"])
    assert code == 0
    payload = json.loads(buf.getvalue().strip())
    assert payload["log"]["verified"] is False, (
        "the demo must call DecisionLog.verify() and return its result"
    )
