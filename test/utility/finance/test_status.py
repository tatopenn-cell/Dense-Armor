"""Tests for the status command."""

import json
import sys
from pathlib import Path

import pytest

from dense_armor.utility.finance.decision_log import DecisionLog
from dense_armor.utility.finance.status import main

skip_py310 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tomllib needs Python 3.11 or later"
)

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


@skip_py310
def test_1_valid_configuration_subprocess(tmp_path):
    cfg = _write(tmp_path)
    r = subprocess_run(["--config", str(cfg)])
    assert r.returncode == 0
    assert r.stderr == ""
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["emergency_stop"] is False
    assert payload["log"]["exists"] is False
    assert payload["log"]["records"] == 0
    assert payload["log"]["last_hash"] is None
    assert payload["log"]["verified"] is None
    assert payload["feed"]["exists"] is False


@skip_py310
def test_2_invalid_configuration_inprocess(tmp_path, capsys):
    body = BASE_TOML.replace("max_size = 10.0\n", "")
    cfg = _write(tmp_path, body=body)
    code = main(["--config", str(cfg)])
    assert code == 2
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is False
    assert "pretrade.max_size" in payload["error"]


@skip_py310
def test_3_emergency_stop_file_present(tmp_path):
    cfg = _write(tmp_path)
    (tmp_path / "STOP").write_bytes(b"")
    code = main(["--config", str(cfg)])
    assert code == 0
    r = subprocess_run(["--config", str(cfg)])
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["emergency_stop"] is True


@skip_py310
def test_4_log_written_and_after_deletion(tmp_path):
    cfg = _write(tmp_path)
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append(1.0, {"a": 1})
    log.append(2.0, {"a": 2})
    log.append(3.0, {"a": 3})

    r = subprocess_run(["--config", str(cfg)])
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["log"]["exists"] is True
    assert payload["log"]["records"] == 3
    assert payload["log"]["verified"] is True
    assert isinstance(payload["log"]["last_hash"], str)
    assert len(payload["log"]["last_hash"]) == 64

    lines = (
        (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    )
    (tmp_path / "log.jsonl").write_text(
        "".join(lines[:-1]), encoding="utf-8", newline="\n"
    )
    r = subprocess_run(["--config", str(cfg)])
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["log"]["records"] == 2
    assert payload["log"]["verified"] is True


def subprocess_run(args):
    import subprocess as _sp

    return _sp.run(
        [sys.executable, "-m", "dense_armor.utility.finance.status", *args],
        capture_output=True,
        text=True,
    )


def test_5_missing_config_exits_two(capsys):
    code = main([])
    assert code == 2
    err = capsys.readouterr().err
    assert "--config" in err


@skip_py310
def test_6_broken_log_middle_gives_verified_false(tmp_path):
    """A record edited in the middle gives verified False and exit 0."""
    cfg = _write(tmp_path)
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append(1.0, {"a": 1})
    log.append(2.0, {"a": 2})
    log.append(3.0, {"a": 3})
    lines = (
        (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    )
    (tmp_path / "log.jsonl").write_text(
        lines[0] + lines[1].replace('"a":2', '"a":99') + lines[2],
        encoding="utf-8",
        newline="\n",
    )
    r = subprocess_run(["--config", str(cfg)])
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["log"]["exists"] is True
    assert payload["log"]["records"] == 3
    assert payload["log"]["verified"] is False
