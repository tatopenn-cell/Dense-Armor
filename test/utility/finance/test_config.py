"""Tests for load_config and the frozen dataclasses of the configuration."""

import dataclasses
import sys
from pathlib import Path

import pytest

from dense_armor.utility.finance.config import load_config

PY310 = sys.version_info < (3, 11)
skip_py310 = pytest.mark.skipif(PY310, reason="tomllib needs Python 3.11 or later")

BASE_TOML = """\
[schedule]
total = 100.0
n = 4

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


def _write(tmp_path, body=BASE_TOML, name="c.toml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8", newline="\n")
    return p


@skip_py310
def test_complete_file_reads_back_every_value(tmp_path):
    c = load_config(_write(tmp_path))
    assert c.schedule.total == 100.0
    assert c.schedule.n == 4
    assert c.pretrade.max_notional == 1_000_000.0
    assert c.pretrade.max_size == 10.0
    assert c.pretrade.price_band == 0.01
    assert c.pretrade.max_orders == 5
    assert c.pretrade.window_s == 1.0
    assert c.pretrade.duplicate_window_s == 0.1
    assert c.ring.capacity == 3
    assert c.ring.width == 2
    assert c.paths.log == tmp_path / "log.jsonl"
    assert c.paths.feed == tmp_path / "feed.bin"
    assert c.paths.emergency_stop == tmp_path / "STOP"


@skip_py310
def test_result_is_frozen(tmp_path):
    c = load_config(_write(tmp_path))
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.schedule.total = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.pretrade.max_size = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.ring.capacity = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.paths.log = Path("/x")  # type: ignore[misc]


@skip_py310
def test_relative_path_resolved_against_file_folder(tmp_path):
    c = load_config(_write(tmp_path))
    assert c.paths.log == tmp_path / "log.jsonl"
    assert c.paths.feed == tmp_path / "feed.bin"
    assert c.paths.emergency_stop == tmp_path / "STOP"


@skip_py310
def test_absolute_path_kept(tmp_path):
    abs_log = str(tmp_path / "elsewhere" / "log.jsonl")
    body = BASE_TOML.replace('log = "log.jsonl"', f"log = '{abs_log}'")
    c = load_config(_write(tmp_path, body=body))
    assert c.paths.log == Path(abs_log)


@skip_py310
def test_missing_table(tmp_path):
    body = BASE_TOML.replace("[ring]\ncapacity = 3\nwidth = 2\n\n", "")
    with pytest.raises(ValueError, match="missing table: ring"):
        load_config(_write(tmp_path, body=body))


@skip_py310
def test_missing_key_names_the_key(tmp_path):
    body = BASE_TOML.replace("max_size = 10.0\n", "")
    with pytest.raises(ValueError, match=r"pretrade\.max_size"):
        load_config(_write(tmp_path, body=body))


@skip_py310
def test_unknown_key_and_unknown_table(tmp_path):
    body = BASE_TOML.replace("n = 4", "n = 4\nmystery = 1")
    with pytest.raises(ValueError, match=r"unknown key: schedule\.mystery"):
        load_config(_write(tmp_path, body=body))
    body = BASE_TOML + "\n[extra]\nx = 1\n"
    with pytest.raises(ValueError, match="unknown table: extra"):
        load_config(_write(tmp_path, body=body))


@skip_py310
@pytest.mark.parametrize(
    "old,new,match",
    [
        ("max_size = 10.0", 'max_size = "big"', r"pretrade\.max_size"),
        ("max_size = 10.0", "max_size = true", r"pretrade\.max_size"),
        ("max_notional = 1000000.0", "max_notional = true", r"pretrade\.max_notional"),
        ("max_orders = 5", "max_orders = true", r"pretrade\.max_orders"),
        ("n = 4", "n = 2.5", r"schedule\.n"),
        ("n = 4", "n = true", r"schedule\.n"),
        ('log = "log.jsonl"', "log = 42", r"paths\.log"),
    ],
)
def test_wrong_type(tmp_path, old, new, match):
    body = BASE_TOML.replace(old, new)
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, body=body))


@skip_py310
def test_not_toml(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, body="not a toml file at all ="))


@skip_py310
def test_out_of_range_names_the_table(tmp_path):
    body = BASE_TOML.replace("max_orders = 5", "max_orders = 0")
    with pytest.raises(ValueError, match=r"^\[pretrade\]"):
        load_config(_write(tmp_path, body=body))
    body = BASE_TOML.replace("max_notional = 1000000.0", "max_notional = nan")
    with pytest.raises(ValueError, match=r"^\[pretrade\]"):
        load_config(_write(tmp_path, body=body))
    body = BASE_TOML.replace("n = 4", "n = 0")
    with pytest.raises(ValueError, match=r"^\[schedule\]"):
        load_config(_write(tmp_path, body=body))
    body = BASE_TOML.replace("width = 2", "width = 0")
    with pytest.raises(ValueError, match=r"^\[ring\]"):
        load_config(_write(tmp_path, body=body))


@skip_py310
def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_tomllib_missing_is_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    with pytest.raises(RuntimeError, match="tomllib"):
        load_config(tmp_path / "does-not-need-to-exist.toml")


@skip_py310
def test_make_checker_returns_new_checker(tmp_path):
    from dense_armor.utility.finance.pretrade import PreTradeChecker

    c = load_config(_write(tmp_path))
    a = c.make_checker()
    b = c.make_checker()
    assert isinstance(a, PreTradeChecker)
    assert a is not b
    assert a.max_size == c.pretrade.max_size


@skip_py310
def test_make_ring_returns_new_ring(tmp_path):
    from dense_armor.utility.finance.ring import RingBuffer

    c = load_config(_write(tmp_path))
    a = c.make_ring()
    b = c.make_ring()
    assert isinstance(a, RingBuffer)
    assert a is not b
    assert a.capacity == c.ring.capacity


@skip_py310
def test_table_key_that_is_not_a_table(tmp_path):
    body = BASE_TOML.replace(
        "[schedule]\ntotal = 100.0\nn = 4\n",
        'schedule = "not a table"\n',
    )
    with pytest.raises(ValueError, match="schedule must be a table"):
        load_config(_write(tmp_path, body=body))


@skip_py310
def test_integer_accepted_for_number(tmp_path):
    """An integer where a number is expected is accepted and stored as float."""
    body = BASE_TOML.replace("max_size = 10.0", "max_size = 10")
    c = load_config(_write(tmp_path, body=body))
    assert c.pretrade.max_size == 10.0
    assert isinstance(c.pretrade.max_size, float)
