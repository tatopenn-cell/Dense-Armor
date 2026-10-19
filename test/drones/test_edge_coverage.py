"""Extra coverage for edge/edge_bench.py and pipelines/drones/io.py."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

import edge.edge_bench as eb
from edge.edge_bench import _cpu_temp_c, _make_data


def test_make_data_invalid_kind_raises():
    with pytest.raises((ValueError, KeyError)):
        _make_data("bogus", n=1, seed=0)


def test_cpu_temp_with_value(monkeypatch):
    fake = {"coretemp": [SimpleNamespace(current=45.0)]}
    monkeypatch.setattr(eb.psutil, "sensors_temperatures", lambda: fake)
    assert _cpu_temp_c() == 45.0


def test_cpu_temp_empty(monkeypatch):
    monkeypatch.setattr(eb.psutil, "sensors_temperatures", dict)
    assert _cpu_temp_c() is None


def test_cpu_temp_raises(monkeypatch):
    def fail():
        raise AttributeError("no sensors")
    monkeypatch.setattr(eb.psutil, "sensors_temperatures", fail)
    assert _cpu_temp_c() is None


def test_edge_bench_main_block(tmp_path):
    out = tmp_path / "bench.json"
    r = subprocess.run(
        [sys.executable, "edge/edge_bench.py", "--smoke",
         "--out", str(out)],
        check=False, capture_output=True, text=True, timeout=180,
        cwd="/content/Dense-Armor",
    )
    assert r.returncode == 0
    assert out.exists()


def test_import_pymavlink_success(monkeypatch):
    import types
    fake = types.ModuleType("pymavlink")
    fake.mavutil = object()
    monkeypatch.setitem(sys.modules, "pymavlink", fake)
    from pipelines.drones.io import _import_pymavlink
    assert _import_pymavlink() is fake.mavutil


def test_make_data_unknown_kind_reaches_raise(monkeypatch):
    import edge.edge_bench as eb
    fake = dict(eb.PIPELINES)
    fake["__bogus__"] = (lambda: None, "unknown_kind_xyz")
    monkeypatch.setattr(eb, "PIPELINES", fake)
    with pytest.raises(ValueError, match="unknown kind"):
        eb._make_data("__bogus__", n=1, seed=0)


def test_cpu_temp_empty_list(monkeypatch):
    import edge.edge_bench as eb
    monkeypatch.setattr(eb.psutil, "sensors_temperatures",
                        lambda: {"coretemp": []})
    assert eb._cpu_temp_c() is None
