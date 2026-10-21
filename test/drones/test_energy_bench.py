"""Tests for edge.energy_bench: helper math, source factory, run, main."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edge import energy_bench as eb


def _args(**kw):
    base = dict(smoke=True, power_node=None, dt_s=0.1, idle_s=0.0, out=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a12_perfect_separation():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.5, 0.6])
    assert eb._a12(a, b) == 0.0
    assert eb._a12(b, a) == 1.0


def test_a12_identical_is_half():
    a = np.array([1.0, 2.0, 3.0])
    assert eb._a12(a, a) == 0.5


def test_a12_ci_brackets_the_point():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 30)
    b = rng.normal(1.0, 1.0, 30)
    lo, hi = eb._a12_ci(a, b, np.random.default_rng(0))
    assert 0.0 <= lo <= hi <= 1.0


def test_wilcoxon_needs_six_pairs():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert eb._wilcoxon(a, b) is None


def test_wilcoxon_with_six_pairs_returns_float():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    p = eb._wilcoxon(a, b)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_wilcoxon_constant_differences_returns_none():
    a = np.array([1.0] * 8)
    b = np.array([1.0] * 8)
    assert eb._wilcoxon(a, b) is None


def test_wilcoxon_note_too_few_pairs():
    assert "fewer than" in eb._wilcoxon_note(None, 3)


def test_wilcoxon_note_constant_differences():
    assert "zero" in eb._wilcoxon_note(None, 10)


def test_wilcoxon_note_computed():
    assert "paired by seed" in eb._wilcoxon_note(0.5, 10)


def test_module_runs_as_main(tmp_path):
    import subprocess
    out = tmp_path / "energy.json"
    r = subprocess.run(
        [sys.executable, "-m", "edge.energy_bench", "--smoke", "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert out.exists()


def test_board_model_on_non_jetson(monkeypatch, tmp_path):
    target = tmp_path / "missing"
    monkeypatch.setattr(eb, "Path", lambda p: target if p == "/proc/device-tree/model" else Path(p))
    assert eb._board_model() is None


def test_board_model_reads_the_file(monkeypatch, tmp_path):
    f = tmp_path / "model"
    f.write_text("NVIDIA Jetson AGX Orin Developer Kit\x00", encoding="utf-8")
    real_path = Path

    def fake_path(p):
        if p == "/proc/device-tree/model":
            return f
        return real_path(p)

    monkeypatch.setattr(eb, "Path", fake_path)
    assert eb._board_model() == "NVIDIA Jetson AGX Orin Developer Kit"


def test_board_model_oserror_returns_none(monkeypatch, tmp_path):
    f = tmp_path / "model"
    f.write_text("x", encoding="utf-8")
    real_path = Path

    def fake_path(p):
        if p == "/proc/device-tree/model":
            return f
        return real_path(p)

    monkeypatch.setattr(eb, "Path", fake_path)
    monkeypatch.setattr(type(f), "read_text",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert eb._board_model() is None


def test_summarise_medians_and_iqr():
    runs = [
        eb._Run(wall_s=1.0, cpu_s=1.0, energy_j=1.0, j_per_1000=0.1, n_samples=10),
        eb._Run(wall_s=2.0, cpu_s=2.0, energy_j=2.0, j_per_1000=0.2, n_samples=10),
        eb._Run(wall_s=3.0, cpu_s=3.0, energy_j=3.0, j_per_1000=0.3, n_samples=10),
        eb._Run(wall_s=4.0, cpu_s=4.0, energy_j=4.0, j_per_1000=0.4, n_samples=10),
    ]
    s = eb._summarise(runs, "x")
    assert s["name"] == "x"
    assert s["wall_s_median"] == 2.5
    assert s["cpu_s_median"] == 2.5
    assert s["energy_j_median"] == 2.5
    assert s["j_per_1000_median"] == pytest.approx(0.25)
    assert s["j_per_1000_iqr"] > 0.0


def test_summarise_all_nan_gives_nan():
    runs = [
        eb._Run(wall_s=1.0, cpu_s=1.0, energy_j=float("nan"),
                j_per_1000=float("nan"), n_samples=10),
    ]
    s = eb._summarise(runs, "x")
    assert np.isnan(s["j_per_1000_median"])


def test_energy_per_1000_scales_linearly():
    trace = eb.PowerTrace(t_s=np.array([0.0, 2.0]),
                          mw=np.array([5000.0, 5000.0]))
    val = eb._energy_per_1000(trace, idle_mw=0.0, n_samples=1000)
    assert val == pytest.approx(10.0, abs=1e-9)
    val2 = eb._energy_per_1000(trace, idle_mw=0.0, n_samples=500)
    assert val2 == pytest.approx(20.0, abs=1e-9)


def test_energy_per_1000_subtracts_idle():
    trace = eb.PowerTrace(t_s=np.array([0.0, 2.0]),
                          mw=np.array([5000.0, 5000.0]))
    val = eb._energy_per_1000(trace, idle_mw=3000.0, n_samples=1000)
    assert val == pytest.approx(4.0, abs=1e-9)


def test_idle_zero_returns_zero():
    mw, note = eb._idle_mw(lambda: None, 0.0)
    assert mw == 0.0
    assert "idle_s" in note


def test_idle_with_synthetic_source():
    def factory():
        return eb.SyntheticPowerSource(mw=4321.0, dt_s=0.01)

    mw, note = eb._idle_mw(factory, 0.05)
    assert mw == pytest.approx(4321.0, abs=1.0)
    assert "0.05" in note


def test_make_source_factory_smoke():
    f, kind = eb._make_source_factory(_args(smoke=True))
    assert kind == "synthetic"
    trace = f().start().stop()
    assert np.all(trace.mw == eb.SMOKE_MW)


def test_make_source_factory_power_node(tmp_path):
    import time as _time
    p = tmp_path / "power1_input"
    p.write_text("1234\n", encoding="utf-8")
    f, kind = eb._make_source_factory(_args(smoke=False, power_node=str(p)))
    assert kind == "sysfs"
    src = f().start()
    _time.sleep(0.35)
    trace = src.stop()
    assert trace.mw.size >= 2
    assert np.all(trace.mw == 1234.0)


def test_make_source_factory_none():
    f, kind = eb._make_source_factory(_args(smoke=False))
    assert f is None
    assert kind is None


def test_make_source_factory_conflict():
    with pytest.raises(SystemExit):
        eb._make_source_factory(_args(smoke=True, power_node="/x"))


def test_run_smoke_synthetic():
    payload = eb.run(_args(smoke=True))
    assert payload["power"] == "synthetic"
    assert payload["n_seeds"] == eb.SMOKE_SEEDS
    assert payload["n_samples"] == eb.SMOKE_SAMPLES
    assert payload["board"] is None
    assert payload["comparison"]["wilcoxon_p"] is None
    assert "fewer than" in payload["comparison"]["wilcoxon_note"]
    a12 = payload["comparison"]["a12_batch_over_online"]
    assert 0.0 <= a12 <= 1.0
    assert payload["batch"]["j_per_1000_median"] > 0.0
    assert payload["online"]["j_per_1000_median"] > 0.0


def test_run_without_source_has_null_energy():
    payload = eb.run(_args(smoke=False, power_node=None))
    assert payload["power"] is None
    assert payload["batch"]["energy_j_median"] is None
    assert payload["batch"]["j_per_1000_median"] is None
    assert payload["online"]["energy_j_median"] is None
    assert payload["online"]["j_per_1000_median"] is None
    assert payload["comparison"]["wilcoxon_p"] is None


def test_run_without_source_keeps_wall_and_cpu():
    payload = eb.run(_args(smoke=False, power_node=None))
    assert payload["batch"]["wall_s_median"] > 0.0
    assert payload["batch"]["cpu_s_median"] > 0.0
    assert payload["online"]["wall_s_median"] > 0.0


def test_main_smoke_writes_json(tmp_path, monkeypatch, capsys):
    out = tmp_path / "energy.json"
    monkeypatch.setattr(
        sys, "argv",
        ["edge.energy_bench", "--smoke", "--out", str(out)])
    rc = eb.main()
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["power"] == "synthetic"
    assert payload["n_seeds"] == eb.SMOKE_SEEDS
    captured = capsys.readouterr().out
    assert "synthetic" in captured
    assert "BatchDecisionTreeClassifier" in captured


def test_main_without_source_writes_json(tmp_path, monkeypatch):
    out = tmp_path / "energy.json"
    monkeypatch.setattr(
        sys, "argv",
        ["edge.energy_bench", "--out", str(out)])
    rc = eb.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["power"] is None
    assert payload["batch"]["j_per_1000_median"] is None


def test_default_out_uses_hostname(monkeypatch):
    monkeypatch.setattr(eb.socket, "gethostname", lambda: "my-host")
    p = eb._default_out()
    assert p.name == "energy_bench_my-host.json"
    assert p.parent == Path("benchmarks/drones")
