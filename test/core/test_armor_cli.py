"""Armor: slow-drift detector, readable report and the command line."""

import json
import runpy
import sys

import numpy as np
import pytest

from dense_armor.armor import Armor, main


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["dense_armor", *args])
    main()


def test_deriva_short_series_reports_no_drift():
    assert Armor().deriva(np.arange(5.0)) == (0.0, False, 0.0)


def test_deriva_ignores_non_finite_values():
    serie = np.array([1.0, np.nan, np.inf, 2.0, 3.0])
    assert Armor().deriva(serie) == (0.0, False, 0.0)


def test_deriva_flags_a_slow_ramp():
    rng = np.random.default_rng(0)
    serie = 1.0 + 0.004 * np.arange(300) + rng.normal(0.0, 0.01, 300)
    tasso, significativa, esponente = Armor().deriva(serie)
    assert significativa
    assert tasso == pytest.approx(0.004, rel=0.1)
    assert esponente > 0.0


def test_deriva_flat_series_is_not_significant():
    rng = np.random.default_rng(1)
    serie = 5.0 + rng.normal(0.0, 0.01, 300)
    tasso, significativa, _ = Armor().deriva(serie)
    assert not significativa
    assert abs(tasso) < 1e-3


def test_deriva_constant_series_has_zero_rate():
    tasso, significativa, esponente = Armor().deriva(np.full(30, 2.0))
    assert tasso == 0.0
    assert not significativa
    assert np.isfinite(esponente)


def test_referto_prints_each_anomaly_and_the_drift_line(capsys):
    rng = np.random.default_rng(2)
    serie = 1.0 + 0.004 * np.arange(300) + rng.normal(0.0, 0.01, 300)
    serie[[10, 50, 90]] = 1e6
    anomalie = Armor().referto(serie, nome="rampa")
    out = capsys.readouterr().out
    assert "[ARMATURA] rampa: 300 punti" in out
    assert "DERIVA LENTA" in out
    assert {10, 50, 90} <= set(anomalie)
    assert out.count("-> indice") == min(len(anomalie), 10)


def test_referto_truncates_after_ten_anomalies(capsys):
    serie = np.ones(200)
    serie[np.arange(5, 200, 8)] = np.nan
    anomalie = Armor().referto(serie)
    out = capsys.readouterr().out
    assert len(anomalie) > 10
    assert out.count("-> indice") == 10
    assert f"... e altre {len(anomalie) - 10}" in out


def test_referto_json_marks_non_finite_values_as_strings():
    serie = np.ones(40)
    serie[7] = np.nan
    r = Armor().referto_json(serie, nome="x")
    entry = next(a for a in r["anomalie"] if a["indice"] == 7)
    assert entry["grezzo"] == "NaN"
    assert isinstance(entry["pulito"], float)
    json.dumps(r)


def test_main_without_arguments_prints_usage_and_exits(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch)
    assert exc.value.code == 0
    assert "USO:" in capsys.readouterr().out


def test_main_reads_numbers_from_the_command_line(monkeypatch, capsys):
    _run(monkeypatch, "1.2", "1.3", "9999", "1.25", "nan", "1.3")
    out = capsys.readouterr().out
    assert "serie da riga di comando: 6 punti" in out
    assert "indice 4" in out


def test_main_json_from_the_command_line(monkeypatch, capsys):
    _run(monkeypatch, "--json", "1.0", "2.0", "nan", "3.0")
    r = json.loads(capsys.readouterr().out)
    assert r["nome"] == "serie da riga di comando"
    assert r["punti"] == 4


def test_main_reads_numbers_from_a_file(monkeypatch, capsys, tmp_path):
    f = tmp_path / "dati.txt"
    f.write_text("1.0 2.0 nan 3.0\n-4e-1 NaN 5\n", encoding="utf-8")
    _run(monkeypatch, str(f))
    out = capsys.readouterr().out
    assert "dati.txt: 7 punti" in out
    assert "indice 2" in out and "indice 5" in out


def test_main_json_from_a_file(monkeypatch, capsys, tmp_path):
    f = tmp_path / "dati.txt"
    f.write_text("1 2 3 4 5", encoding="utf-8")
    _run(monkeypatch, "--json", str(f))
    r = json.loads(capsys.readouterr().out)
    assert r["nome"] == "dati.txt"
    assert r["punti"] == 5


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_as_a_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dense_armor", "1.0", "2.0", "3.0"])
    runpy.run_module("dense_armor.armor", run_name="__main__")
    assert "3 punti" in capsys.readouterr().out
