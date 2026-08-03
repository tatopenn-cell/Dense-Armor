# -*- coding: utf-8 -*-
import hashlib
import json

import numpy as np

import dense_armor.core as core_pkg
from dense_armor.core.visualizer import AIEngineVisualizer


def test_engine_signature_usa_la_versione_reale_del_pacchetto():
    assert AIEngineVisualizer.ENGINE_SIGNATURE == f"dense-armor-v{core_pkg.__version__}"


def test_export_provenance_archive_scrive_file_con_hash_coerente(tmp_path):
    viz = AIEngineVisualizer(output_dir=str(tmp_path))
    run_history = [{"step": 1, "value": 0.5}]

    sha256 = viz.export_provenance_archive(run_history, filename="archive.json")

    archive_path = tmp_path / "archive.json"
    assert archive_path.exists()
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    assert payload["records"] == run_history
    assert payload["metadata"]["integrity_sha256"] == sha256
    assert len(sha256) == 64  # hex digest di sha256


def test_export_provenance_archive_hash_cambia_con_dati_diversi(tmp_path):
    viz = AIEngineVisualizer(output_dir=str(tmp_path))
    sha_a = viz.export_provenance_archive([{"a": 1}], filename="a.json")
    sha_b = viz.export_provenance_archive([{"a": 2}], filename="b.json")
    assert sha_a != sha_b


def test_export_trend_report_text_scrive_metriche_leggibili(tmp_path):
    raw = np.random.default_rng(0).normal(size=(3, 20))
    filtered = raw * 0.5
    filename = str(tmp_path / "report.txt")

    AIEngineVisualizer.export_trend_report_text(raw, filtered, filename=filename)

    content = open(filename, encoding="utf-8").read()
    assert "Scenari monitorati:      3" in content
    assert "Passi temporali:         20" in content
    assert "Smorzamento rumore:" in content


def test_export_trend_report_text_varianza_zero_non_solleva_eccezioni(tmp_path):
    raw = np.zeros((2, 5))
    filtered = np.zeros((2, 5))
    filename = str(tmp_path / "report_zero.txt")

    AIEngineVisualizer.export_trend_report_text(raw, filtered, filename=filename)

    content = open(filename, encoding="utf-8").read()
    assert "Smorzamento rumore:      0.00%" in content
