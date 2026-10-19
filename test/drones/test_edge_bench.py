"""Tests for edge/edge_bench.py (task D1, item C5).

The benchmark runs the six pipelines on synthetic data and prints a
table. The tests run --smoke (200 samples, 5 runs) and check the
output, the JSON, and the helper functions.
"""

import json
import sys

from edge.edge_bench import (
    PIPELINES,
    _bench_pipeline,
    _cpu_temp_c,
    _host,
    _make_data,
    _time_one,
    main,
)


def test_make_data_stream():
    for name in ["actuator_fault", "sensor_consistency",
                 "propeller_health", "battery_endurance"]:
        data = _make_data(name, n=10, seed=0)
        assert isinstance(data, list)
        assert len(data) == 10


def test_make_data_scans():
    data = _make_data("safety_filter", n=10, seed=0)
    assert len(data) == 10
    assert hasattr(data[0], "shape")


def test_make_data_packets():
    data = _make_data("link_health", n=10, seed=0)
    assert len(data) == 10
    assert isinstance(data[0], dict)


def test_time_one_returns_ms_per_sample():
    run_fn = PIPELINES["link_health"][0]
    data = _make_data("link_health", n=20, seed=0)
    ms, drss = _time_one(run_fn, data)
    assert ms >= 0
    assert isinstance(drss, float)


def test_bench_pipeline():
    row = _bench_pipeline("link_health", n_samples=20, n_runs=2, seed=0)
    assert row["name"] == "link_health"
    assert row["n_samples"] == 20
    assert row["n_runs"] == 2
    assert row["ms_per_sample_median"] >= 0
    assert row["ms_per_sample_iqr"] >= 0
    assert row["ms_per_sample_p99"] >= 0


def test_cpu_temp_c():
    t = _cpu_temp_c()
    assert t is None or isinstance(t, float)


def test_host():
    h = _host()
    assert "hostname" in h
    assert "cpu_count" in h
    assert "python" in h


def test_main_smoke(tmp_path, monkeypatch):
    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv",
                        ["edge_bench", "--smoke", "--out", str(out)])
    rc = main()
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "results" in payload
    assert len(payload["results"]) == 6
