"""Test for the script that measures generate with and without the cache."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "neural" / "kv_cache.py"


def test_smoke_writes_every_field_and_both_paths_agree(tmp_path):
    out = tmp_path / "kv.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke", "--json", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    row = json.loads(out.read_text(encoding="utf-8"))
    fields = (
        "backend",
        "device_kind",
        "versions",
        "config",
        "n_params",
        "prompt_len",
        "new_tokens",
        "repeats",
        "cache",
        "no_cache",
        "speedup_median",
        "same_tokens",
    )
    assert all(k in row for k in fields)
    assert row["same_tokens"] is True
    assert (row["prompt_len"], row["new_tokens"], row["repeats"]) == (8, 8, 1)
    assert row["n_params"] == 1040 + (16 + 4 * 16 * 16 + 16 + 3 * 16 * 64) + 16 + 1105
    c = row["cache"]
    assert c["range_s"][0] <= c["median_s"] <= c["range_s"][1]
    assert row["speedup_median"] > 0.0
