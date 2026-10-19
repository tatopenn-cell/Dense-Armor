#!/usr/bin/env bash
# Self-test for a Raspberry Pi or a Jetson board.
#
# Prints the board model, the OS and the Python version, installs
# dense-armor[drone] in a virtual environment, runs the smoke benchmark
# and writes the JSON with the model in its file name, reads nvpmodel and
# tegrastats when they exist, and prints the block to paste into the
# issue. On a machine that is not a board it says so for the parts that
# do not apply and runs to the end.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_FILE="/proc/device-tree/model"
if [ -r "$MODEL_FILE" ]; then
    MODEL="$(tr -d '\000' < "$MODEL_FILE")"
else
    MODEL="not a Jetson / Raspberry Pi"
fi
echo "board: $MODEL"

if [ -r /etc/os-release ]; then
    . /etc/os-release
    echo "os: ${PRETTY_NAME:-unknown}"
else
    echo "os: unknown"
fi
echo "python: $(python3 --version 2>&1)"

VENV=".board_selftest_venv"
VENV_OK=1
if [ ! -d "$VENV" ]; then
    if ! python3 -m venv "$VENV" 2>/dev/null; then
        VENV_OK=0
    fi
fi
if [ "$VENV_OK" = 1 ] && [ -f "$VENV/bin/activate" ]; then
    . "$VENV/bin/activate"
    echo "installed: dense-armor[drone] in $VENV"
else
    echo "venv: python3 -m venv failed, installing in the system python"
fi
python -m pip install -q -e ".[drone]"

SLUG="$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"
JSON="edge_bench_${SLUG}.json"
python -m edge.edge_bench --smoke --out "$JSON"
echo "json: $JSON"

if command -v nvpmodel >/dev/null 2>&1; then
    echo "nvpmodel:"
    nvpmodel -q || true
else
    echo "nvpmodel: not a Jetson / Raspberry Pi"
fi

if command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats: 10 s"
    timeout 10 tegrastats --interval 1000 || true
else
    echo "tegrastats: not a Jetson / Raspberry Pi"
fi

echo
echo "--- paste this in the issue ---"
echo "board: $MODEL"
echo "json: $JSON"
if [ "$VENV_OK" != 1 ]; then
    echo "venv: python3 -m venv failed, the benchmark ran on the system python"
fi
python - "$JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for row in payload.get("results", []):
    name = row.get("name", "?")
    med = row.get("ms_per_sample_median", row.get("median_ms"))
    print(f"{name}: median {med} ms/sample")
PY
echo "--- end ---"
