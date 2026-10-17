"""Kaggle kernel: full run of the CASPER hour-episodes benchmark.

Requires the dataset ``hkayan/industrial-robotic-arm-anomaly-detection``.
Run from the repository root in the Kaggle notebook environment.
"""

import subprocess
import sys

r = subprocess.run(
    [sys.executable, "benchmarks/casper_hours_episodes.py",
     "--hours", "0-14", "--minutes", "60.0",
     "--out", "benchmarks/casper_hours_episodes.json"],
    capture_output=True, text=True,
)
print(r.stdout)
print(r.stderr, file=sys.stderr)
raise SystemExit(r.returncode)
