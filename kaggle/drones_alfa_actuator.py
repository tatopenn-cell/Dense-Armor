"""Kaggle kernel: run benchmarks/drones/alfa_actuator.py on the ALFA data.

The SMOKE switch runs the first two flights; the full run is the owner's.
The JSON is written to /kaggle/working/alfa_actuator.json.
"""

import os
import subprocess
import sys
from pathlib import Path

BRANCH = "taskD2-drones"
SMOKE = os.environ.get("SMOKE", "0") == "1"
REPO = Path("/kaggle/working/Dense-Armor")
DATA = Path("/kaggle/working/alfa/processed/processed")
OUT = Path("/kaggle/working/alfa_actuator.json")

if not REPO.exists():
    subprocess.run(["git", "clone",
                    "https://github.com/tatopenn-cell/Dense-Armor.git",
                    str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", BRANCH],
                   check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                f"{REPO}[drone]"], check=True)

if not DATA.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sL", "-o", "/kaggle/working/processed.zip",
                    "https://ndownloader.figshare.com/files/24095870"],
                   check=True)
    subprocess.run(["unzip", "-o", "-q", "/kaggle/working/processed.zip",
                    "-d", "/kaggle/working/alfa"], check=True)

cmd = [sys.executable, str(REPO / "benchmarks/drones/alfa_actuator.py"),
       "--data", str(DATA), "--out", str(OUT)]
if SMOKE:
    cmd.append("--smoke")
print("$", " ".join(cmd))
r = subprocess.run(cmd, check=False)
print("exit:", r.returncode)
