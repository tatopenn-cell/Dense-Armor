"""Generate the two panels of docs/assets/jetson_energy/jetson_energy.png.

Panel 1: the LENS power trace (Hines, Milford, Fischer 2025, Zenodo
13999449), whole-board power in mW over the 1583 samples.
Panel 2: the joules per 1000 samples of the batch and online models,
from benchmarks/drones/energy_bench_lens_trace.json.
Run from the repository root: python docs/assets/jetson_energy/make_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TRACE = Path("/content/datasets/Figure3/Figure3/Jetson_LENS_inference.npy")
JSON = ROOT / "benchmarks/drones/energy_bench_lens_trace.json"
OUT = Path(__file__).resolve().parent / "jetson_energy.png"

arr = np.load(TRACE, allow_pickle=True)
mw = np.array([float(d["tot"]["power"]) for d in arr])

with JSON.open() as f:
    j = json.load(f)

models = ["BatchDecisionTreeClassifier (CART)", "HoeffdingTreeClassifier"]
j1000 = [j["batch"]["j_per_1000_median"], j["online"]["j_per_1000_median"]]
iqr = [j["batch"]["j_per_1000_iqr"], j["online"]["j_per_1000_iqr"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

ax1.plot(mw, color="#2c6fa6", linewidth=0.8)
ax1.set_xlabel("sample")
ax1.set_ylabel("whole-board power (mW)")
ax1.set_title("LENS Jetson trace, tot.power", weight="bold", fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.bar(["Batch (CART)", "Online (Hoeffding)"], j1000, yerr=iqr,
        color=["#2c6fa6", "#a63c2c"], capsize=6)
ax2.set_ylabel("J per 1000 samples")
ax2.set_title("energy_bench --trace (10 seeds, p = 0.002)", weight="bold", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
