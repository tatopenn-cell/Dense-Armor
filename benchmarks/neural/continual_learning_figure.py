"""Write the figure of the continual-learning page.

Three panels: forgetting per learner, MSE per lifecycle state, exact
recall per lifecycle state. Synthetic data, labels say synthetic.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LEARNERS = ["baseline", "EWC", "GEM", "A-GEM", "DGR", "Archival"]
FORGETTING = [0.324, 0.118, 0.285, 0.283, 0.380, 0.130]

STATES = ["active", "warm", "cold", "archive"]
MSE = [0.0000, 0.0001, 0.0505, 0.9339]
RECALL = [1.0, 1.0, 1.0, 0.0]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].bar(LEARNERS, FORGETTING, color=["gray", "C0", "C2", "C2", "C3", "C1"])
axes[0].set_ylabel("forgetting (median)")
axes[0].set_title("Forgetting, 5 tasks, synthetic")
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(STATES, MSE, color=["C0", "C1", "C2", "C3"])
axes[1].set_ylabel("reconstruction MSE")
axes[1].set_title("ArchivalMemory, MSE per state")
axes[1].set_yscale("symlog", linthresh=1e-4)
axes[1].grid(axis="y", alpha=0.3)

axes[2].bar(STATES, RECALL, color=["C0", "C1", "C2", "C3"])
axes[2].set_ylabel("exact recall")
axes[2].set_ylim(-0.05, 1.05)
axes[2].set_title("ArchivalMemory, exact recall per state")
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("Synthetic data", fontsize=9, y=1.02)
fig.tight_layout()

out = Path(__file__).resolve().parents[2] / "docs/assets/neural/continual_learning.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"figure: {out}")
print(f"  forgetting: {FORGETTING}")
print(f"  MSE: {MSE}")
print(f"  recall: {RECALL}")
