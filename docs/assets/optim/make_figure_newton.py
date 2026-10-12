import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import OnlineNewtonStep

out = Path(__file__).parent

curves = [
    (0.1, "tab:blue",   "gamma = 0.1  (small step)"),
    (0.5, "tab:green",  "gamma = 0.5"),
    (1.0, "tab:orange", "gamma = 1.0  (default)"),
    (5.0, "tab:red",    "gamma = 5.0  (large)"),
]

fig, ax = plt.subplots(figsize=(8, 3.2))
for gamma, color, name in curves:
    reg = LinearRegression(optimizer=OnlineNewtonStep(gamma=gamma, eps=1.0),
                           intercept=False, feature_keys=["x"])
    w_hist = [0.0]
    for _ in range(60):
        reg.learn_one({"x": 1.0}, 2.0)
        w_hist.append(float(reg._w[0]))
    ax.plot(w_hist, color=color, lw=1.2, label=name)

ax.axhline(2.0, color="0.5", lw=0.8, ls="--")
ax.text(35, 2.15, "target weight w* = 2", fontsize=8, color="0.4")
ax.set_xlabel("sample")
ax.set_ylabel("weight w")
ax.set_ylim(-1, 5)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "newton_gamma_choice.png", dpi=130)
print("scritto:", out / "newton_gamma_choice.png")
