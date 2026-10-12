import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import AdaGrad

out = Path(__file__).parent

curves = [
    (0.01,  "tab:blue",   "lr = 0.01   (too small)"),
    (0.1,   "tab:green",  "lr = 0.1    (slow)"),
    (1.0,   "tab:orange", "lr = 1.0    (converges)"),
    (10.0,  "tab:red",    "lr = 10.0   (overshoots, converges)"),
]

fig, ax = plt.subplots(figsize=(8, 3.2))
for lr, color, name in curves:
    reg = LinearRegression(optimizer=AdaGrad(lr=lr), intercept=False,
                           feature_keys=["x"])
    w_hist = [0.0]
    for _ in range(60):
        reg.learn_one({"x": 1.0}, 2.0)
        w_hist.append(float(reg._w[0]))
    ax.plot(w_hist, color=color, lw=1.2, label=name)

ax.axhline(2.0, color="0.5", lw=0.8, ls="--")
ax.text(35, 2.15, "target weight w* = 2", fontsize=8, color="0.4")
ax.set_xlabel("sample")
ax.set_ylabel("weight w")
ax.set_ylim(-1, 11)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "adagrad_lr_choice.png", dpi=130)
print("scritto:", out / "adagrad_lr_choice.png")
