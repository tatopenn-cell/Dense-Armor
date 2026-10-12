import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import (
    SGD, AdaGrad, Adam, FTRLProximal, OnlineNewtonStep,
)

out = Path(__file__).parent
rng = np.random.default_rng(0)

n = 400
X = rng.normal(size=(n, 2))
y = np.empty(n)
for i in range(n):
    w = np.array([1.0, -2.0]) if i < n // 2 else np.array([-1.0, 3.0])
    y[i] = float(w @ X[i]) + 0.05 * rng.normal()

opts = [
    (SGD(lr=0.05), "SGD(lr=0.05)", "tab:blue"),
    (AdaGrad(lr=0.5), "AdaGrad(lr=0.5)", "tab:orange"),
    (Adam(lr=0.05), "Adam(lr=0.05)", "tab:green"),
    (FTRLProximal(l2=0.1), "FTRLProximal(l2=0.1)", "tab:red"),
    (OnlineNewtonStep(), "OnlineNewtonStep", "tab:purple"),
]

fig, ax = plt.subplots(figsize=(8, 3.4))
win = 20
for opt, name, color in opts:
    est = LinearRegression(optimizer=opt, intercept=True,
                           feature_keys=["a", "b"])
    errs = np.empty(n)
    for i in range(n):
        x = {"a": float(X[i, 0]), "b": float(X[i, 1])}
        errs[i] = abs(est.predict_one(x) - y[i])
        est.learn_one(x, float(y[i]))
    smooth = np.convolve(errs, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(len(smooth)), smooth, color=color, lw=1.2, label=name)

ax.axvline(n // 2, color="0.3", lw=1.0, ls="--")
ax.text(n // 2 + 6, ax.get_ylim()[1] * 0.9,
        "true weights change", fontsize=8, color="0.3")
ax.set_xlabel("sample")
ax.set_ylabel("|error| (20-sample moving average)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "convergence.png", dpi=130)
print("scritto:", out / "convergence.png")
