import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.forest import OnlineExtraTreesRegressor
from dense_armor.utility.rules import AMRulesRegressor
from dense_armor.utility.tree import MondrianForestRegressor

out = Path(__file__).parent
N, D = 2000, 4

LEARNERS = {
    "OXT":              (lambda s: OnlineExtraTreesRegressor(seed=s),             "tab:blue"),
    "MondrianForest":   (lambda s: MondrianForestRegressor(n_trees=10, seed=s),   "tab:green"),
    "AMRules":          (lambda s: AMRulesRegressor(seed=s),                      "tab:orange"),
}

curves = {k: [] for k in LEARNERS}
for seed in range(5):
    X, y = DriftStream(n_samples=N, n_features=D, n_drifts=1,
                       kind="sudden", seed=seed).data()
    names = [f"x{i}" for i in range(D)]
    for name, (make, _) in LEARNERS.items():
        est = make(seed)
        err = np.empty(N)
        for i, (xi, yi) in enumerate(zip(X, y)):
            d = dict(zip(names, map(float, xi)))
            err[i] = abs(float(est.predict_one(d)) - float(yi))
            est.learn_one(d, float(yi))
        curves[name].append(err)

win = 50
smooth_curves = {}
for name, errs in curves.items():
    smooth = np.median(
        [np.convolve(e, np.ones(win) / win, mode="valid") for e in errs],
        axis=0,
    )
    smooth_curves[name] = smooth

fig, ax = plt.subplots(figsize=(9, 3.6))

for name, (_, color) in LEARNERS.items():
    s = smooth_curves[name]
    x = np.arange(len(s)) + win // 2
    ax.plot(x, s, color=color, lw=1.6, label=name)

# zona intorno al cambio
ax.axvspan(N // 2, N // 2 + 200, color="0.9", zorder=0)
ax.axvline(N // 2, color="0.4", lw=1.0, ls="--", zorder=1)
ax.text(N // 2 + 8, ax.get_ylim()[1] * 0.96, "rule changes",
        fontsize=8, color="0.35", va="top")

# etichette degli assi corte
ax.set_xlabel("sample")
ax.set_ylabel("median absolute error")

ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="upper right", fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig(out / "oxt_recovery.png", dpi=130)
print("scritto:", out / "oxt_recovery.png")
