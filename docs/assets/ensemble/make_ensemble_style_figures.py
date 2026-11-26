"""The figures of the Ensemble pages (3D at the start and finance figure), drawn with the library itself.

One stream of a stock index: the daily return r is N(0, 1) and the desk sees a signal s = r + sigma * noise; the label is
1 when r is positive (the index goes up). The classifier ensembles read the signal and learn the label, one day at a time,
scored before learning (progressive_val_score); the surface is the running accuracy for 25 sigmas over 300 days. The
regressor (weighted average) predicts r from s with three linear models of different learning rate; the surface is the
running mean absolute error for 25 values of eta.

Run from the repo root:
    python docs/assets/ensemble/make_ensemble_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.evaluate import progressive_val_score
from dense_armor.core.evaluation.metrics import Accuracy, MeanAbsoluteError
from dense_armor.core.online.ensemble import (
    ExponentiallyWeightedAverageRegressor,
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.core.online.linear.glm import LinearRegression
from dense_armor.core.online.optim import SGD
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
N = 300
SIGMAS = np.linspace(0.05, 1.5, 25)
ETAS = np.linspace(0.1, 5.0, 25)


def ht(g=20):
    return HoeffdingTreeClassifier(grace_period=g)


MODELS = {
    "voting": lambda: VotingClassifier([ht(20), ht(50), ht(100)], mode="hard"),
    "bagging": lambda: OnlineBaggingClassifier(lambda: ht(20), n_models=5, seed=1),
    "boosting": lambda: OnlineBoostingClassifier(lambda: ht(20), n_models=5, seed=1),
    "stacking": lambda: StackingClassifier([ht(20), ht(100)], ht(20)),
}
TITLES = {
    "voting": ("A VOTE HELPS WHEN THE SIGNAL IS NOISY", "a stock index, same 300 days: running accuracy of a vote of three trees for 25 noise levels"),
    "bagging": ("A BAG OF TREES ON A NOISY SIGNAL", "a stock index, same 300 days: running accuracy of a bag of five trees for 25 noise levels"),
    "boosting": ("BOOSTING ON A NOISY SIGNAL", "a stock index, same 300 days: running accuracy of five boosted trees for 25 noise levels"),
    "stacking": ("A SECOND MODEL ON TOP OF TWO TREES", "a stock index, same 300 days: running accuracy of a stack of two trees and a meta-tree for 25 noise levels"),
    "ewa": ("THE WEIGHTED AVERAGE AGAINST THE LEARNING RATE", "a stock index, same 300 days: running mean absolute error of the weighted average of three models for 25 values of eta"),
}


def stream(sigma, seed=71):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, N)
    s = r + sigma * rng.normal(0, 1, N)
    return [({"signal": float(a)}, int(b > 0)) for a, b in zip(s, r)]


def reg_stream(seed=72):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, N)
    y = 0.5 * s + 0.05 * rng.normal(0, 1, N)
    return [({"s": float(a)}, float(b)) for a, b in zip(s, y)]


def trace_cls(make, sigma):
    return np.array(progressive_val_score(stream(sigma), make(), Accuracy(), every=1)[1])


def reg_members():
    return [LinearRegression(optimizer=SGD(lr=lr), feature_keys=["s"]) for lr in (0.002, 0.02, 0.2)]


def trace_reg(model):
    return np.array(progressive_val_score(reg_stream(), model, MeanAbsoluteError(), every=1)[1])


def surface(Z, xs, name, xl, zl, zlim, key):
    X, T = np.meshgrid(np.arange(20, N + 1), xs)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("days seen", color=CYAN, labelpad=8)
    ax.set_ylabel(xl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, TITLES[key][0], fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, TITLES[key][1], fontsize=11.5, color=CYAN)
    ax.plot_surface(X, T, Z, facecolors=CMAP(np.clip((Z - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)), rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title, ylabel, ylim, loc="lower right"):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("days seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, MUTED)):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, days):
    return [round(float(v[d - 1]), 3) for d in days]


if __name__ == "__main__":
    for key, make in MODELS.items():
        full = np.array([trace_cls(make, s) for s in SIGMAS])
        surface(full[:, 19:], SIGMAS, f"ensemble_{key}", "noise of the signal (sigma)", "running accuracy", (0.4, 1.0), key)
        pick = [int(np.argmin(np.abs(SIGMAS - v))) for v in (0.05, 0.8, 1.5)]
        print(key, "accuracy at day 300 for sigma", [round(float(SIGMAS[i]), 2) for i in pick], "->", [round(float(full[i, -1]), 3) for i in pick])
        single = trace_cls(lambda: ht(20), 0.8)
        ens = trace_cls(make, 0.8)
        print(" sigma 0.8, day 50/150/300: single", at(single, (50, 150, 300)), "| ensemble", at(ens, (50, 150, 300)))
        lines([("single tree", single), (key, ens)], f"ensemble_{key}", f"A stock index, signal with noise 0.8: one tree and the {key} of trees", "Accuracy", (0.4, 1.0))
    full = np.array([trace_reg(ExponentiallyWeightedAverageRegressor(reg_members(), eta=e)) for e in ETAS])
    surface(full[:, 19:], ETAS, "ensemble_ewa", "learning rate eta", "running mean absolute error", (0.0, float(np.ceil(full.max() * 10) / 10)), "ewa")
    pick = [int(np.argmin(np.abs(ETAS - v))) for v in (0.1, 1.0, 5.0)]
    print("ewa MAE at day 300 for eta", [round(float(ETAS[i]), 2) for i in pick], "->", [round(float(full[i, -1]), 3) for i in pick])
    ms = reg_members()
    members = [trace_reg(m) for m in ms]
    ew = trace_reg(ExponentiallyWeightedAverageRegressor(reg_members(), eta=1.0))
    print(" MAE day 50/150/300: members", [at(m, (50, 150, 300)) for m in members], "| ewa eta 1", at(ew, (50, 150, 300)))
    lines([("learning rate 0.002", members[0]), ("learning rate 0.02", members[1]), ("learning rate 0.2", members[2]), ("weighted average, eta 1", ew)], "ensemble_ewa",
          "A stock index: the three models and their weighted average", "MeanAbsoluteError", (0.0, float(np.ceil(max(m[19:].max() for m in members + [ew]) * 10) / 10)), "upper right")
