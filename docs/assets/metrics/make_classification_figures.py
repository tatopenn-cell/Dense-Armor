"""The figures of the Classification pages, drawn with the library itself.

precision_recall_f1: an alarm score for the crash days of a stock index; the surface is the running F1 of the alarm
    for 25 thresholds on the same 400 days; the second figure is one threshold on 250 days.
confusion_matrix: a model that never flags a crash; the two surfaces are the running Accuracy and BalancedAccuracy
    for crash shares from 1% to 50%; the second figure is the confusion matrix of the example alarm on 250 days.
kappa: a forecast that always says "up"; the two surfaces are the running Accuracy and CohenKappa for up shares from
    50% to 95%; the second figure is one stream of 250 days.

Run from the repo root:
    python docs/assets/metrics/make_classification_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.metrics import (
    Accuracy,
    BalancedAccuracy,
    CohenKappa,
    ConfusionMatrix,
    F1,
    Precision,
    Recall,
)

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])


def run(metric_factory, truth, flags):
    m = metric_factory()
    out = np.zeros(len(truth))
    for i, (a, b) in enumerate(zip(truth, flags)):
        m.update(int(a), int(b))
        v = m.get()
        out[i] = np.nan if v is None else v
    return out


def base_3d(title, subtitle, xl, yl, zl):
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel(xl, color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    return fig, ax


def finance_axes(title, xl, yl):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel(xl, color=CYAN)
    ax.set_ylabel(yl, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    return fig, ax


def legend(ax, loc="lower right"):
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)


def precision_recall_f1():
    n_days, thr_list = 400, np.linspace(0.0, 3.0, 25)
    rng = np.random.default_rng(5)
    crash = (rng.random(n_days) < 0.10).astype(int)
    score = 1.5 * crash + rng.normal(0, 1, n_days)
    Z = np.array([run(lambda: F1(positive=1), crash, score > thr) for thr in thr_list])
    t = np.arange(20, n_days + 1)
    Z = Z[:, 19:]
    T, Y = np.meshgrid(t, thr_list)
    fig, ax = base_3d(
        "HOW THE F1 OF AN ALARM SETTLES",
        "crash days of a stock index, same 400 days: running F1 of an alarm for 25 thresholds",
        "days seen", "alarm threshold", "running F1",
    )
    ax.plot_surface(T, Y, Z, facecolors=CMAP(np.clip(Z, 0, 1)), rstride=1, cstride=4, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0.0, 1.0)
    fig.savefig(HERE / "f1_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    best = int(np.argmax(Z[:, -1]))
    print("F1 after 400 days: best threshold", round(float(thr_list[best]), 2), "F1", round(float(Z[best, -1]), 3),
          "| threshold 0.0:", round(float(Z[0, -1]), 3), "| threshold 3.0:", round(float(Z[-1, -1]), 3))

    rng = np.random.default_rng(9)
    n = 250
    crash = (rng.random(n) < 0.10).astype(int)
    flag = (1.5 * crash + rng.normal(0, 1, n)) > 1.0
    p, r, f = (run(lambda: c(positive=1), crash, flag) for c in (Precision, Recall, F1))
    fig, ax = finance_axes("Crash days of a stock index: an alarm at threshold 1.0", "trading days seen", "score")
    ax.plot(np.arange(1, n + 1), p, color=CYAN, lw=1.8, label="Precision")
    ax.plot(np.arange(1, n + 1), r, color=ORANGE, lw=1.8, label="Recall")
    ax.plot(np.arange(1, n + 1), f, color=PINK, lw=2.2, label="F1")
    ax.set_ylim(0, 1.05)
    legend(ax)
    fig.tight_layout()
    fig.savefig(HERE / "f1_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"250 days, {int(crash.sum())} crash days, {int(flag.sum())} alarms: Precision, Recall, F1 after 250 days:",
          [round(float(v[-1]), 3) for v in (p, r, f)])


def confusion_matrix():
    n_days, shares = 300, np.linspace(0.01, 0.5, 25)
    u = np.random.default_rng(3).random(n_days)
    Za = np.array([run(Accuracy, u < s, np.zeros(n_days)) for s in shares])
    Zb = np.array([run(BalancedAccuracy, u < s, np.zeros(n_days)) for s in shares])
    t = np.arange(20, n_days + 1)
    Za, Zb = Za[:, 19:], Zb[:, 19:]
    T, Y = np.meshgrid(t, shares)
    fig, ax = base_3d(
        "ACCURACY LOOKS GOOD WHEN CRASHES ARE RARE",
        "a model that never flags a crash, same 300 days: Accuracy (cyan) and BalancedAccuracy (pink) for 25 crash shares",
        "days seen", "share of crash days", "score",
    )
    ax.plot_surface(T, Y, Za, color=CYAN, rstride=1, cstride=6, linewidth=0, antialiased=False, alpha=0.8)
    ax.plot_surface(T, Y, Zb, color=PINK, rstride=1, cstride=6, linewidth=0, antialiased=False, alpha=0.8)
    ax.set_zlim(0.0, 1.0)
    fig.savefig(HERE / "confusion_matrix_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("never flags: Accuracy, BalancedAccuracy after 300 days for crash share 1%, 10% and 50%:",
          [(round(float(Za[k, -1]), 3), round(float(Zb[k, -1]), 3)) for k in (0, int(np.argmin(np.abs(shares - 0.10))), -1)])

    rng = np.random.default_rng(9)
    n = 250
    crash = (rng.random(n) < 0.10).astype(int)
    flag = ((1.5 * crash + rng.normal(0, 1, n)) > 1.0).astype(int)
    cm = ConfusionMatrix()
    for a, b in zip(crash, flag):
        cm.update(int(a), int(b))
    mat = cm.get()
    M = np.array([[mat[a].get(b, 0.0) for b in (0, 1)] for a in (0, 1)])
    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.imshow(M, cmap=CMAP)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", color=INK, fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1], ["no alarm", "alarm"], color=MUTED)
    ax.set_yticks([0, 1], ["no crash", "crash"], color=MUTED)
    ax.set_xlabel("what the alarm said", color=CYAN)
    ax.set_ylabel("what happened", color=CYAN)
    ax.set_title("Crash days of a stock index: 250 days", color=INK, loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "confusion_matrix_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("confusion matrix of the 250 days [[TN, FP], [FN, TP]]:", M.astype(int).tolist())


def kappa():
    n_days, shares = 300, np.linspace(0.5, 0.95, 25)
    u = np.random.default_rng(4).random(n_days)
    Za = np.array([run(Accuracy, u < s, np.ones(n_days)) for s in shares])
    Zk = np.array([run(CohenKappa, u < s, np.ones(n_days)) for s in shares])
    t = np.arange(20, n_days + 1)
    Za, Zk = Za[:, 19:], Zk[:, 19:]
    T, Y = np.meshgrid(t, shares)
    fig, ax = base_3d(
        "KAPPA STAYS AT ZERO FOR A FORECAST THAT IS A HABIT",
        "a forecast that always says up, same 300 days: Accuracy (cyan) and CohenKappa (pink) for 25 up shares",
        "days seen", "share of up days", "score",
    )
    ax.plot_surface(T, Y, Za, color=CYAN, rstride=1, cstride=6, linewidth=0, antialiased=False, alpha=0.8)
    ax.plot_surface(T, Y, Zk, color=PINK, rstride=1, cstride=6, linewidth=0, antialiased=False, alpha=0.8)
    ax.set_zlim(-0.2, 1.0)
    fig.savefig(HERE / "kappa_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    k80 = int(np.argmin(np.abs(shares - 0.8)))
    print("always up: Accuracy, CohenKappa after 300 days for up share 50%, 80% and 95%:",
          [(round(float(Za[k, -1]), 3), round(float(Zk[k, -1]), 3)) for k in (0, k80, -1)])

    rng = np.random.default_rng(12)
    n = 250
    up = (rng.random(n) < 0.8).astype(int)
    acc, kap = run(Accuracy, up, np.ones(n)), run(CohenKappa, up, np.ones(n))
    fig, ax = finance_axes('Stock index: a forecast that always says "up"', "trading days seen", "score")
    ax.plot(np.arange(1, n + 1), acc, color=CYAN, lw=1.8, label="Accuracy")
    ax.plot(np.arange(1, n + 1), kap, color=PINK, lw=2.2, label="CohenKappa")
    ax.set_ylim(-0.2, 1.05)
    legend(ax, "center right")
    fig.tight_layout()
    fig.savefig(HERE / "kappa_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"250 days, {int(up.sum())} up days: Accuracy and CohenKappa after 250 days:", round(float(acc[-1]), 3), round(float(kap[-1]), 3))


if __name__ == "__main__":
    precision_recall_f1()
    confusion_matrix()
    kappa()
