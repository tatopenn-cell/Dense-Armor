"""The figures of the Uncertainty pages, drawn with the library itself.

One stream of daily returns of a stock index: 160 calm days (standard deviation 0.5%) and then 160 stormy days
(1.5%). The forecast of each day is an interval, or a mean and a variance, built from the standard deviation s of
the 20 days before it: the interval is -k s to +k s and the Gaussian forecast has mean 0 and variance (m s)^2.
*_3d: the running metric for 25 values of k (or m) from 0.5 to 3 over 300 days.
*_finance: three values of k (or m) over the first 250 of those days.

Run from the repo root:
    python docs/assets/metrics/make_uncertainty_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.metrics import GaussianNLL, IntervalCoverage, MeanIntervalWidth

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
RNG = np.random.default_rng(41)
R = np.concatenate([RNG.normal(0, 0.005, 160), RNG.normal(0, 0.015, 160)])
S = np.array([R[t - 20 : t].std() for t in range(20, 320)])
Y = R[20:]
KS = np.linspace(0.5, 3.0, 25)


def run(kind, k):
    m = {"cov": IntervalCoverage, "wid": MeanIntervalWidth, "nll": GaussianNLL}[kind]()
    out = np.zeros(len(Y))
    for i, (y, s) in enumerate(zip(Y, S)):
        pred = (-k * s, k * s) if kind != "nll" else (0.0, (k * s) ** 2)
        m.update(float(y), pred)
        out[i] = m.get()
    return out


def surface(kind, name, title, subtitle, xl, zl, zlim):
    Z = np.array([run(kind, k)[19:] for k in KS])
    T, G = np.meshgrid(np.arange(20, 301), KS)
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
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    ax.plot_surface(T, G, Z, facecolors=CMAP(np.clip((Z - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)), rstride=1, cstride=4, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    pick = [int(np.argmin(np.abs(KS - v))) for v in (1.0, 1.645, 3.0)]
    best = int(np.argmin(Z[:, -1])) if kind == "nll" else None
    print(name, "after 300 days, k:", [round(float(KS[i]), 3) for i in pick], "->", [round(float(Z[i, -1]), 4) for i in pick],
          "| best k (nll only):", None if best is None else (round(float(KS[best]), 3), round(float(Z[best, -1]), 4)))


def lines(kind, name, title, ylabel, ks, labels, target=None, loc="center right"):
    n = 250
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("trading days seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s_ in ax.spines.values():
        s_.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    if target is not None:
        ax.axhline(target, color=MUTED, lw=1.0, ls="--")
    out = {}
    for k, col, lab in zip(ks, (CYAN, ORANGE, PINK), labels):
        v = run(kind, k)[:n]
        out[lab] = [round(float(v[i]), 4) for i in (9, 49, 99, n - 1)]
        ax.plot(np.arange(1, n + 1), v, color=col, lw=2.0, label=lab)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(name, "after 10, 50, 100, 250 days:", out)


if __name__ == "__main__":
    surface("cov", "uncertainty_coverage", "COVERAGE RISES WITH THE SIZE OF THE BAND",
            "daily returns of a stock index, same 300 days: running IntervalCoverage for 25 band sizes k", "band size k (times the recent volatility)", "running coverage", (0.0, 1.0))
    surface("wid", "uncertainty_width", "THE WIDTH FOLLOWS THE VOLATILITY",
            "same 300 days: running MeanIntervalWidth for 25 band sizes k", "band size k (times the recent volatility)", "running width", (0.0, 0.18))
    surface("nll", "uncertainty_gaussian_nll", "THE NLL IS LOWEST FOR THE HONEST VARIANCE",
            "same 300 days: running GaussianNLL for 25 scales m of the quoted standard deviation", "scale m of the quoted standard deviation", "running NLL", (-4.0, 0.0))
    lines("cov", "uncertainty_coverage", "Stock index: running coverage of the band of k times the recent volatility", "IntervalCoverage",
          (1.0, 1.645, 3.0), ("k = 1", "k = 1.645", "k = 3"), target=0.9, loc="lower right")
    lines("wid", "uncertainty_width", "Stock index: running width of the same bands", "MeanIntervalWidth",
          (1.0, 1.645, 3.0), ("k = 1", "k = 1.645", "k = 3"), loc="upper left")
    lines("nll", "uncertainty_gaussian_nll", "Stock index: running NLL of a Gaussian forecast with standard deviation m times the recent volatility", "GaussianNLL",
          (0.5, 1.0, 2.0), ("m = 0.5", "m = 1", "m = 2"), loc="upper right")
