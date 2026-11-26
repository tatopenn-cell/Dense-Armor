"""The figures of the Probabilities pages, drawn with the library itself.

log_loss_3d, brier_score_3d: 300 bonds, each defaults with probability 5% (one stream of random draws for all);
    the surfaces are the running LogLoss and BrierScore of a constant quoted default probability, for 25 quotes
    from 1% to 30%. The second figures follow 250 bonds for three quotes.

Run from the repo root:
    python docs/assets/metrics/make_probabilities_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.metrics import BrierScore, LogLoss

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])


def run(factory, truth, q):
    m = factory()
    out = np.zeros(len(truth))
    for i, y in enumerate(truth):
        m.update(int(y), float(q))
        out[i] = m.get()
    return out


def surface(factory, name, title, subtitle, zl, zmax):
    n, quotes = 300, np.linspace(0.01, 0.30, 25)
    default = np.random.default_rng(21).random(n) < 0.05
    Z = np.array([run(factory, default, q)[19:] for q in quotes])
    T, Y = np.meshgrid(np.arange(20, n + 1), quotes)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("bonds seen", color=CYAN, labelpad=8)
    ax.set_ylabel("quoted default probability", color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    ax.plot_surface(T, Y, Z, facecolors=CMAP(np.clip(Z / zmax, 0, 1)), rstride=1, cstride=4, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, zmax)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    best = int(np.argmin(Z[:, -1]))
    print(name, "after 300 bonds: best quote", round(float(quotes[best]), 3), round(float(Z[best, -1]), 4),
          "| quote 0.01:", round(float(Z[0, -1]), 4), "| quote 0.30:", round(float(Z[-1, -1]), 4),
          "| defaults", int(default.sum()))


def lines(factory, name, title, ylabel):
    n = 250
    default = np.random.default_rng(8).random(n) < 0.05
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("bonds seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    out = {}
    for q, col, lab in ((0.05, CYAN, "quote 5% (honest)"), (0.005, ORANGE, "quote 0.5% (too sure)"), (0.5, PINK, "quote 50% (knows nothing)")):
        v = run(factory, default, q)
        out[q] = round(float(v[-1]), 4)
        ax.plot(np.arange(1, n + 1), v, color=col, lw=2.0, label=lab)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(name, "250 bonds,", int(default.sum()), "defaults, final scores by quote:", out)


if __name__ == "__main__":
    surface(LogLoss, "probabilities_log_loss", "LOG LOSS IS LOWEST FOR THE HONEST QUOTE",
            "bonds that default 5% of the time, same 300 bonds: running LogLoss of 25 constant quotes", "running LogLoss", 1.5)
    surface(BrierScore, "probabilities_brier_score", "BRIER SCORE IS LOWEST FOR THE HONEST QUOTE",
            "bonds that default 5% of the time, same 300 bonds: running BrierScore of 25 constant quotes", "running BrierScore", 0.1)
    lines(LogLoss, "probabilities_log_loss", "Bonds that default 5% of the time: running LogLoss", "LogLoss")
    lines(BrierScore, "probabilities_brier_score", "Bonds that default 5% of the time: running BrierScore", "BrierScore")
