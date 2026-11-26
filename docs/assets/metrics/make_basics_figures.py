"""The two figures of the page "Metric: a score that updates one pair at a time", drawn with the library itself.

basics_3d.png: a daily up or down forecast of a stock index that is right with probability p. For 25 values of p
and 30 simulated streams, the surface is the running ``Accuracy`` after every pair; the front curve is the forecast
"same direction as yesterday" of the first example.
basics_finance.png: one stream of 250 trading days and the running ``Accuracy`` of that forecast.

Run from the repo root:
    python docs/assets/metrics/make_basics_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.metrics import Accuracy

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
N, SEEDS, P_EXAMPLE = 300, 30, 0.55


def running_accuracy(p, seed, n):
    rng = np.random.default_rng(seed)
    right = rng.random(n) < p
    m = Accuracy()
    out = np.zeros(n)
    for i, r in enumerate(right):
        m.update(1, 1 if r else 0)
        out[i] = m.get()
    return out


def running_from_draws(right):
    m = Accuracy()
    out = np.zeros(len(right))
    for i, r in enumerate(right):
        m.update(1, 1 if r else 0)
        out[i] = m.get()
    return out


def figure_3d():
    ps = np.linspace(0.5, 0.9, 25)
    u = np.random.default_rng(11).random(N)
    Z = np.zeros((len(ps), N))
    for k, p in enumerate(ps):
        Z[k] = running_from_draws(u < p)
    t = np.arange(1, N + 1)
    T, P = np.meshgrid(t, ps)
    cmap = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot_surface(T, P, Z, facecolors=cmap((P - 0.5) / 0.4), rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.93)
    y_front = 0.5 - 0.03
    example = running_from_draws(u < P_EXAMPLE)
    ax.plot(t, np.full(N, y_front), example, color=ORANGE, lw=2.8)
    ax.set_ylim(y_front, 0.9)
    ax.set_zlim(0.4, 1.0)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("pairs seen", color=CYAN, labelpad=8)
    ax.set_ylabel("how often the forecast is right", color=CYAN, labelpad=8)
    ax.set_zlabel("running Accuracy", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(N * 0.45, y_front, 0.93, "the forecast of the example: right 55% of the days", color=ORANGE, fontsize=10)
    fig.text(0.05, 0.955, "METRIC: A SCORE THAT UPDATES ONE PAIR AT A TIME", fontsize=21, fontweight="bold", color=INK)
    fig.text(
        0.05, 0.915,
        "stock index, same 300 days: running Accuracy of forecasts right 50% to 90% of the days; orange: the example (55%)",
        fontsize=11.5, color=CYAN,
    )
    fig.savefig(HERE / "basics_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    k90 = int(np.argmin(np.abs(ps - 0.9)))
    k50 = 0
    for label, curve in (("right 55% of the days", example), ("right 90% of the days", Z[k90]), ("right 50% of the days", Z[k50])):
        print(f"Accuracy of the forecast {label}, after 10, 50 and 300 pairs:", [round(float(curve[n - 1]), 3) for n in (10, 50, 300)])
    print("highest and lowest Accuracy of the example between pair 20 and pair 300:",
          round(float(example[19:].max()), 3), round(float(example[19:].min()), 3))


def figure_finance():
    rng = np.random.default_rng(7)
    n = 250
    up = np.zeros(n, dtype=int)
    up[0] = 1
    for i in range(1, n):
        up[i] = up[i - 1] if rng.random() < P_EXAMPLE else 1 - up[i - 1]
    m = Accuracy()
    curve = np.zeros(n - 1)
    for i, (yesterday, today) in enumerate(zip(up[:-1], up[1:])):
        m.update(today, yesterday)
        curve[i] = m.get()
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor("#111a24")
    ax.plot(np.arange(1, n), curve, color=CYAN, lw=2.0, label="running Accuracy of the forecast")
    ax.axhline(0.5, color=ORANGE, ls="--", lw=1.5, label="a coin toss: 0.50")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("trading days seen", color=CYAN)
    ax.set_ylabel("Accuracy", color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title('Stock index: "tomorrow moves like today"', color=INK, loc="left", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#111a24", edgecolor=GRID, labelcolor=INK, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "basics_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("250 trading days, 249 pairs: Accuracy after 10, 50, 100 and 249 pairs:",
          [round(float(curve[k - 1]), 3) for k in (10, 50, 100, 249)], "| n =", m.n)


if __name__ == "__main__":
    figure_3d()
    figure_finance()
