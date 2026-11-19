"""Dark 3D figure for the Moments page, drawn with the library itself.

A card terminal sees payments with an average amount of 12 euros; at payment 300 the average amount becomes 35.
Each curve is the mean of 40 simulated streams. The surface is the mean that ``EWStats`` reports after every payment,
for 40 values of its forgetting factor alpha; the front curve is the mean of ``RunningMoments``, which never
forgets, and the dashed back curve is the true average of the day.

Run from the repo root:
    python docs/assets/moments/make_moments_3d.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.signals.stats.moments import EWStats, RunningMoments

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
N, CUT = 600, 300


SEEDS = 40


def stream(seed):
    rng = np.random.default_rng(seed)
    mean = np.where(np.arange(N) < CUT, 12.0, 35.0)
    return rng.gamma(4.0, mean / 4.0), mean


def main():
    alphas = np.geomspace(0.005, 0.15, 36)
    Z = np.zeros((len(alphas), N))
    run = np.zeros(N)
    for seed in range(SEEDS):
        x, true = stream(seed)
        for k, a in enumerate(alphas):
            e = EWStats(alpha=float(a))
            for i, v in enumerate(x):
                e.learn_one({"x": float(v)})
                Z[k, i] += e.mean / SEEDS
        rm = RunningMoments()
        for i, v in enumerate(x):
            rm.learn_one({"x": float(v)})
            run[i] += rm.mean / SEEDS
    t = np.arange(1, N + 1)
    T, A = np.meshgrid(t, np.log10(alphas))
    cmap = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    norm = (Z - 8.0) / (40.0 - 8.0)
    ax.plot_surface(
        T,
        A,
        Z,
        facecolors=cmap(np.clip(norm, 0, 1)),
        rstride=1,
        cstride=8,
        linewidth=0,
        antialiased=False,
        alpha=0.93,
    )
    y_front = np.log10(alphas[0]) - 0.45
    y_back = np.log10(alphas[-1]) + 0.25
    ax.plot(t, np.full(N, y_front), run, color=ORANGE, lw=2.8)
    ax.plot(t, np.full(N, y_back), true, color=INK, lw=1.8, ls="--")
    ax.set_ylim(y_front, y_back)
    ax.set_zlim(5.0, 45.0)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("payments seen", color=CYAN, labelpad=8)
    ax.set_ylabel("forgetting factor alpha (log10)", color=CYAN, labelpad=8)
    ax.set_zlabel("average payment, euros", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(
        N * 0.62,
        y_front - 0.05,
        run[-1] + 6,
        "RunningMoments: never forgets",
        color=ORANGE,
        fontsize=10,
    )
    ax.text(N * 0.1, y_back, 40.0, "true average of the day", color=INK, fontsize=10)
    fig.text(
        0.05,
        0.955,
        "MOMENTS: HOW FAST THE AVERAGE FOLLOWS THE PAYMENTS",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.05,
        0.915,
        "average payment 12 to 35 euros at payment 300; surface: EWStats for each alpha; front curve: RunningMoments (mean of 40 streams)",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "moments_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    i350, i600 = 349, N - 1
    print(
        "RunningMoments mean at payment 350 and 600:",
        round(run[i350], 1),
        round(run[i600], 1),
    )
    for a in (0.005, 0.05, 0.15):
        k = int(np.argmin(np.abs(alphas - a)))
        print(
            f"EWStats alpha {alphas[k]:.3f} mean at payment 350 and 600:",
            round(Z[k, i350], 1),
            round(Z[k, i600], 1),
        )


if __name__ == "__main__":
    main()
