"""The two figures of the page that compares the four drift detectors, drawn with the library itself.

``drift_four_cubes.png``: one 3D surface for each detector with the same axes and the same colours: the median delay
of the first flag after a step of the mean of 1 to 4 standard deviations (the shift), against one parameter of the
detector (h of CUSUM, the threshold of PageHinkley, delta of ADWIN, alpha of KSWIN). A step that is not found in the 300
values after it counts as 300. 5 seeded streams of 300 normal values and then 300 values after the step.

``drift_race.png``: the four detectors with their default settings (CUSUM with a fixed reference) on the same streams:
the delay for each shift as 3D bars, and the flags on 20 stationary streams of 3000 normal values.

Run from the repo root:
    python docs/assets/drift/make_compare_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_detectors_style_figures import BG, CAP, CMAP, CYAN, GRID, HERE, INK, MUTED, ORANGE, PINK, SHIFTS, SEEDS, delay, grid

from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.drift.detector import CUSUMDriftDetector
from dense_armor.core.online.drift.kswin import KSWIN
from dense_armor.core.online.drift.page_hinkley import PageHinkley

VIOLET = "#9a7cff"


def style(ax):
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(colors=MUTED, labelsize=7)


def cubes():
    specs = [
        ("CUSUM", lambda p: CUSUMDriftDetector(reference="fixed", h=p), np.linspace(5.0, 40.0, 8), "h", False),
        ("PageHinkley", lambda p: PageHinkley(delta=0.05, threshold=p), np.linspace(5.0, 50.0, 8), "threshold", False),
        ("ADWIN", lambda p: ADWIN(delta=p, max_window=None), np.logspace(-6, -1, 8), "log10 delta", True),
        ("KSWIN", lambda p: KSWIN(alpha=p, seed=0), np.logspace(-6, -2, 8), "log10 alpha", True),
    ]
    fig = plt.figure(figsize=(14, 10.5), facecolor=BG)
    fig.text(0.04, 0.965, "Four detectors, one step", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.04, 0.935, "delay of the first flag after a step of the daily cost of executing orders (median of 5 streams; not found = 300)", fontsize=11.5, color=CYAN)
    for k, (name, make, params, yl, log) in enumerate(specs):
        Z = grid(make, params)
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        style(ax)
        X, Y = np.meshgrid(SHIFTS, np.log10(params) if log else params)
        ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / CAP, 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
        ax.set_zlim(0, CAP)
        ax.set_title(name, color=INK, fontsize=15, fontweight="bold", y=0.98)
        ax.set_xlabel("shift (sd)", color=CYAN, fontsize=8, labelpad=2)
        ax.set_ylabel(yl, color=CYAN, fontsize=8, labelpad=2)
        ax.set_zlabel("delay", color=CYAN, fontsize=8, labelpad=0)
        print(name, "delay at the shift 1.0 / 2.5 / 4.0 for the first, middle and last parameter:",
              [[float(Z[i, j]) for j in (0, 4, 7)] for i in (0, 4, 7)])
    fig.subplots_adjust(top=0.9, bottom=0.02, left=0.0, right=1.0, hspace=0.02, wspace=-0.05)
    fig.savefig(HERE / "drift_four_cubes.png", dpi=140, facecolor=BG)
    plt.close(fig)


def race():
    makers = [("CUSUM", lambda: CUSUMDriftDetector(reference="fixed"), CYAN), ("PageHinkley", lambda: PageHinkley(), ORANGE),
              ("ADWIN", lambda: ADWIN(), PINK), ("KSWIN", lambda: KSWIN(seed=0), VIOLET)]
    D = np.array([[np.median([delay(mk, sh, s) for s in SEEDS]) for sh in SHIFTS] for _, mk, _ in makers])
    fa = []
    for _, mk, _ in makers:
        n = 0
        for s in range(20):
            det = mk()
            n += sum(det.update(v).drift_detected for v in np.random.default_rng(s).normal(0, 1, 3000))
        fa.append(n)
    for (name, _, _), row, f in zip(makers, D, fa):
        print(name, "delay for shifts", [round(float(s), 1) for s in SHIFTS], "->", [float(v) for v in row], "| flags on 20 stationary streams of 3000:", f)
    fig = plt.figure(figsize=(14, 7.2), facecolor=BG)
    fig.text(0.04, 0.94, "The race", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.04, 0.9, "default settings, same streams: delay after the step (bars) and flags when nothing changes (right)", fontsize=11.5, color=CYAN)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    style(ax)
    for i, (name, _, c) in enumerate(makers):
        ax.bar3d(np.arange(len(SHIFTS)), np.full(len(SHIFTS), i), np.zeros(len(SHIFTS)), 0.7, 0.6, D[i], color=c, alpha=0.9, shade=True)
    ax.set_xticks(np.arange(len(SHIFTS)) + 0.35)
    ax.set_xticklabels([f"{s:.1f}" for s in SHIFTS])
    ax.set_yticks(np.arange(4) + 0.3)
    ax.set_yticklabels([m[0] for m in makers], fontsize=8)
    ax.set_zlim(0, CAP)
    ax.set_xlabel("shift (sd)", color=CYAN, labelpad=6)
    ax.set_zlabel("delay (values)", color=CYAN, labelpad=2)
    ax2 = fig.add_subplot(1, 2, 2, facecolor=BG)
    ax2.bar([m[0] for m in makers], fa, color=[m[2] for m in makers])
    for i, v in enumerate(fa):
        ax2.text(i, v, str(v), ha="center", va="bottom", color=INK, fontsize=12)
    ax2.set_ylabel("flags on 20 streams of 3000 normal values", color=CYAN)
    ax2.tick_params(colors=MUTED)
    for sp in ax2.spines.values():
        sp.set_color(GRID)
    fig.subplots_adjust(top=0.85, bottom=0.08, left=0.0, right=0.96, wspace=0.1)
    fig.savefig(HERE / "drift_race.png", dpi=140, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    cubes()
    race()
