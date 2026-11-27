"""The 3D figures of the Drift detectors pages, drawn with the library itself.

A desk watches the daily cost of executing orders: 300 normal days, then a step of the mean of `shift` standard
deviations. For each detector the surface is the median delay, in values after the step, of the first flag, over 5
seeded streams, for 8 values of the shift and 8 values of one parameter of the detector. A step that is not found in
the 300 values after it counts as 300.

Run from the repo root:
    python docs/assets/drift/make_detectors_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.drift.kswin import KSWIN
from dense_armor.core.online.drift.page_hinkley import PageHinkley

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
SHIFTS = np.linspace(1.0, 4.0, 8)
SEEDS = range(5)
CAP = 300


def delay(make, shift, seed):
    r = np.random.default_rng(seed)
    y = np.concatenate([r.normal(0, 1, 300), r.normal(shift, 1, CAP)])
    det = make()
    for i, v in enumerate(y):
        if det.update(v).drift_detected and i >= 300:
            return i - 300
    return CAP


def grid(make_for, params):
    return np.array([[np.median([delay(lambda: make_for(p), sh, s) for s in SEEDS]) for sh in SHIFTS] for p in params])


def surface(Z, params, name, title, subtitle, yl, log=False):
    X, Y = np.meshgrid(SHIFTS, np.log10(params) if log else params)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("shift of the mean (standard deviations)", color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel("delay (values after the step)", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / CAP, 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, CAP)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    sub = "median delay of the first flag on the daily cost of executing orders, 5 seeded streams; a step not found counts as 300"
    th = np.linspace(5.0, 50.0, 8)
    surface(grid(lambda p: PageHinkley(delta=0.05, threshold=p), th), th, "drift_page_hinkley",
            "PageHinkley: delay against the threshold", sub, "threshold")
    de = np.logspace(-6, -1, 8)
    surface(grid(lambda p: ADWIN(delta=p, max_window=None), de), de, "drift_adwin",
            "ADWIN: delay against delta", sub, "log10 of delta", log=True)
    al = np.logspace(-6, -2, 8)
    surface(grid(lambda p: KSWIN(alpha=p, seed=0), al), al, "drift_kswin",
            "KSWIN: delay against alpha", sub, "log10 of alpha", log=True)
    print("done")


if __name__ == "__main__":
    main()
