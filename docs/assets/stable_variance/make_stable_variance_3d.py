"""Dark 3D figure for the stable variance page, drawn with the library itself.

For a grid of offsets (the mean of the data) and spreads (their standard deviation), 1000 float32 values are
drawn; the variance is computed in float32 by the one-pass textbook formula ``mean(x**2) - mean(x)**2`` and by
``stable_var``, and compared with the float64 variance of the same float32 values. When the exact variance is 0
(the values are identical) any other answer counts as an error of 100 %. The vertical axis is the log10 of the
relative error, between 1e-8 and 1e2.

Run from the repo root:
    python docs/assets/stable_variance/make_stable_variance_3d.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.signals.stats.stable import stable_var

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, PINK = "#3ec6e8", "#ff2d87"
LO, HI = -8.0, 2.0


def error(value, exact):
    if exact == 0.0:
        return 0.0 if value == 0.0 else 1.0
    return abs(value - exact) / exact


def main():
    offsets = np.arange(1.0, 6.51, 0.5)
    spreads = np.arange(-2.0, 1.01, 0.5)
    naive_err = np.zeros((len(spreads), len(offsets)))
    stable_err = np.zeros_like(naive_err)
    for j, lo in enumerate(offsets):
        for i, ls in enumerate(spreads):
            rng = np.random.default_rng(0)
            x = jnp.asarray(rng.normal(10.0**lo, 10.0**ls, 1000).astype(np.float32))
            exact = float(np.var(np.asarray(x, dtype=np.float64)))
            naive_err[i, j] = error(float(jnp.mean(x * x) - jnp.mean(x) ** 2), exact)
            stable_err[i, j] = error(float(stable_var(x)), exact)
    Zn = np.clip(np.log10(np.maximum(naive_err, 1e-30)), LO, HI)
    Zs = np.clip(np.log10(np.maximum(stable_err, 1e-30)), LO, HI)
    OFF, S = np.meshgrid(offsets, spreads)
    cn = LinearSegmentedColormap.from_list("n", ["#6a5acd", PINK])
    cs = LinearSegmentedColormap.from_list("s", ["#1d6f8a", CYAN])
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot_surface(
        OFF,
        S,
        Zn,
        facecolors=cn((Zn - LO) / (HI - LO)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        alpha=0.92,
    )
    ax.plot_surface(
        OFF,
        S,
        Zs,
        facecolors=cs((Zs - LO) / (HI - LO)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        alpha=0.95,
    )
    ax.set_zlim(LO, HI)
    ax.view_init(elev=24, azim=-122)
    ax.set_box_aspect((1.3, 1.1, 0.7), zoom=1.1)
    ax.set_xlabel("offset of the data, log10 of the mean", color=CYAN, labelpad=8)
    ax.set_ylabel("spread of the data, log10 of the std", color=CYAN, labelpad=8)
    ax.set_zlabel("relative error of the variance, log10", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(
        0.05,
        0.955,
        "STABLE VARIANCE: WHERE THE TEXTBOOK FORMULA FAILS",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.05,
        0.915,
        "float32, 1000 values per cell; pink: mean(x**2) - mean(x)**2; blue: stable_var; error 1e2 and 1e-8 are the top and the floor",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "stable_variance_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    wrong = naive_err > 1e-3
    print(
        "cells:",
        naive_err.size,
        "| naive error above 1e-3:",
        int(wrong.sum()),
        "| above 100 %:",
        int((naive_err >= 1.0).sum()),
    )
    print(
        "stable_var largest error over all cells:",
        f"{stable_err.max():.2e}",
        "| naive largest:",
        f"{naive_err.max():.2e}",
    )
    for lo_, ls_ in ((3.0, -1.0), (6.0, 0.0), (6.0, -1.0)):
        j = int(np.argmin(np.abs(offsets - lo_)))
        i = int(np.argmin(np.abs(spreads - ls_)))
        print(
            f"offset 1e{lo_:g}, spread 1e{ls_:g}: naive {naive_err[i, j]:.2e}, stable_var {stable_err[i, j]:.2e}"
        )


if __name__ == "__main__":
    main()
