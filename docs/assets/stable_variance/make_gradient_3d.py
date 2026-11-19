"""Dark 3D figure of the stable variance page: how the gradient degrades with the depth, from the GPU run.

For a stack of normalised layers in float32 on a GPU (benchmarks/stable_variance_depth.py --grid), the relative
error of the gradient against the number of layers and against the offset over the spread of the activations,
for the usual layer normalisation (pink) and for the one with stable_var (blue).

Run from the repo root:
    python docs/assets/stable_variance/make_gradient_3d.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "benchmarks" / "stable_variance_depth_grid_gpu.json"
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, PINK = "#3ec6e8", "#ff2d87"


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    rows = d["rows"]
    depths = sorted({r["depth"] for r in rows})
    spreads = sorted({r["spread"] for r in rows})
    Zu = np.zeros((len(spreads), len(depths)))
    Zs = np.zeros_like(Zu)
    for r in rows:
        i, j = spreads.index(r["spread"]), depths.index(r["depth"])
        Zu[i, j] = np.log10(r["usual"]["grad_rel_err"])
        Zs[i, j] = np.log10(r["stable"]["grad_rel_err"])
    X, Y = np.meshgrid(np.log2(depths), np.log10([d["offset"] / s for s in spreads]))
    lo, hi = -7.0, 0.0
    cn = LinearSegmentedColormap.from_list("n", ["#9d6bff", "#ff5fa8", "#ff2d87"])
    cs = LinearSegmentedColormap.from_list("s", ["#2a8fb0", "#3ec6e8", "#9ff0ff"])
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot_surface(
        X,
        Y,
        Zu,
        facecolors=cn((Zu - Zu.min()) / (Zu.max() - Zu.min())),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        alpha=0.9,
    )
    ax.plot_surface(
        X,
        Y,
        Zs,
        facecolors=cs((Zs - Zs.min()) / (Zs.max() - Zs.min())),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        alpha=0.97,
    )
    ax.set_zlim(lo, hi)
    ax.view_init(elev=22, azim=-128)
    ax.set_box_aspect((1.3, 1.1, 0.75), zoom=1.1)
    ax.set_xlabel("layers, log2", color=CYAN, labelpad=8)
    ax.set_ylabel("offset over spread, log10", color=CYAN, labelpad=8)
    ax.set_zlabel("gradient error, log10", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(
        0.05,
        0.955,
        "THE GRADIENT DEGRADES WITH THE DEPTH",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.05,
        0.915,
        f"stacked normalised layers on a {d['backend'].upper()}, float32; pink: the usual layer; blue: the layer with stable_var",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "gradient_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("written gradient_3d.png")


if __name__ == "__main__":
    main()
