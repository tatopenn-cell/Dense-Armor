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
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.interpolate import RectBivariateSpline

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
        Zu[i, j] = 100.0 * r["usual"]["grad_rel_err"]
        Zs[i, j] = 100.0 * r["stable"]["grad_rel_err"]
    xs = np.log2(depths)
    ys = np.log10([d["offset"] / s for s in spreads])
    order = np.argsort(ys)
    ys, Zu, Zs = ys[order], Zu[order], Zs[order]
    fx = np.linspace(xs.min(), xs.max(), 90)
    fy = np.linspace(ys.min(), ys.max(), 60)
    X, Y = np.meshgrid(fx, fy)

    def smooth(Z):
        return RectBivariateSpline(ys, xs, Z, kx=3, ky=3)(fy, fx)

    SU, SS = smooth(Zu), smooth(Zs)
    lo, hi = 0.0, float(Zu.max()) * 1.05
    cn = LinearSegmentedColormap.from_list("n", ["#7b2ff7", "#f72585", "#ffb703"])
    cs = LinearSegmentedColormap.from_list("s", ["#35d0ff", "#35d0ff"])
    ls = LightSource(azdeg=300, altdeg=40)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.05, 0.08, 0.11, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    for S, cm in ((SS, cs), (SU, cn)):
        shade = ls.shade(
            S, cmap=cm, vert_exag=0.4, blend_mode="soft", vmin=S.min(), vmax=S.max()
        )
        if S is SU:
            shade[..., 3] = np.clip(S / (0.35 * S.max()), 0.0, 1.0) ** 0.7
        ax.plot_surface(
            X,
            Y,
            S,
            facecolors=shade,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.plot_wireframe(
            X, Y, S, rstride=6, cstride=6, color="white", linewidth=0.25, alpha=0.35
        )
    ax.contourf(X, Y, SU, zdir="z", offset=lo, levels=16, cmap=cn, alpha=0.35)
    ax.set_zlim(lo, hi)
    ax.view_init(elev=24, azim=-128)
    ax.set_box_aspect((1.35, 1.1, 0.9), zoom=1.1)
    ax.set_xlabel("layers, log2", color=CYAN, labelpad=8)
    ax.set_ylabel("offset over spread, log10", color=CYAN, labelpad=8)
    ax.set_zlabel("gradient wrong by, %", color=CYAN, labelpad=6)
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
        f"stacked normalised layers on a {d['backend'].upper()}, float32; pink to gold: the usual layer; blue: the layer with stable_var",
        fontsize=11.5,
        color=CYAN,
    )
    worst = max(rows, key=lambda r: r["usual"]["grad_rel_err"])
    top = 100 * worst["usual"]["grad_rel_err"]
    low = 100 * max(r["stable"]["grad_rel_err"] for r in rows)
    nl = chr(10)
    fig.text(
        0.06,
        0.78,
        f"usual layer: the gradient is wrong by{nl}{top:.0f} % at {worst['depth']} layers",
        fontsize=14,
        fontweight="bold",
        color=PINK,
        linespacing=1.4,
    )
    fig.text(
        0.06,
        0.68,
        f"layer with stable_var: wrong by less than{nl}{low:.3f} % everywhere (the flat blue floor)",
        fontsize=14,
        fontweight="bold",
        color=CYAN,
        linespacing=1.4,
    )
    fig.savefig(HERE / "gradient_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("written gradient_3d.png")


if __name__ == "__main__":
    main()
