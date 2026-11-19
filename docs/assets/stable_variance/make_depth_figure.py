"""Dark figure for the depth test of the stable variance page, from benchmarks/stable_variance_depth_gpu.json.

Left: the relative error of the gradient against the number of layers, for the usual layer normalisation and for
the one with stable_var. Right: the time of one forward and backward step.

Run from the repo root:
    python docs/assets/stable_variance/make_depth_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "benchmarks" / "stable_variance_depth_gpu.json"
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, PINK = "#3ec6e8", "#ff2d87"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    rows = d["rows"]
    L = [r["depth"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.06, right=0.985, wspace=0.18)
    fig.text(
        0.04,
        0.945,
        "HOW MANY LAYERS BEFORE THE GRADIENT IS WRONG",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.885,
        f"stacked normalised layers on a {d['backend'].upper()}, float32, {d['batch']} rows of {d['width']} values around {d['offset']:g} with a spread of {d['spread']:g}",
        fontsize=11.5,
        color=CYAN,
    )
    a = axes[0]
    a.plot(
        L,
        [r["usual"]["grad_rel_err"] for r in rows],
        "o-",
        color=PINK,
        lw=2.6,
        label="usual layer",
    )
    a.plot(
        L,
        [r["stable"]["grad_rel_err"] for r in rows],
        "o-",
        color=CYAN,
        lw=2.6,
        label="with stable_var",
    )
    a.axhline(1e-2, color=MUTED, lw=1.0, ls="--")
    a.text(1.1, 1.25e-2, "1 %", color=MUTED, fontsize=9)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("layers")
    a.set_ylabel("relative error of the gradient")
    a.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a.grid(True)
    b = axes[1]
    b.plot(
        L,
        [r["usual"]["ms"] for r in rows],
        "o-",
        color=PINK,
        lw=2.6,
        label="usual layer",
    )
    b.plot(
        L,
        [r["stable"]["ms"] for r in rows],
        "o-",
        color=CYAN,
        lw=2.6,
        label="with stable_var",
    )
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xlabel("layers")
    b.set_ylabel("time of one forward and backward step, ms")
    b.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    b.grid(True)
    fig.savefig(HERE / "depth_gradient.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("written depth_gradient.png")


if __name__ == "__main__":
    main()
