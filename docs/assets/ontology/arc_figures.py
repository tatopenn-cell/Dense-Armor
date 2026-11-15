"""Generate the figures of docs/ontology/grids.md.

Run from the repository root:

    python docs/assets/ontology/arc_figures.py

Every cell of a grid is a coloured square. The ten ARC colours follow
the ARC-AGI-1 palette used in the benchmark's web display.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.core.graph import (  # noqa: E402
    Grid,
    extract_objects,
)

OUT = Path(__file__).resolve().parent

ARC_PALETTE = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
CMAP = ListedColormap(ARC_PALETTE)


def draw(ax, grid, title=None):
    rows = np.array(grid.to_list())
    ax.imshow(rows, cmap=CMAP, vmin=0, vmax=9, interpolation="nearest")
    h, w = grid.shape()
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#333", linewidth=0.5)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#333", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def grids_objects():
    g = Grid.from_list(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 2],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    objs = extract_objects(g)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    draw(ax, g)
    for o in objs:
        if o.color == 0:
            continue
        r0, c0, r1, c1 = o.bbox
        ax.add_patch(
            plt.Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor="#FFDC00",
                linewidth=2,
            )
        )
        ax.text(
            (c0 + c1 - 1) / 2,
            (r0 + r1 - 1) / 2,
            str(o.color),
            color="white",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUT / "arc_grids_objects.png", dpi=140)
    plt.close(fig)


def grids_holes():
    ring = Grid.from_list([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    u = Grid.from_list([[1, 0, 1], [1, 0, 1], [1, 1, 1]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    draw(a, ring, "ring: one hole")
    draw(b, u, "U shape: no hole")
    fig.tight_layout()
    fig.savefig(OUT / "arc_grids_holes.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    grids_objects()
    grids_holes()
    print(f"  wrote {OUT / 'arc_grids_objects.png'}")
    print(f"  wrote {OUT / 'arc_grids_holes.png'}")
