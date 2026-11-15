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
    build_grid_graph,
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


def grids_graph():
    import networkx as nx

    g = Grid.from_list(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 2],
        ]
    )
    t = build_grid_graph(g)
    G = t.to_networkx()

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    draw(a, g, "3 by 3 grid")

    pos = {}
    for n in G.nodes():
        if n.startswith("cell:"):
            _, rc = n.split(":")
            r, c = rc.split(",")
            pos[n] = (int(c), -int(r))
    for n in G.nodes():
        if not n.startswith("obj:"):
            continue
        cells = G.nodes[n]["state"]["cells"]
        rs = [rr for rr, _ in cells]
        cs = [cc for _, cc in cells]
        pos[n] = (sum(cs) / len(cs) + 0.6, -sum(rs) / len(rs) + 0.6)

    cell_nodes = [n for n in G.nodes() if n.startswith("cell:")]
    obj_nodes = [n for n in G.nodes() if n.startswith("obj:")]
    belongs = [
        (u, v) for u, v in G.edges() if u.startswith("cell:") and v.startswith("obj:")
    ]
    adj = [
        (u, v) for u, v in G.edges() if u.startswith("obj:") and v.startswith("obj:")
    ]

    nx.draw_networkx_edges(G, pos, edgelist=belongs, ax=b, edge_color="#888", width=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=adj, ax=b, edge_color="#d9822b", width=2.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cell_nodes,
        ax=b,
        node_size=180,
        node_color="#7FDBFF",
        edgecolors="#333",
        linewidths=0.6,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=obj_nodes,
        ax=b,
        node_size=520,
        node_color="#FFDC00",
        edgecolors="#333",
        linewidths=1.0,
    )
    labels = {n: n.split(":")[0] + ":" + n.split(":")[1] for n in obj_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=b, font_size=8)
    b.set_title("typed graph", fontsize=10)
    b.axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "arc_grids_graph.png", dpi=140)
    plt.close(fig)


def transformations():
    g = Grid.from_list(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    from dense_armor.core.graph import hmirror, vmirror

    s1 = hmirror(g)
    s2 = vmirror(s1)
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.2))
    draw(axs[0], g, "input")
    draw(axs[1], s1, "after hmirror")
    draw(axs[2], s2, "after hmirror, then vmirror")
    fig.tight_layout()
    fig.savefig(OUT / "arc_grids_transformations.png", dpi=140)
    plt.close(fig)


def program_search():
    from dense_armor.core.graph import Grid, Program, rot90

    g_in = Grid.from_list([[1, 2, 3], [4, 5, 6]])
    g_out = rot90(g_in)
    p = Program([rot90])

    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.2))
    draw(axs[0], g_in, "input")
    draw(axs[1], g_out, "expected output")
    draw(axs[2], p.apply(g_in), "Program([rot90]).apply(input)")
    fig.tight_layout()
    fig.savefig(OUT / "arc_grids_program_search.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    grids_objects()
    grids_holes()
    grids_graph()
    transformations()
    program_search()
    print(f"  wrote {OUT / 'arc_grids_objects.png'}")
    print(f"  wrote {OUT / 'arc_grids_holes.png'}")
