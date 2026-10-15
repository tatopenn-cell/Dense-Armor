"""Generate the figures of docs/multioutput/.

Run from the repository root:

    python docs/assets/multioutput/make_figure.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent


def fig_chain_structure():
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    ax.axis("off")

    x_x = 0.10
    y_ys = [0.80, 0.50, 0.20]
    y0, y1, y2 = y_ys
    tgt_x = 0.50

    ax.add_patch(plt.Circle((x_x, 0.5), 0.05, color="#2c6fa6", zorder=3))
    ax.text(x_x, 0.5, "x", color="white", ha="center", va="center",
            fontsize=12, weight="bold", zorder=4)

    for y, lbl in zip(y_ys, ["y0", "y1", "y2"]):
        ax.add_patch(plt.Circle((tgt_x, y), 0.05, color="#2c6fa6", zorder=3))
        ax.text(tgt_x, y, lbl, color="white", ha="center", va="center",
                fontsize=12, weight="bold", zorder=4)

    for y in y_ys:
        ax.annotate("", xy=(tgt_x - 0.07, y), xytext=(x_x + 0.07, 0.5),
                    arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#7a7a7a",
                                "connectionstyle": "arc3,rad=0.15"})

    ax.annotate("", xy=(tgt_x - 0.07, y1), xytext=(tgt_x + 0.07, y0),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2c6fa6"})
    ax.annotate("", xy=(tgt_x - 0.07, y2), xytext=(tgt_x + 0.07, y1),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2c6fa6"})
    ax.annotate("", xy=(tgt_x + 0.13, y2), xytext=(tgt_x + 0.13, y0),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2c6fa6",
                            "connectionstyle": "arc3,rad=-0.5"})

    ax.text(0.10, 0.92, "features (always present)",
            ha="center", fontsize=10, style="italic", color="#333333")
    ax.text(0.72, 0.92, "targets, in chain order",
            ha="center", fontsize=10, style="italic", color="#333333")
    ax.text(0.72, 0.06,
            "each link sees x plus the previous targets",
            ha="center", fontsize=9, color="#666666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("A classifier chain", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "chain_structure.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  chain_structure.png")


if __name__ == "__main__":
    print("Generating docs/assets/multioutput/")
    fig_chain_structure()
    print("Done.")
