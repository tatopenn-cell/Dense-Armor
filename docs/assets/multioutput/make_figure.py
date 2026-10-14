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

from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.optim import SGD

HERE = Path(__file__).resolve().parent


def _factory():
    return LogisticRegression(optimizer=SGD(lr=0.5), intercept=True)


def fig_chain_structure():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    ax.axis("off")
    xs = [0.08, 0.32, 0.56, 0.80]
    ys = [0.5, 0.5, 0.5, 0.5]
    labels = ["x", "y0", "y1", "y2"]
    for x, y, lbl in zip(xs, ys, labels):
        ax.add_patch(plt.Circle((x, y), 0.05, color="#2c6fa6", zorder=3))
        ax.text(
            x,
            y,
            lbl,
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            zorder=4,
        )
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.07, ys[i + 1]),
            xytext=(xs[i] + 0.07, ys[i]),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#555555"},
        )
    ax.text(
        0.44,
        0.72,
        "the chain: each target joins the input of the next link",
        ha="center",
        fontsize=11,
        style="italic",
    )
    ax.text(0.08, 0.3, "features", ha="center", fontsize=10)
    ax.text(
        0.56,
        0.3,
        "targets (true at training, predicted at prediction)",
        ha="center",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("A classifier chain", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "chain_structure.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  chain_structure.png")


if __name__ == "__main__":
    print("Generating docs/assets/multioutput/")
    fig_chain_structure()
    print("Done.")
