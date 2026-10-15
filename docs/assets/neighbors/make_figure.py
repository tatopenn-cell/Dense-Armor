"""Generate the figures of docs/neighbors/.

Run from the repository root:

    python docs/assets/neighbors/make_figure.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.neighbors import KNNClassifier

HERE = Path(__file__).resolve().parent


def fig_decision_boundary():
    """A 2-D two-class problem, coloured by the k-NN vote."""
    rng = np.random.default_rng(0)
    n = 60
    A = rng.normal(loc=(-1.0, -1.0), scale=0.5, size=(n, 2))
    B = rng.normal(loc=(1.0, 1.0), scale=0.5, size=(n, 2))
    X = np.vstack([A, B])
    y = np.array(["A"] * n + ["B"] * n)

    clf = KNNClassifier(window_size=200, n_neighbors=5)
    for i in range(len(X)):
        clf.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, str(y[i]))

    gx, gy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    Z = np.empty(gx.shape, dtype=int)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            p = clf.predict_proba_one({"a": float(gx[i, j]), "b": float(gy[i, j])})
            Z[i, j] = 0 if p.get("A", 0) > p.get("B", 0) else 1

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    ax.contourf(
        gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f3", "#f3cfcf"], alpha=0.7
    )
    ax.scatter(A[:, 0], A[:, 1], c="#2c6fa6", edgecolor="white", s=30, label="class A")
    ax.scatter(B[:, 0], B[:, 1], c="#c0392b", edgecolor="white", s=30, label="class B")
    ax.set_title(
        "k-NN with k=5: the boundary follows the data, not a shape",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("feature a")
    ax.set_ylabel("feature b")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "decision_boundary.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  decision_boundary.png")


def fig_forget():
    """A window that forgets: the oldest sample drops when the window fills."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    n_show = 15
    window_size = 6
    positions = list(range(n_show))
    # the window keeps the last 6 positions; mark them
    for i in positions:
        in_window = i >= n_show - window_size
        color = "#2c6fa6" if in_window else "#d0d0d0"
        ax.scatter(i, 0, s=200, c=color, edgecolor="black", zorder=3)
        ax.text(i, -0.15, str(i), ha="center", fontsize=9)
    ax.annotate(
        "the window (last 6 samples)",
        xy=(n_show - window_size / 2, 0.05),
        xytext=(n_show - window_size / 2, 0.25),
        ha="center",
        arrowprops={"arrowstyle": "->"},
        fontsize=11,
        weight="bold",
    )
    ax.annotate(
        "forgotten: one pop, no recomputation",
        xy=(0, 0.05),
        xytext=(2, 0.25),
        ha="center",
        arrowprops={"arrowstyle": "->"},
        fontsize=11,
    )
    ax.set_ylim(-0.4, 0.5)
    ax.set_xlim(-1, n_show)
    ax.axis("off")
    ax.set_title(
        "A sliding window: add on the right, forget on the left, O(1) both ways",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(HERE / "forget_window.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  forget_window.png")


if __name__ == "__main__":
    print("Generating docs/assets/neighbors/")
    fig_decision_boundary()
    fig_forget()
    print("Done.")
