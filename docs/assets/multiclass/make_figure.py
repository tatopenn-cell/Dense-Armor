"""Generate the figures of docs/multiclass/.

Run from the repository root:

    python docs/assets/multiclass/make_figure.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from dense_armor.utility.optim import SGD

HERE = Path(__file__).resolve().parent


def _factory():
    return LogisticRegression(optimizer=SGD(lr=0.5), intercept=True)


def _stream(n=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.2]])
    X, y = [], []
    for i in range(n):
        c = i % 3
        p = centers[c] + rng.normal(0, 0.4, size=2)
        X.append({"a": float(p[0]), "b": float(p[1])})
        y.append(int(c))
    return X, y


def fig_decision_regions():
    X, y = _stream()
    for Cls, name in [
        (OneVsRestClassifier, "One-vs-rest"),
        (OneVsOneClassifier, "One-vs-one (round robin)"),
        (OutputCodeClassifier, "Output code"),
    ]:
        clf = Cls(binary_factory=_factory)
        for _ in range(20):
            for x, yy in zip(X, y):
                clf.learn_one(x, yy)
        gx, gy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
        Z = np.zeros(gx.shape, dtype=int)
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                p = clf.predict_one({"a": float(gx[i, j]), "b": float(gy[i, j])})
                Z[i, j] = int(p) if p is not None else 0
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#fafafa")
        ax.set_facecolor("#fafafa")
        ax.contourf(
            gx,
            gy,
            Z,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=["#cfe3f3", "#f3cfcf", "#d5f0d5"],
            alpha=0.7,
        )
        for c, color in zip(range(3), ["#2c6fa6", "#c0392b", "#27ae60"]):
            pts = np.array([[X[i]["a"], X[i]["b"]] for i in range(len(X)) if y[i] == c])
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                edgecolor="white",
                s=25,
                label=f"class {c}",
            )
        ax.set_title(name, fontsize=12, weight="bold")
        ax.legend(loc="upper left")
        ax.set_xlabel("feature a")
        ax.set_ylabel("feature b")
        fig.tight_layout()
        fname = name.split()[0].lower().replace("-", "_") + "_regions.png"
        fig.savefig(HERE / fname, dpi=140, facecolor="#fafafa")
        plt.close(fig)
        print(f"  {fname}")


if __name__ == "__main__":
    print("Generating docs/assets/multiclass/")
    fig_decision_regions()
    print("Done.")
