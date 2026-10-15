"""Generate the two PNGs of docs/assets/rules/.

Run from the repository root:

    python docs/assets/rules/make_figure.py

The figures come from the library's own class on a seeded stream.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.rules import AMRulesRegressor

HERE = Path(__file__).resolve().parent
BLUE, RED, GREY = "#2b6ca3", "#c0392b", "#7f8c8d"


def _stream(n: int, seed: int, flip: bool):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        y = 5.0 * (a <= 0.0) if flip else 5.0 * (a > 0.0)
        yield {"a": a, "b": b}, y + 0.1 * float(rng.normal())


def main() -> None:
    m = AMRulesRegressor(n_min=100, seed=0)
    for x, y in _stream(3000, 0, False):
        m.learn_one(x, y)
    cond = m.explain_one({"a": 0.5, "b": 0.0})[0]
    thr = cond[0]["value"]
    pts = np.array([[x["a"], y] for x, y in _stream(400, 5, False)])
    grid = np.linspace(-1, 1, 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=GREY, alpha=0.6, label="samples")
    ax.plot(
        grid,
        [m.predict_one({"a": g, "b": 0.0}) for g in grid],
        color=BLUE,
        lw=2,
        label="rule set",
    )
    ax.axvline(thr, color=RED, ls="--", lw=1.2)
    ax.text(thr + 0.03, 2.4, f"condition found: a > {thr:.3f}", color=RED)
    ax.set_xlabel("a")
    ax.set_ylabel("target")
    ax.set_title("One condition on the feature that matters")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "step_rules.png", dpi=140)
    plt.close(fig)
    print("threshold", round(thr, 4), "rules", len(m.rules_), "removed", m.n_removed_)

    removed = m.n_removed_
    err = []
    for x, y in _stream(2000, 1, True):
        err.append(abs(m.predict_one(x) - y))
        m.learn_one(x, y)
    roll = np.convolve(err, np.ones(50) / 50, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(roll)) + 50, roll, color=BLUE, lw=2)
    ax.set_xlabel("samples after the targets are swapped")
    ax.set_ylabel("absolute error over the last 50 samples")
    ax.set_title("The Page-Hinkley test drops the rules that went wrong")
    fig.tight_layout()
    fig.savefig(HERE / "regime_change.png", dpi=140)
    plt.close(fig)
    print(
        "first100",
        round(float(np.mean(err[:100])), 2),
        "last300",
        round(float(np.mean(err[-300:])), 2),
        "removed after",
        m.n_removed_ - removed,
    )


if __name__ == "__main__":
    main()
