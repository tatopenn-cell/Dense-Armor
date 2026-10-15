"""Generate the four PNGs of docs/assets/forest/.

Run from the repository root:

    python docs/assets/forest/make_figure.py

Every figure comes from the library's own classes. No external data and
no hard-coded numbers: the streams are ``DriftStream`` and a seeded noisy
sine, the curves are what the classes predict.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.forest import (
    AdaptiveRandomForestClassifier,
    AggregatedMondrianForestRegressor,
)
from dense_armor.utility.forest.amf import _AmfTree, _Mean
from dense_armor.utility.tree import HoeffdingAdaptiveTreeClassifier

HERE = Path(__file__).resolve().parent
N, D, CUT = 2000, 4, 1000
NAMES = [f"x{i}" for i in range(D)]
BLUE, RED, GREY = "#2b6ca3", "#c0392b", "#7f8c8d"


def _rolling(a: np.ndarray, w: int = 50) -> np.ndarray:
    return np.convolve(a, np.ones(w) / w, mode="valid")


def _run_classifiers() -> tuple:
    arf_acc, hat_acc, events = [], [], None
    for seed in range(5):
        X, y = DriftStream(
            n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed
        ).data()
        arf = AdaptiveRandomForestClassifier(n_models=10, seed=seed)
        hat = HoeffdingAdaptiveTreeClassifier(grace_period=50, drift_detector=ADWIN())
        a, h, w, d = [], [], [], []
        for xi, yi in zip(X, y):
            x = dict(zip(NAMES, map(float, xi)))
            a.append(float(arf.predict_one(x) == yi))
            h.append(float(hat.predict_one(x) == yi))
            arf.learn_one(x, int(yi))
            hat.learn_one(x, int(yi))
            w.append(arf.n_warnings_)
            d.append(arf.n_drifts_)
        arf_acc.append(_rolling(np.array(a)))
        hat_acc.append(_rolling(np.array(h)))
        if seed == 0:
            events = (np.array(w), np.array(d))
    return np.median(arf_acc, axis=0), np.median(hat_acc, axis=0), events


def figure_recovery(arf_acc: np.ndarray, hat_acc: np.ndarray) -> None:
    t = np.arange(len(arf_acc)) + 50
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, hat_acc, color=GREY, lw=2, label="one adaptive tree")
    ax.plot(t, arf_acc, color=BLUE, lw=2, label="adaptive random forest, 10 trees")
    ax.axvline(CUT, color=RED, ls="--", lw=1.2)
    ax.text(CUT + 15, 0.07, "the stream changes", color=RED)
    ax.set_xlabel("sample")
    ax.set_ylabel("accuracy over the last 50 samples")
    ax.set_ylim(0, 1.03)
    ax.set_title("After the change the forest is right again sooner")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "change_recovery.png", dpi=140)
    plt.close(fig)


def figure_events(events: tuple) -> None:
    w, d = events
    t = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(
        t,
        w,
        color=BLUE,
        lw=2,
        where="post",
        label="warnings (a background tree starts)",
    )
    ax.step(
        t,
        d,
        color=RED,
        lw=2,
        where="post",
        label="drifts (a background tree takes over)",
    )
    ax.axvline(CUT, color=GREY, ls="--", lw=1.2)
    ax.set_xlabel("sample")
    ax.set_ylabel("total so far, all 10 trees")
    ax.set_title("What the detectors of the trees signalled")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "detector_events.png", dpi=140)
    plt.close(fig)


def figure_aggregation() -> None:
    rng = np.random.default_rng(0)
    xs = rng.random(300)
    ys = np.clip(0.8 * np.sin(2 * np.pi * xs) + 0.15 * rng.normal(size=300), -1, 1)
    f = AggregatedMondrianForestRegressor(n_trees=10, seed=0)
    for x, y in zip(xs, ys):
        f.learn_one({"x": float(x)}, float(y))
    grid = np.linspace(0.01, 0.99, 200)
    agg = [f.predict_one({"x": float(g)}) for g in grid]
    leaf = [f.predict_unaggregated_one({"x": float(g)}) for g in grid]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(xs, ys, s=8, color=GREY, alpha=0.5, label="samples")
    ax.plot(
        grid,
        0.8 * np.sin(2 * np.pi * grid),
        color="black",
        lw=1.2,
        ls=":",
        label="the curve behind the samples",
    )
    ax.plot(grid, leaf, color=RED, lw=1.4, label="the leaf alone")
    ax.plot(grid, agg, color=BLUE, lw=2.2, label="all prunings, weighted")
    ax.set_xlabel("x")
    ax.set_ylabel("prediction")
    ax.set_title("Weighting every pruning smooths what a single leaf memorises")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "aggregation.png", dpi=140)
    plt.close(fig)


def _regret(structured: bool, seed: int, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    if structured:
        Y = np.clip(np.sin(6 * X[:, 0]) + 0.1 * rng.normal(size=n), -1, 1)
    else:
        Y = np.clip(0.3 * rng.normal(size=n), -1, 1)
    fc = _Mean()
    tree = _AmfTree(2, 0.0, 1.0, 1.0 / 8.0, fc, np.random.default_rng(seed + 1000))
    root = fc.empty()
    out = np.zeros(n)
    acc = 0.0
    for i, (x, y) in enumerate(zip(X, Y)):
        tree.add_point(x)
        p = tree.predict(x)
        acc += (p - y) ** 2 - fc.loss(root, y)
        fc.update(root, y)
        tree.update_weights(x, y)
        out[i] = acc
    return out


def figure_regret() -> None:
    bound = 8 * math.log(2.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for seed in range(3):
        a1.plot(_regret(False, seed), color=BLUE, lw=1.6)
        a2.plot(_regret(True, seed), color=RED, lw=1.6)
    a1.axhline(bound, color="black", ls="--", lw=1.2)
    a1.text(5, bound - 0.9, "bound of the paper, 8 log 2", fontsize=9)
    a1.set_ylim(-1.5, bound + 1.5)
    a1.set_title("targets that do not depend on x")
    a2.set_title("targets that follow a curve of x")
    for ax in (a1, a2):
        ax.axhline(0.0, color=GREY, lw=0.8)
        ax.set_xlabel("sample")
    a1.set_ylabel("loss of the forest minus loss of the root alone")
    fig.tight_layout()
    fig.savefig(HERE / "regret.png", dpi=140)
    plt.close(fig)


def main() -> None:
    arf_acc, hat_acc, events = _run_classifiers()
    figure_recovery(arf_acc, hat_acc)
    figure_events(events)
    figure_aggregation()
    figure_regret()
    print("warnings, drifts seed 0:", int(events[0][-1]), int(events[1][-1]))


if __name__ == "__main__":
    main()
