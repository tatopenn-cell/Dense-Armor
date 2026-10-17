"""Generate the four figures of docs/assets/ensemble/.

The numbers come from benchmarks/ensemble_quality.json (20 seeds,
DriftStream sudden, 2000 samples per seed). No number is written by
hand: every value comes from the JSON, so the figure is regenerated
when the benchmark is.

Run from the repository root:

    python docs/assets/ensemble/make_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "benchmarks" / "ensemble_quality.json"

BLUE = "#2c6fa6"
ORANGE = "#d97706"
GREEN = "#15803d"
RED = "#b91c1c"
GRAY = "#6b7280"
COLORS = {"tree": GRAY, "bagging": BLUE, "boosting": GREEN,
          "voting": ORANGE, "stacking": RED}

NAMES = ["tree", "bagging", "boosting", "voting", "stacking"]


def _load():
    return json.loads(DATA.read_text())


def fig_accuracy(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(NAMES))
    med = [data["classifiers"][n]["accuracy_median"] for n in NAMES]
    iqr = [data["classifiers"][n]["accuracy_iqr"] for n in NAMES]
    bars = ax.bar(x, med, yerr=iqr, capsize=6,
                  color=[COLORS[n] for n in NAMES],
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("accuracy, median over 20 seeds")
    ax.set_title("Accuracy on DriftStream sudden, one change")
    ax.set_ylim(0.85, 1.0)
    for bar, m in zip(bars, med):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "ensemble_accuracy.png", dpi=140)
    plt.close(fig)
    print("  ensemble_accuracy.png")


def fig_cost(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(NAMES))
    ms = [data["classifiers"][n]["ms_per_sample_median"] for n in NAMES]
    bars = ax.bar(x, ms, color=[COLORS[n] for n in NAMES],
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("ms per sample, median")
    ax.set_title("Cost of one learn_one and one predict_one")
    for bar, v in zip(bars, ms):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "ensemble_cost.png", dpi=140)
    plt.close(fig)
    print("  ensemble_cost.png")


def fig_ewa(data):
    e = data["ewa"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["EWA", "best member", "Hedge bound"]
    vals = [e["cumulative_loss_median"], e["best_member_loss_median"],
            e["bound_median"]]
    colors = [BLUE, GREEN, GRAY]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black",
                  linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 20,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("cumulative loss, median")
    ax.set_title("Exponentially weighted average on the regression stream")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "ensemble_ewa.png", dpi=140)
    plt.close(fig)
    print("  ensemble_ewa.png")


def fig_a12(data):
    pairs = data["pairwise_accuracy"]
    labels = []
    a12 = []
    lo = []
    hi = []
    for p in pairs:
        labels.append(f"{p['a']} vs {p['b']}")
        a12.append(p["a12"])
        lo.append(p["a12"] - p["a12_ci95"][0])
        hi.append(p["a12_ci95"][1] - p["a12"])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.errorbar(a12, y, xerr=[lo, hi], fmt="o", color=BLUE,
                ecolor=GRAY, capsize=4, markersize=8)
    ax.axvline(0.5, color=RED, linestyle="--", linewidth=1.0,
               label="0.5: no difference")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("A12 with bootstrap 95% interval")
    ax.set_title("Effect size of each ensemble against one tree")
    ax.set_xlim(0.35, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "ensemble_a12.png", dpi=140)
    plt.close(fig)
    print("  ensemble_a12.png")


if __name__ == "__main__":
    print("Generating docs/assets/ensemble/")
    data = _load()
    fig_accuracy(data)
    fig_cost(data)
    fig_ewa(data)
    fig_a12(data)
    print("Done.")
