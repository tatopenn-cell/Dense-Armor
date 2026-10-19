"""Generate the PNGs of docs/drones/static_vs_online.md.

Two figures, from the library's own functions on a synthetic stream
labelled "synthetic" (pipelines.drones.io.synthetic_flight).

- ``static_vs_online_accuracy.png``: per-seed accuracy of the batch
  and online models over 20 seeds, and the diff.
- ``static_vs_online_time.png``: time per sample and peak memory,
  batch vs online.

Run from the repository root:

    python benchmarks/drones/make_figure_static.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.drones.static_vs_online import run

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _accuracy_plot(r):
    t = np.arange(r.n_seeds)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, r.accuracy_batch, marker="o", label="batch (fit_batch)",
            color="#2c6fa6")
    ax.plot(t, r.accuracy_online, marker="s", label="online (learn_one)",
            color="crimson")
    ax.set_xlabel("seed")
    ax.set_ylabel("accuracy on the test half")
    ax.set_title(
        f"Batch {r.median_accuracy_batch:.4f} vs online "
        f"{r.median_accuracy_online:.4f}, {r.n_seeds} seeds",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "static_vs_online_accuracy.png", dpi=140)
    plt.close(fig)


def _cost_plot(r):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    x = np.arange(2)
    ax.bar(x, [r.time_per_sample_batch_ms, r.time_per_sample_online_ms],
           color=["#2c6fa6", "crimson"])
    ax.set_xticks(x)
    ax.set_xticklabels(["batch", "online"])
    ax.set_ylabel("time per sample (ms)")
    ax.set_title("time per sample", fontsize=11, weight="bold")
    for i, v in enumerate([r.time_per_sample_batch_ms,
                            r.time_per_sample_online_ms]):
        ax.text(i, v, f"{v:.4f} ms", ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(x, [r.memory_batch_kb, r.memory_online_kb],
           color=["#2c6fa6", "crimson"])
    ax.set_xticks(x)
    ax.set_xticklabels(["batch", "online"])
    ax.set_ylabel("peak memory (KB)")
    ax.set_title("peak memory", fontsize=11, weight="bold")
    for i, v in enumerate([r.memory_batch_kb, r.memory_online_kb]):
        ax.text(i, v, f"{v:.0f} KB", ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Cost of the batch vs online bagging",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "static_vs_online_time.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print("Running 20 seeds (about 40 s) ...")
    r = run(n_seeds=20, n_samples=2000, n_models=5)
    print(f"  batch:  median={r.median_accuracy_batch:.4f}, "
          f"t={r.time_per_sample_batch_ms:.4f} ms, "
          f"mem={r.memory_batch_kb:.0f} KB")
    print(f"  online: median={r.median_accuracy_online:.4f}, "
          f"t={r.time_per_sample_online_ms:.4f} ms, "
          f"mem={r.memory_online_kb:.0f} KB")
    _accuracy_plot(r)
    _cost_plot(r)
    print("Done.")
