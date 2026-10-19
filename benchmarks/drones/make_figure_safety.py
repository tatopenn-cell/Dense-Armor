"""Generate the PNGs of docs/drones/safety_filter.md.

Two figures, from the library's own functions on synthetic lidar scans
(benchmarks.lidar_sim.scan, labelled "simulated" in the page).

- ``safety_filter_velocity.png``: the requested velocity vs the filtered
  velocity on a clean stream and on a dirty one, at v_max = 2 m/s.
- ``safety_filter_tradeoff.png``: fraction of modified scans vs v_max,
  on the clean stream and on the dirty one.

Run from the repository root:

    python benchmarks/drones/make_figure_safety.py
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

import benchmarks.lidar_sim as ls
from pipelines.drones.safety_filter import run

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _velocity_plot():
    rng = np.random.default_rng(0)
    clean = [ls.scan(i, rng, None, None) for i in range(200)]
    rng2 = np.random.default_rng(0)
    dirty = [ls.scan(20000 + i, rng2, "dirt", 5000) for i in range(200)]
    r_clean = run(clean, v_max=2.0)
    r_dirty = run(dirty, v_max=2.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    t = np.arange(200)
    axes[0].plot(t, -r_clean.u_safe, color="#2c6fa6", label="filtered speed")
    axes[0].axhline(2.0, color="black", linestyle="--", label="requested 2 m/s")
    axes[0].set_title("clean stream", fontsize=11, weight="bold")
    axes[0].set_xlabel("scan")
    axes[0].set_ylabel("speed toward obstacle (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, -r_dirty.u_safe, color="crimson", label="filtered speed")
    axes[1].axhline(2.0, color="black", linestyle="--", label="requested 2 m/s")
    axes[1].set_title("dirty stream", fontsize=11, weight="bold")
    axes[1].set_xlabel("scan")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("CBF filter at v_max = 2 m/s", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "safety_filter_velocity.png", dpi=140)
    plt.close(fig)
    print(f"  safety_filter_velocity.png: "
          f"clean max_change={r_clean.max_change:.3f}, "
          f"dirty max_change={r_dirty.max_change:.3f}")


def _tradeoff_plot():
    rng = np.random.default_rng(0)
    clean = [ls.scan(i, rng, None, None) for i in range(200)]
    rng2 = np.random.default_rng(0)
    dirty = [ls.scan(20000 + i, rng2, "dirt", 5000) for i in range(200)]
    v_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                0.75, 1.0, 1.5, 2.0]
    frac_clean = []
    frac_dirty = []
    for v in v_values:
        frac_clean.append(run(clean, v_max=v).fraction_modified)
        frac_dirty.append(run(dirty, v_max=v).fraction_modified)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(v_values, frac_clean, marker="o", label="clean", color="#2c6fa6")
    ax.plot(v_values, frac_dirty, marker="s", label="dirty", color="crimson")
    ax.set_xlabel("requested speed v_max (m/s)")
    ax.set_ylabel("fraction of scans modified")
    ax.set_title(
        "The CBF depends on distance and speed together",
        fontsize=12, weight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "safety_filter_tradeoff.png", dpi=140)
    plt.close(fig)
    print(f"  safety_filter_tradeoff.png: "
          f"clean at 0.25={frac_clean[v_values.index(0.25)]:.2f}, "
          f"dirty at 0.10={frac_dirty[v_values.index(0.10)]:.2f}")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (simulated lidar)")
    _velocity_plot()
    _tradeoff_plot()
    print("Done.")
