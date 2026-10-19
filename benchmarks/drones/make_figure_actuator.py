"""Generate the PNGs of docs/drones/actuator_fault.md.

Two figures, both from the library's own functions on a synthetic
stream labelled "synthetic" (pipelines.drones.io.synthetic_flight).
No real flight data, no downloaded datasets.

- ``actuator_fault_residual.png``: the three residual components
  (commanded minus measured rate) around the fault, with the
  detected episode marked.
- ``actuator_fault_mahalanobis.png``: the Mahalanobis score with
  the threshold.

Run from the repository root:

    python benchmarks/drones/make_figure_actuator.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.drones.actuator_fault import run
from pipelines.drones.io import synthetic_flight

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _residual_plot():
    stream = list(synthetic_flight(n=1000, seed=0, fault="actuator_loss"))
    r = run(stream)
    n = r.n_samples
    res = np.full((n, 3), np.nan)
    for i, s in enumerate(stream):
        cmd = s.get("commanded")
        gyr = s.get("gyro")
        if cmd is not None and gyr is not None:
            res[i] = np.asarray(cmd) - np.asarray(gyr)
    t = np.array([s["t"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, name in enumerate(("r_x", "r_y", "r_z")):
        ax.plot(t, res[:, j], label=name, linewidth=0.9)
    if r.fault_index >= 0:
        ax.axvline(t[r.fault_index], color="black", linestyle="--",
                   label="fault")
    episode_idx = np.flatnonzero(
        np.r_[r.cusum_alarms[0], r.cusum_alarms[1:] & ~r.cusum_alarms[:-1]]
    )
    if len(episode_idx):
        ax.scatter(t[episode_idx], np.zeros(len(episode_idx)),
                   color="red", marker="x", zorder=5,
                   label=f"episode (n={len(episode_idx)})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("residual, rad/s")
    ax.set_title(
        "Actuator loss, synthetic stream, residual of the rate",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "actuator_fault_residual.png", dpi=140)
    plt.close(fig)
    print(f"  actuator_fault_residual.png: fault_index={r.fault_index}, "
          f"episodes={r.n_episodes}, delays={[round(d, 3) for d in r.delays_s]}")


def _mahalanobis_plot():
    stream = list(synthetic_flight(n=1000, seed=0, fault="actuator_loss"))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, r.mahalanobis_scores, color="#2c6fa6",
            label="Mahalanobis score")
    ax.axhline(7.0, color="red", linestyle="--", label="threshold 7")
    if r.fault_index >= 0:
        ax.axvline(t[r.fault_index], color="black", linestyle="--",
                   label="fault")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(
        "Actuator loss, Mahalanobis score on the three residuals",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "actuator_fault_mahalanobis.png", dpi=140)
    plt.close(fig)
    print(f"  actuator_fault_mahalanobis.png: "
          f"max_score={np.nanmax(r.mahalanobis_scores):.2f}, "
          f"above_threshold={(r.mahalanobis_scores > 7.0).sum()}")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic stream)")
    _residual_plot()
    _mahalanobis_plot()
    print("Done.")
