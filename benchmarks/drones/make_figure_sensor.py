"""Generate the PNGs of docs/drones/sensor_consistency.md.

Two figures, from the library's own functions on a synthetic stream
labelled "synthetic" (pipelines.drones.io.synthetic_flight).

- ``sensor_consistency_residual.png``: the three components of
  gps_v minus the IMU-integrated velocity, around the ramp.
- ``sensor_consistency_score.png``: the Euclidean norm of the
  residual, with the episodes marked.

Run from the repository root:

    python benchmarks/drones/make_figure_sensor.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.drones.io import synthetic_flight
from pipelines.drones.sensor_consistency import run

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _residual_plot():
    stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, name in enumerate(("north", "east", "down")):
        ax.plot(t, r.residual[:, j], label=name, linewidth=0.9)
    if r.fault_index >= 0:
        ax.axvline(t[r.fault_index], color="black", linestyle="--",
                   label="attack starts")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gps_v - imu_v (m/s)")
    ax.set_title(
        "GPS ramp vs integrated IMU, synthetic stream",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "sensor_consistency_residual.png", dpi=140)
    plt.close(fig)
    print(f"  residual: fault_index={r.fault_index}, "
          f"episodes={r.n_episodes}, "
          f"delays={[round(d, 3) for d in r.delays_s]}")


def _score_plot():
    stream = list(synthetic_flight(n=800, seed=0, fault="gps_ramp"))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, r.score, color="#2c6fa6", label="Euclidean norm of residual")
    if r.fault_index >= 0:
        ax.axvline(t[r.fault_index], color="black", linestyle="--",
                   label="attack starts")
    starts = np.flatnonzero(
        np.r_[r.alarms[0], r.alarms[1:] & ~r.alarms[:-1]])
    if len(starts):
        ax.scatter(t[starts], r.score[starts], color="red", marker="x",
                   zorder=5, label=f"episode (n={len(starts)})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(
        "Euclidean norm of (gps_v - imu_v), synthetic stream",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "sensor_consistency_score.png", dpi=140)
    plt.close(fig)
    print(f"  score: fault_index={r.fault_index}, "
          f"score[-1]={r.score[-1]:.3f}, "
          f"max={np.nanmax(r.score):.3f}")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic stream)")
    _residual_plot()
    _score_plot()
    print("Done.")
