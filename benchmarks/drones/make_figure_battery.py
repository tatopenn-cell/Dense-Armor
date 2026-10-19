"""Generate the PNGs of docs/drones/battery_endurance.md.

Two figures, from the library's own functions on a synthetic stream
labelled "synthetic" (pipelines.drones.io.synthetic_flight).

- ``battery_endurance_voltage.png``: predicted voltage, conformal
  interval, measured voltage.
- ``battery_endurance_time.png``: remaining flight time over the
  stream, with the q10/q50/q90 band.

Run from the repository root:

    python benchmarks/drones/make_figure_battery.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.drones.battery_endurance import run
from pipelines.drones.io import synthetic_flight

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _voltage_plot():
    stream = list(synthetic_flight(n=600, seed=0))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    v_true = np.array([s["battery_v"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, v_true, color="black", linewidth=0.9, label="measured V")
    ax.plot(t, r.v_pred, color="#2c6fa6", linewidth=0.9, label="predicted V")
    ax.fill_between(t, r.v_lo, r.v_hi, color="#2c6fa6", alpha=0.2,
                    label="conformal interval")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("battery voltage (V)")
    ax.set_title(
        "Battery voltage, online regression with conformal interval",
        fontsize=12, weight="bold",
    )
    ax.set_ylim(8.0, 15.0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "battery_endurance_voltage.png", dpi=140)
    plt.close(fig)
    print(f"  battery_endurance_voltage.png: coverage={r.coverage:.3f}, "
          f"width={r.mean_width:.2f} V")


def _time_plot():
    stream = list(synthetic_flight(n=600, seed=0))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    valid = np.isfinite(r.t_remaining)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t[valid], r.t_remaining[valid] / 60.0,
            color="#2c6fa6", linewidth=0.9, label="remaining time")
    ax.axhline(r.t_remaining_q10 / 60.0, color="red", linestyle=":",
               label=f"q10 = {r.t_remaining_q10/60:.1f} min")
    ax.axhline(r.t_remaining_q50 / 60.0, color="red", linestyle="--",
               label=f"q50 = {r.t_remaining_q50/60:.1f} min")
    ax.axhline(r.t_remaining_q90 / 60.0, color="red", linestyle=":",
               label=f"q90 = {r.t_remaining_q90/60:.1f} min")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("remaining flight time (min)")
    ax.set_title(
        "Remaining flight time, capacity_wh=63, soc_min=0.10",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "battery_endurance_time.png", dpi=140)
    plt.close(fig)
    print(f"  battery_endurance_time.png: "
          f"q10/q50/q90 = {r.t_remaining_q10/60:.1f}/"
          f"{r.t_remaining_q50/60:.1f}/{r.t_remaining_q90/60:.1f} min")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic stream)")
    _voltage_plot()
    _time_plot()
    print("Done.")
