"""Generate the PNGs of docs/drones/link_health.md.

Two figures, from the library's own functions on a synthetic MQTT
stream labelled "synthetic" (pipelines.drones.link_health.synthetic_mqtt).

- ``link_health_signals.png``: inter-arrival and RSSI, with the fault
  marked, on degradation and dropout.
- ``link_health_alarms.png``: the combined alarm per packet.

Run from the repository root:

    python benchmarks/drones/make_figure_link.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.drones.link_health import run, synthetic_mqtt

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _with_fault(packets, kind):
    fa = round(0.6 * len(packets))
    for i, p in enumerate(packets):
        if i >= fa:
            p["fault_started"] = True
            p["fault_kind"] = kind
    return packets


def _signals_plot():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for col, kind in enumerate(("link_degradation", "link_dropout")):
        packets = _with_fault(synthetic_mqtt(n=800, seed=0, fault=kind), kind)
        r = run(packets)
        t = np.array([p["t"] for p in packets])
        if r.fault_index >= 0:
            t_fault = t[r.fault_index]
        else:
            t_fault = float("nan")
        axes[0, col].plot(t, r.inter_arrival * 1000.0, linewidth=0.7,
                          color="#2c6fa6")
        axes[0, col].axvline(t_fault, color="black", linestyle="--",
                             label="fault")
        axes[0, col].set_title(f"inter-arrival, {kind}", fontsize=10,
                               weight="bold")
        axes[0, col].set_ylabel("inter-arrival (ms)")
        axes[0, col].set_xlabel("time (s)")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(t, r.rssi, linewidth=0.7, color="#2c6fa6")
        axes[1, col].axvline(t_fault, color="black", linestyle="--")
        axes[1, col].set_title(f"RSSI, {kind}", fontsize=10, weight="bold")
        axes[1, col].set_ylabel("RSSI (dBm)")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].grid(True, alpha=0.3)
    fig.suptitle("Synthetic MQTT link, 800 packets", fontsize=12,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "link_health_signals.png", dpi=140)
    plt.close(fig)


def _alarms_plot():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, kind in zip(axes, ("link_degradation", "link_dropout")):
        packets = _with_fault(synthetic_mqtt(n=800, seed=0, fault=kind), kind)
        r = run(packets)
        t = np.array([p["t"] for p in packets])
        if r.fault_index >= 0:
            t_fault = t[r.fault_index]
        else:
            t_fault = float("nan")
        ax.plot(t, r.alarm.astype(int), linewidth=0.7, color="crimson")
        ax.axvline(t_fault, color="black", linestyle="--", label="fault")
        ax.set_title(f"combined alarm, {kind} (episodes={r.n_episodes})",
                     fontsize=10, weight="bold")
        ax.set_ylabel("alarm")
        ax.set_yticks([0, 1])
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("One episode per sustained fault", fontsize=12,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "link_health_alarms.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic MQTT)")
    _signals_plot()
    _alarms_plot()
    print("Done.")
