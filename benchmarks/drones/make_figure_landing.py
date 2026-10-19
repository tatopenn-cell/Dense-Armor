"""Generate the PNGs of docs/drones/visual_landing.md.

Two figures, from the library's own primitives (GlobalMotion,
FrameAnomaly) on a synthetic stream labelled "synthetic".

Run from the repository root:

    python benchmarks/drones/make_figure_landing.py
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

from pipelines.drones.visual_landing import run, synthetic_frames

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _frames_plot():
    frames, _, _ = synthetic_frames(
        n=40, shift=(1, 0), seed=0, person_at=20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].imshow(frames[5]["array"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("frame 5 (no person)", fontsize=11, weight="bold")
    axes[0].axis("off")
    axes[1].imshow(frames[30]["array"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("frame 30 (person)", fontsize=11, weight="bold")
    axes[1].axis("off")
    fig.suptitle("Synthetic landing pad, 640x360 frames",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "visual_landing_frames.png", dpi=140)
    plt.close(fig)


def _signals_plot():
    frames, truth, pa = synthetic_frames(
        n=40, shift=(1, 0), seed=0, person_at=20)
    r = run(frames, truth, pa, anomaly_warmup=10)
    t = np.arange(r.n_frames)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(t, r.anomaly, marker="o", color="#2c6fa6",
                 label="FrameAnomaly score")
    axes[0].axvline(pa, color="black", linestyle="--", label="person appears")
    axes[0].set_ylabel("anomaly score")
    axes[0].set_title(
        f"FrameAnomaly before={r.anomaly_before_mean:.3f} "
        f"after={r.anomaly_after_mean:.3f}",
        fontsize=11, weight="bold",
    )
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, r.error_px, marker="o", color="crimson",
                 label="shift error (px)")
    axes[1].set_ylabel("error (px)")
    axes[1].set_xlabel("frame")
    axes[1].set_title(
        f"GlobalMotion error, mean={r.mean_error_px:.3f} px, "
        f"max={r.max_error_px:.3f} px",
        fontsize=11, weight="bold",
    )
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Three primitives, two working signals",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "visual_landing_signals.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic frames)")
    _frames_plot()
    _signals_plot()
    print("Done.")
