"""Generate the PNGs of docs/drones/visual_landing.md.

Two figures, from the library's own functions on synthetic frames
labelled "synthetic" (pipelines.drones.visual_landing.synthetic_frames).

- ``visual_landing_frames.png``: two consecutive frames and the
  estimated shift.
- ``visual_landing_error.png``: estimated vs true shift over frames,
  and error in pixels.

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
    frames, truth = synthetic_frames(n=8, shift=(4, -2), seed=0)
    r = run(frames, truth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].imshow(frames[0]["array"], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("frame 0", fontsize=11, weight="bold")
    axes[0].axis("off")
    axes[1].imshow(frames[1]["array"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"frame 1 (true shift {r.true_dx:+d}, {r.true_dy:+d} px, "
        f"est {r.est_dx[1]:.0f}, {r.est_dy[1]:.0f})",
        fontsize=11, weight="bold",
    )
    axes[1].axis("off")
    fig.suptitle("Synthetic landing pad, 640x360 frames",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "visual_landing_frames.png", dpi=140)
    plt.close(fig)
    print(f"  visual_landing_frames.png: true=({r.true_dx},{r.true_dy}) "
          f"est=({r.est_dx[1]:.2f},{r.est_dy[1]:.2f})")


def _error_plot():
    frames, truth = synthetic_frames(n=20, shift=(3, -1), seed=0)
    r = run(frames, truth)
    t = np.arange(r.n_frames)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    axes[0].plot(t, r.est_dx, marker="o", label="estimated dx")
    axes[0].plot(t, [truth[i][0] - truth[i-1][0] if i > 0 else 0
                     for i in t], linestyle="--", label="true dx")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("dx (pixels)")
    axes[0].set_title("horizontal shift per frame", fontsize=11, weight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, r.error_px, marker="o", color="crimson",
                 label="error (px)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("Euclidean error (pixels)")
    axes[1].set_title(
        f"error, mean={r.mean_error_px:.3f} px, "
        f"max={r.max_error_px:.3f} px",
        fontsize=11, weight="bold",
    )
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("GlobalMotion recovers the shift exactly",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "visual_landing_error.png", dpi=140)
    plt.close(fig)
    print(f"  visual_landing_error.png: mean_err={r.mean_error_px:.4f} px, "
          f"max_err={r.max_error_px:.4f} px, "
          f"ms_per_frame={r.time_per_frame_ms:.0f}")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic frames)")
    _frames_plot()
    _error_plot()
    print("Done.")
