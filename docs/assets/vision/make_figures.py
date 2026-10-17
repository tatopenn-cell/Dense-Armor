"""Generate the figures of docs/vision/.

Every quantity comes from a Dense-Armor primitive on a synthetic stream
that this script builds itself. numpy builds the inputs; matplotlib
only draws. The script is deterministic: two runs give the same PNGs
and the same figures.json.

Run from the repository root:

    python docs/assets/vision/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.vision.features import FrameFeatures
from dense_armor.utility.vision.streams import ArrayStream, CubeStream

HERE = Path(__file__).resolve().parent
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 9,
})

FIG = {}


def scene(seed, y=20, x=20, size=8, bg=0.1, sq=1.0):
    rng = np.random.default_rng(seed)
    f = rng.normal(bg, 0.01, (64, 64)).astype(np.float32)
    f[y:y + size, x:x + size] = sq
    return f


def _save(fig, page, name):
    fig.tight_layout()
    d = HERE / page
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=150,
                pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"  {page}/{name}.png")


def fig_01_frames():
    frames = np.stack([scene(0, 20 + 2 * i, 10 + 3 * i) for i in range(5)])
    s = ArrayStream(frames, fps=10.0)
    frames_read = list(s)
    ts = [f.t for f in frames_read]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    for ax, f, t in zip(axes, frames_read, ts):
        ax.imshow(f.array, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t = {t:.1f} s", fontsize=8)
        ax.axis("off")
    _save(fig, "frames", "five_frames")
    FIG["fig01_times"] = [float(x) for x in ts]


def fig_02_thermal():
    rng = np.random.default_rng(0)
    cube = np.empty((1, 64, 64, 4), dtype=np.float32)
    for b in range(4):
        cube[0, :, :, b] = rng.normal(300.0, 5.0, (64, 64))
    cube[0, 20:28, 20:28, :] += 50.0
    s = CubeStream(cube, fps=30.0, gray=False)
    f = next(iter(s))
    bands = f.array
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for b, ax in enumerate(axes):
        im = ax.imshow(bands[:, :, b], cmap="gray", vmin=280, vmax=360)
        ax.set_title(f"band {b}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    _save(fig, "frames", "thermal_bands")
    FIG["fig02_shape"] = list(bands.shape)
    FIG["fig02_hot_mean"] = float(bands[20:28, 20:28, :].mean())
    bg = np.concatenate([bands[:20].ravel(), bands[28:].ravel()])
    FIG["fig02_bg_mean"] = float(bg.mean())


PAGES = [
    ("fig_01_frames", fig_01_frames),
    ("fig_02_thermal", fig_02_thermal),
]


if __name__ == "__main__":
    print("Generating docs/assets/vision/")
    for name, fn in PAGES:
        fn()
    (HERE / "figures.json").write_text(
        json.dumps(FIG, indent=1, default=float)
    )
    print(f"figures.json: {len(FIG)} keys")
    print("Done.")
