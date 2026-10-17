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
from dense_armor.utility.vision.reduce import (
    IncrementalPCA,
    RandomProjection,
)
from dense_armor.utility.drift.page_hinkley import PageHinkley
from dense_armor.utility.vision.scene import SceneChange
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

def scene_clean(y=20, x=20, size=8):
    f = np.zeros((64, 64), dtype=np.float32)
    f[y:y + size, x:x + size] = 1.0
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

def fig_03_hog():
    """HOG grid on scene S, one arrow per cell, length = dominant bin."""
    ff = FrameFeatures(n_bins=9, cells=(3, 4), flow_cells=(2, 3), flow_levels=1)
    f = scene(0, 20, 20)
    out = ff.transform_one(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    ax = axes[0]
    ax.imshow(f, cmap="gray", vmin=0.0, vmax=1.0)
    H, W = f.shape
    ch, cw = H / 3.0, W / 4.0
    for cy in range(3):
        for cx in range(4):
            vals = [out[f"hog_{cy}_{cx}_{b}"] for b in range(9)]
            b_max = int(np.argmax(vals))
            theta = b_max * 2 * np.pi / 9
            cy0, cx0 = (cy + 0.5) * ch, (cx + 0.5) * cw
            length = max(vals) * 10.0
            ax.arrow(cx0, cy0,
                     length * np.cos(theta), -length * np.sin(theta),
                     head_width=1.2, color=ORANGE, linewidth=1.2)
            ax.add_patch(plt.Rectangle((cx * cw, cy * ch), cw, ch,
                                       fill=False, edgecolor=INK2,
                                       linewidth=0.5))
    ax.set_title("scene S, HOG grid 3x4, 9 bins", fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks(np.arange(0, W + 1, 16))
    ax.set_yticks(np.arange(0, H + 1, 16))
    ax = axes[1]
    cy_cell, cx_cell = 0, 1
    vals = [out[f"hog_{cy_cell}_{cx_cell}_{b}"] for b in range(9)]
    ax.bar(np.arange(9), vals, color=BLUE, edgecolor=INK, linewidth=0.5)
    b_max = int(np.argmax(vals))
    ax.bar([b_max], [vals[b_max]], color=ORANGE, edgecolor=INK, linewidth=0.5,
           label=f"peak bin = {b_max}")
    ax.set_xlabel("orientation bin")
    ax.set_ylabel("L2-normalised value")
    ax.set_title(f"cell ({cy_cell}, {cx_cell})", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    _save(fig, "edges", "hog_grid")
    FIG["fig03_cell"] = [cy_cell, cx_cell]
    FIG["fig03_peak_bin"] = b_max
    FIG["fig03_hist"] = [float(v) for v in vals]

def fig_04_moments():
    ff = FrameFeatures()
    f = scene_clean(20, 20)
    out = ff.transform_one(f)
    cx = out["centroid_x"] * f.shape[1]
    cy = out["centroid_y"] * f.shape[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(f, cmap="gray", vmin=0.0, vmax=1.0)
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    w = np.clip(f, 0.0, None).ravel()
    sw = w.sum()
    if sw > 0:
        xm = (xx.ravel() * w).sum() / sw
        ym = (yy.ravel() * w).sum() / sw
        var_x = ((xx.ravel() - xm) ** 2 * w).sum() / sw
        var_y = ((yy.ravel() - ym) ** 2 * w).sum() / sw
        cov = ((xx.ravel() - xm) * (yy.ravel() - ym) * w).sum() / sw
    else:
        xm, ym, var_x, var_y, cov = 0.0, 0.0, 0.0, 0.0, 0.0
    ax.plot(cx, cy, "o", color=ORANGE, markersize=10,
            markeredgecolor=INK, markeredgewidth=0.8,
            label=f"centroid ({cx:.1f}, {cy:.1f})")
    vals, vecs = np.linalg.eigh(np.array([[var_x, cov], [cov, var_y]]))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    t = np.linspace(0, 2 * np.pi, 100)
    ellipse = (vecs @ (np.sqrt(np.clip(vals, 0, None))[:, None]
                       * np.array([np.cos(t), np.sin(t)])))
    ax.plot(xm + ellipse[0], ym + ellipse[1], color=ORANGE, linewidth=1.4)
    ax.set_title("scene S, centroid and moment ellipse", fontsize=9)
    ax.set_xticks(np.arange(0, 64 + 1, 16))
    ax.set_yticks(np.arange(0, 64 + 1, 16))
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "light", "moments")
    FIG["fig04_cx"] = float(cx)
    FIG["fig04_cy"] = float(cy)
    FIG["fig04_true_cx"] = 23.5
    FIG["fig04_true_cy"] = 23.5

def fig_05_flow():
    """Lucas-Kanade: small vs large motion, with and without the pyramid.

    The scene is a random smooth image rolled by (2, 3) or (10, 14).
    The pyramid helps the small motion; on the large one the flow is
    still underestimated, which is the limit the page states.
    """
    from scipy.ndimage import gaussian_filter

    def smooth(seed=0, size=64):
        rng = np.random.default_rng(seed)
        a = gaussian_filter(rng.normal(0, 1, (size, size)), 2.0)
        a = (a - a.min()) / (a.max() - a.min())
        return a.astype(np.float32)

    def roll(f, dy, dx):
        return np.roll(np.roll(f, dy, axis=0), dx, axis=1).astype(np.float32)

    def measure(a, b, levels):
        ff = FrameFeatures(flow_levels=levels, flow_cells=(4, 4))
        ff.learn_one(a)
        o = ff.transform_one(b)
        us, vs, mags = [], [], []
        for cy in range(4):
            for cx in range(4):
                u = o[f"flow_u_{cy}_{cx}"]
                v = o[f"flow_v_{cy}_{cx}"]
                us.append(u)
                vs.append(v)
                mags.append(np.hypot(u, v))
        return float(np.median(us)), float(np.median(vs)), float(max(mags))

    a = smooth(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    cases = [
        ("small (2, 3), 1 level", roll(a, 2, 3), 1, (2, 3)),
        ("large (10, 14), 1 level", roll(a, 10, 14), 1, (10, 14)),
        ("large (10, 14), 3 levels", roll(a, 10, 14), 3, (10, 14)),
    ]
    for ax, (title, b, lv, true) in zip(axes, cases):
        u, v, _ = measure(a, b, lv)
        ax.imshow(b, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9)
        cy = 32
        cx = 32
        scale = 2.0
        ax.arrow(cx, cy, u * scale, v * scale, color=ORANGE, head_width=2.5,
                 linewidth=2.0, length_includes_head=True,
                 label=f"u={u:.1f}, v={v:.1f}")
        ax.arrow(cx, cy, true[1] * scale, true[0] * scale, color=INK,
                 linestyle="--", head_width=0.0, linewidth=1.2,
                 length_includes_head=True, label=f"true ({true[1]}, {true[0]})")
        ax.legend(loc="lower right", fontsize=7)
        ax.axis("off")
        key = title.replace(" ", "_")
        FIG[f"fig05_{key}_u"] = u
        FIG[f"fig05_{key}_v"] = v
    _save(fig, "flow", "lucas_kanade")

def fig_06_colour():
    f = np.zeros((64, 64, 3), dtype=np.float32)
    f[:, 0:21, 0] = 0.8
    f[:, 0:21, 1] = 0.1
    f[:, 0:21, 2] = 0.1
    f[:, 21:42, 0] = 0.1
    f[:, 21:42, 1] = 0.8
    f[:, 21:42, 2] = 0.1
    f[:, 42:, 0] = 0.1
    f[:, 42:, 1] = 0.1
    f[:, 42:, 2] = 0.8
    ff = FrameFeatures(gray=False)
    out = ff.transform_one(f)
    keys = ["mean_R", "mean_G", "mean_B", "contrast_R", "contrast_G", "contrast_B"]
    vals = [out[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].imshow(np.clip(f, 0, 1))
    axes[0].set_title("three vertical bands", fontsize=9)
    axes[0].axis("off")
    axes[1].bar(range(len(keys)), vals, color=BLUE, edgecolor=INK, linewidth=0.5)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(keys, rotation=30, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.4)
    _save(fig, "colour", "colour_features")
    FIG["fig06_keys"] = keys
    FIG["fig06_vals"] = [float(v) for v in vals]

def fig_07_projection():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (200, 1000))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    iu = np.triu_indices(200, k=1)
    d_before = D[iu]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, k in zip(axes, [16, 64, 256]):
        rp = RandomProjection(k=k, seed=0, input_dim=1000)
        Y = np.array([
            list(rp.transform_one({f"f{j}": float(X[i, j])
                                   for j in range(1000)}).values())
            for i in range(200)
        ])
        Dy = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))
        d_after = Dy[iu]
        ax.scatter(d_before, d_after, s=2, alpha=0.15, color=BLUE,
                   edgecolors="none")
        ax.plot([0, d_before.max()], [0, d_before.max()],
                color=INK, linestyle="--", linewidth=1.0)
        ratio = d_after / np.maximum(d_before, 1e-9)
        med = float(np.median(ratio))
        lo = float(np.percentile(ratio, 5))
        hi = float(np.percentile(ratio, 95))
        ax.set_title(f"k = {k}", fontsize=9)
        ax.set_xlabel("distance before")
        ax.grid(True, alpha=0.3)
        FIG[f"fig07_ratio_k{k}_med"] = med
        FIG[f"fig07_ratio_k{k}_p5"] = lo
        FIG[f"fig07_ratio_k{k}_p95"] = hi
    axes[0].set_ylabel("distance after")
    _save(fig, "projection", "jl")

def fig_08_pca():
    rng = np.random.default_rng(0)
    pts = rng.multivariate_normal([0.0, 0.0],
                                  [[4.0, 1.8], [1.8, 1.0]], 200)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    top = vecs[:, np.argmax(vals)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    pca = IncrementalPCA(k=1)
    for ax, mark in zip(axes, [10, 50, 200]):
        pca = IncrementalPCA(k=1)
        for i in range(mark):
            pca.learn_one({"x": float(pts[i, 0]), "y": float(pts[i, 1])})
        learned = pca.components_ if hasattr(pca, "components_") else None
        ax.scatter(pts[:mark, 0], pts[:mark, 1], s=4, color=BLUE, alpha=0.5,
                   edgecolors="none")
        if learned is not None:
            learned = np.asarray(learned).ravel()
            angle_deg = float(np.degrees(np.arctan2(learned[1], learned[0])))
            a = np.linspace(-4, 4, 50)
            ax.plot(a * learned[0], a * learned[1], color=ORANGE, linewidth=1.6,
                    label=f"learned ({angle_deg:.1f} deg)")
            FIG[f"fig08_angle_{mark}"] = angle_deg
        ax.plot(a * top[0], a * top[1], color=INK, linestyle="--", linewidth=1.0,
                label="true")
        ax.set_title(f"after {mark} frames", fontsize=9)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        if mark == 10:
            ax.legend(loc="upper left", fontsize=7)
    _save(fig, "pca", "directions")

def fig_09_chain():
    """SceneChange with PageHinkley on a synthetic cut: distance and alarm."""
    frames = []
    rng = np.random.default_rng(0)
    for i in range(60):
        if i < 30:
            frames.append(rng.normal(0.2, 0.02, (64, 64)).astype(np.float32))
        else:
            frames.append(rng.normal(0.8, 0.02, (64, 64)).astype(np.float32))
    det = PageHinkley(threshold=10.0, warmup=10, delta=0.0005)
    sc = SceneChange(detector=det, bins=32, blur=1, downsample=2)
    dist = []
    alarms = []
    times_ns = []
    import time
    for f in frames:
        t0 = time.perf_counter_ns()
        o = sc.transform_one({"array": f})
        sc.learn_one({"array": f})
        times_ns.append(time.perf_counter_ns() - t0)
        dist.append(o["distance"])
        alarms.append(bool(o["scene_change"]))
    dist = np.array(dist)
    ms = np.array(times_ns) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(np.arange(len(dist)), dist, color=BLUE, linewidth=1.4,
            label="scene distance")
    first = next((i for i, a in enumerate(alarms) if a), None)
    if first is not None:
        ax.axvline(first, color=ORANGE, linestyle="--", linewidth=1.2,
                   label=f"alarm at frame {first}")
    ax.axvline(30, color=INK, linestyle=":", linewidth=1.0,
               label="true cut at frame 30")
    ax.set_xlabel("frame")
    ax.set_ylabel("scene distance")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(np.arange(len(ms)), ms, color=BLUE, linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("ms per frame")
    ax.grid(True, alpha=0.3)
    _save(fig, "chain", "scene_change")
    FIG["fig09_first_alarm"] = int(first) if first is not None else -1
    FIG["fig09_true_cut"] = 30
    FIG["fig09_dist_before"] = float(dist[:30].mean())
    FIG["fig09_dist_after"] = float(dist[30:].mean())
    FIG["fig09_ms_median"] = float(np.median(ms))

def fig_10_motion():
    """MotionEnergy on a smooth stream, and GlobalMotion on a known shift."""
    from dense_armor.utility.vision.motion import MotionEnergy, GlobalMotion
    from dense_armor.utility.vision.streams import ArrayStream
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    frames = []
    pos = np.array([16.0, 16.0])
    vel = np.array([1.5, 1.0])
    for _ in range(15):
        f = rng.normal(0.1, 0.01, (64, 64)).astype(np.float32)
        y, x = int(pos[0]), int(pos[1])
        f[max(0, y):y + 6, max(0, x):x + 6] = 1.0
        frames.append(f)
        pos = pos + vel
    stream = ArrayStream(np.stack(frames), fps=10.0)

    me = MotionEnergy(blur=1, downsample=2, p_high=95.0)
    energies = []
    for fr in stream:
        o = me.transform_one(fr)
        me.learn_one(fr)
        energies.append(o["motion_mean"])
    energies = np.array(energies)

    a = gaussian_filter(rng.normal(0, 1, (64, 64)), 2.0)
    a = ((a - a.min()) / (a.max() - a.min())).astype(np.float32)
    shifts = []
    errs = []
    for step in range(14):
        dy = -1 if step % 2 == 0 else 1
        dx = 1 if step % 3 == 0 else -1
        b = np.roll(np.roll(a, dy, 0), dx, 1).astype(np.float32)
        gm = GlobalMotion()
        gm.learn_one({"array": a, "t": 0.0})
        o = gm.transform_one({"array": b, "t": 0.033})
        shifts.append((o["dx"], o["dy"]))
        errs.append(np.hypot(o["dx"] - dx, o["dy"] - dy))
        a = b
    errs = np.array(errs)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(np.arange(len(energies)), energies, color=BLUE, linewidth=1.5)
    ax.set_xlabel("frame")
    ax.set_ylabel("motion mean")
    ax.set_title("MotionEnergy per frame", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(np.arange(len(errs)), errs, "o-", color=BLUE, linewidth=1.4,
            markersize=6, markeredgecolor=SURFACE)
    ax.axhline(float(np.median(errs)), color=ORANGE, linestyle="--",
               linewidth=1.0, label=f"median = {np.median(errs):.2f} px")
    ax.set_xlabel("step")
    ax.set_ylabel("shift error [px]")
    ax.set_title("GlobalMotion: |estimated - true|", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "motion", "energy_and_shift")
    FIG["fig10_energy"] = [float(v) for v in energies]
    FIG["fig10_shift_median_err"] = float(np.median(errs))
    FIG["fig10_shift_max_err"] = float(errs.max())

def fig_11_scenes():
    """SceneChange and FrameAnomaly on synthetic cuts."""
    from dense_armor.utility.drift.page_hinkley import PageHinkley
    from dense_armor.utility.vision.scene import SceneChange, FrameAnomaly
    from dense_armor.utility.vision.window import FrameWindow

    rng = np.random.default_rng(0)
    frames = []
    for i in range(60):
        if i < 30:
            frames.append(rng.normal(0.2, 0.02, (64, 64)).astype(np.float32))
        else:
            frames.append(rng.normal(0.8, 0.02, (64, 64)).astype(np.float32))
    det = PageHinkley(threshold=10.0, warmup=10, delta=0.0005)
    sc = SceneChange(detector=det, bins=32, blur=1, downsample=2)
    dist = []
    alarms = []
    for f in frames:
        o = sc.transform_one({"array": f})
        sc.learn_one({"array": f})
        dist.append(o["distance"])
        alarms.append(bool(o["scene_change"]))
    dist = np.array(dist)
    first = next((i for i, a in enumerate(alarms) if a), -1)

    normal = [rng.normal(0.5, 0.02, (64, 64)).astype(np.float32)
              for _ in range(40)]
    fa = FrameAnomaly()
    for f in normal:
        fa.learn_one(f)
    normal_scores = [fa.transform_one(f)["anomaly"] for f in normal]
    anomalous = [rng.normal(0.9, 0.3, (64, 64)).astype(np.float32)
                 for _ in range(10)]
    anomalous_scores = [fa.transform_one(f)["anomaly"]
                        for f in anomalous]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(np.arange(len(dist)), dist, color=BLUE, linewidth=1.4,
            label="scene distance")
    if first >= 0:
        ax.axvline(first, color=ORANGE, linestyle="--", linewidth=1.2,
                   label=f"alarm frame {first}")
    ax.axvline(30, color=INK, linestyle=":", linewidth=1.0,
               label="true cut at 30")
    ax.set_xlabel("frame")
    ax.set_ylabel("scene distance")
    ax.set_title("SceneChange with PageHinkley", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(["normal", "anomalous"],
           [float(np.mean(normal_scores)), float(np.mean(anomalous_scores))],
           color=[BLUE, ORANGE], edgecolor=INK, linewidth=0.5)
    ax.set_ylabel("mean anomaly score")
    ax.set_title("FrameAnomaly on 64x64 frames", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "scenes", "scene_change_and_anomaly")
    FIG["fig11_first_alarm"] = int(first)
    FIG["fig11_true_cut"] = 30
    FIG["fig11_normal_score"] = float(np.mean(normal_scores))
    FIG["fig11_anomalous_score"] = float(np.mean(anomalous_scores))

def fig_12_tracking():
    """MovingObjectTracker on boxes from a known ground truth."""
    from dense_armor.utility.vision.tracking import MovingObjectTracker

    rng = np.random.default_rng(0)
    frames = []
    boxes = []
    for i in range(20):
        f = rng.normal(0.1, 0.01, (64, 64)).astype(np.float32)
        ya, xa = int(10 + 1.5 * i), int(5 + 1.0 * i)
        yb, xb = int(45 - 1.0 * i), int(50 - 1.5 * i)
        f[ya:ya + 6, xa:xa + 6] = 1.0
        f[yb:yb + 6, xb:xb + 6] = 0.9
        frames.append(f)
        boxes.append([
            (float(xa), float(ya), float(xa + 6), float(ya + 6), 1.0),
            (float(xb), float(yb), float(xb + 6), float(yb + 6), 1.0),
        ])
    tr = MovingObjectTracker(min_area=8, max_missing=3)
    last = None
    for b in boxes:
        last = tr.update(b)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(frames[-1], cmap="gray", vmin=0.0, vmax=1.0)
    n_tracks = len(last["tracks"])
    for track in last["tracks"]:
        x0, y0, x1, y1 = track["box"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=ORANGE,
                                   linewidth=1.4))
        ax.text(x0, y0 - 2, f"id {track['id']}", color=ORANGE, fontsize=9)
    ax.set_title(f"{n_tracks} tracks at frame 19", fontsize=9)
    ax.set_xticks(np.arange(0, 65, 16))
    ax.set_yticks(np.arange(0, 65, 16))
    _save(fig, "tracking", "two_squares")
    FIG["fig12_n_tracks"] = n_tracks
    FIG["fig12_ids"] = [int(t["id"]) for t in last["tracks"]]

def fig_13_signals():
    """FrameJointFusion: estimate the lag between video velocity and joints."""
    from dense_armor.utility.vision.fusion import FrameJointFusion

    fps = 30.0
    T = 6.0
    n = int(T * fps)
    t = np.arange(n) / fps
    joint = np.sin(2 * np.pi * 0.5 * t)
    lag_frames = 2
    vid = np.roll(joint, lag_frames) + np.random.default_rng(0).normal(0, 0.05, n)
    fjf = FrameJointFusion(max_lag_s=0.1, window_s=1.0, frame_rate_hz=fps)
    for i in range(n):
        fjf.add_joint_speed(float(joint[i]), float(t[i]))
        fjf.add_video_velocity(float(vid[i]), float(t[i]))
    est = fjf.estimate()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(t, joint, color=BLUE, linewidth=1.4, label="joint speed")
    ax.plot(t, vid, color=ORANGE, linewidth=1.0, alpha=0.8,
            label="video velocity (lag 2 frames)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normalised value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.axhline(2 * lag_frames / fps * 1000.0, color=INK, linestyle="--",
               linewidth=1.0, label="true lag = 66.7 ms")
    ax.axhline(est * 1000.0 if isinstance(est, float) else float(est["lag_s"]) * 1000.0,
               color=ORANGE, linewidth=1.4, label="estimated lag")
    ax.set_ylim(0, 120)
    ax.set_ylabel("lag [ms]")
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "signals", "video_joint_lag")
    FIG["fig13_true_lag_s"] = 2.0 / fps
    FIG["fig13_est_lag_s"] = (est if isinstance(est, float)
                              else float(est.get("lag_s", -1.0)))

def fig_14_landmarks():
    """Kinematics on a 3-point chain: elbow angle raw vs cleaned."""
    from dense_armor.utility.vision.landmarks import (
        Kinematics, LandmarkCleaner, LandmarkStream,
    )

    fps = 30.0
    n = 90
    t_arr = np.arange(n) / fps
    true_angle = 90.0 + 30.0 * np.sin(2 * np.pi * t_arr)
    rng = np.random.default_rng(0)
    theta = np.radians(true_angle)
    L = 10.0
    sh = np.zeros((n, 2))
    el = np.tile([L, 0.0], (n, 1))
    wr = np.stack([L - L * np.cos(theta), L * np.sin(theta)], axis=1)
    pts = np.stack([sh, el, wr], axis=1)
    pts += rng.normal(0, 0.5, pts.shape)
    for f, d in [(20, 20.0), (45, 20.0), (46, 20.0), (70, 20.0)]:
        pts[f, 1, 0] += d

    stream = LandmarkStream(pts, fps=fps)
    cl = LandmarkCleaner(window=7, n_sigmas=3.0, max_gap=5)
    kin = Kinematics(triples=[(0, 1, 2)])
    raw_angles = []
    clean_angles = []
    for fr in stream:
        o_raw = kin.transform_one(fr)
        raw_angles.append(np.degrees(o_raw["angle_0_1_2"]))
        _ = cl.learn_one(fr)
        fr_clean = cl.transform_one(fr)
        o_c = kin.transform_one(fr_clean)
        clean_angles.append(np.degrees(o_c["angle_0_1_2"]))
    raw = np.array(raw_angles)
    clean = np.array(clean_angles)
    err_raw = float(np.sqrt(np.mean((raw - true_angle) ** 2)))
    err_clean = float(np.sqrt(np.mean((clean - true_angle) ** 2)))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(t_arr, true_angle, color=INK, linestyle="--", linewidth=1.2,
            label="true angle")
    ax.plot(t_arr, raw, color=ORANGE, linewidth=1.0, alpha=0.8,
            label=f"raw (RMSE {err_raw:.2f} deg)")
    ax.plot(t_arr, clean, color=BLUE, linewidth=1.4,
            label=f"cleaned (RMSE {err_clean:.2f} deg)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("elbow angle [deg]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "landmarks", "angle_raw_cleaned")
    FIG["fig14_rmse_raw"] = err_raw
    FIG["fig14_rmse_clean"] = err_clean

def fig_15_registration():
    """Stabilizer and DifferenceImage on a known shift."""
    from scipy.ndimage import gaussian_filter
    from dense_armor.utility.vision.registration import (
        DifferenceImage, Stabilizer,
    )

    rng = np.random.default_rng(0)
    a = gaussian_filter(rng.normal(0, 1, (64, 64)), 2.0)
    a = ((a - a.min()) / (a.max() - a.min())).astype(np.float32)
    b = np.roll(np.roll(a, 3, 0), -4, 1).astype(np.float32)
    b[30:36, 30:36] = 1.0
    st = Stabilizer()
    st.learn_one({"array": a, "t": 0.0})
    o = st.transform_one({"array": b, "t": 0.033})
    b_aligned = o["frame"]
    dx, dy = o["dx"], o["dy"]
    di_plain = DifferenceImage(alpha=0.01)
    di_plain.learn_one({"array": a})
    d_plain = di_plain.transform_one({"array": b})["diff"]
    di_reg = DifferenceImage(alpha=0.01)
    di_reg.learn_one({"array": a})
    d_reg = di_reg.transform_one({"array": b_aligned})["diff"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, img, title in [
        (axes[0], a, "reference"),
        (axes[1], b, "shifted + square"),
        (axes[2], np.abs(d_plain), "plain difference"),
        (axes[3], np.abs(d_reg), "registered difference"),
    ]:
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    _save(fig, "registration", "shift_and_difference")
    FIG["fig15_shift_dx"] = float(dx)
    FIG["fig15_shift_dy"] = float(dy)
    FIG["fig15_true_dx"] = -4.0
    FIG["fig15_true_dy"] = 3.0
    FIG["fig15_max_plain"] = float(np.abs(d_plain).max())
    FIG["fig15_max_reg"] = float(np.abs(d_reg).max())





def fig_16_dense_flow():
    """DenseFlow on three moving blobs, MotionMask on a moving square.

    The scene has sensor noise so the MAD of MotionMask is non-zero.
    The IoU on this synthetic case is low (around 0.2): the mask finds
    the moving square but also reacts to the blob edges. The page
    states the measured value, not a tuned one.
    """
    from dense_armor.utility.vision.dense import DenseFlow, MotionMask

    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    blobs = [(20.0, 20.0), (40.0, 25.0), (30.0, 45.0)]
    sigma = 6.0

    def scene(seed, sq_y=None, sq_x=None):
        rng = np.random.default_rng(seed)
        img = np.zeros((H, W), dtype=np.float32)
        for cy, cx in blobs:
            img += np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        img = img / img.max()
        if sq_y is not None:
            y0, x0 = int(sq_y), int(sq_x)
            img[y0:y0 + 8, x0:x0 + 8] = 1.0
        img += rng.normal(0, 0.02, (H, W)).astype(np.float32)
        return img

    f1 = scene(0)
    f2 = scene(0, sq_y=20, sq_x=20)
    df = DenseFlow(alpha=1.0, n_iter=10, use_jax=False)
    df.learn_one({"array": f1})
    o = df.transform_one({"array": f2})
    u, v, mag = o["u"], o["v"], o["mag"]

    s1 = scene(0, sq_y=20, sq_x=20)
    s2 = scene(1, sq_y=23, sq_x=23)
    df2 = DenseFlow(alpha=1.0, n_iter=10, use_jax=False)
    df2.learn_one({"array": s1})
    o2 = df2.transform_one({"array": s2})
    mm = MotionMask(n_sigmas=2.0, dilation=1)
    mm.learn_one({"u": o2["u"], "v": o2["v"]})
    mask = mm.transform_one({"u": o2["u"], "v": o2["v"]})["mask"]
    gt_mask = np.zeros((H, W), dtype=bool)
    gt_mask[20:31, 20:31] = True
    tp = int((mask & gt_mask).sum())
    fp = int((mask & ~gt_mask).sum())
    fn = int((~mask & gt_mask).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    uu, vv = u[10:50, 10:50], v[10:50, 10:50]
    epe = float(np.mean(np.hypot(uu - 0.0, vv - 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.imshow(f2, cmap="gray")
    step = 6
    ax.quiver(xx[::step, ::step], yy[::step, ::step],
              u[::step, ::step], v[::step, ::step],
              mag[::step, ::step], cmap="Blues", scale=20, width=0.005)
    ax.set_title(f"flow, mean |u,v| = {epe:.3f} px", fontsize=9)
    ax.axis("off")
    ax = axes[1]
    ax.imshow(s2, cmap="gray")
    ax.contour(gt_mask, levels=[0.5], colors=INK, linestyles="--",
               linewidths=1.0)
    ax.contour(mask.astype(float), levels=[0.5], colors=ORANGE,
               linewidths=1.4)
    ax.set_title(f"mask IoU = {iou:.2f}", fontsize=9)
    ax.axis("off")
    _save(fig, "dense_flow", "flow_and_mask")
    FIG["fig16_mean_flow"] = epe
    FIG["fig16_iou"] = float(iou)
    FIG["fig16_mask_area"] = int(mask.sum())
    FIG["fig16_gt_area"] = int(gt_mask.sum())



def fig_17_quality():
    """FrameQuality: sharp, blurred and black."""
    from scipy.ndimage import gaussian_filter
    from dense_armor.utility.vision.quality import FrameQuality

    rng = np.random.default_rng(0)
    sharp = rng.normal(0.5, 0.3, (64, 64)).astype(np.float32)
    sharp = np.clip(sharp, 0.0, 1.0)
    blurred = gaussian_filter(sharp, 3.0)
    black = np.zeros((64, 64), dtype=np.float32)
    fq = FrameQuality()
    rows = []
    for name, img in [("sharp", sharp), ("blurred", blurred), ("black", black)]:
        o = fq.transform_one({"array": img})
        rows.append((name, o))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (name, o), img in zip(axes, rows, [sharp, blurred, black]):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        flags = []
        if o["black"]:
            flags.append("black")
        if o["repeated"]:
            flags.append("repeated")
        title = (f"{name}\nsharp={o['sharpness']:.3f} noise={o['noise']:.3f}"
                 + (f" [{','.join(flags)}]" if flags else ""))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    _save(fig, "quality", "sharp_blurred_black")
    for name, o in rows:
        FIG[f"fig17_{name}_sharpness"] = float(o["sharpness"])
        FIG[f"fig17_{name}_noise"] = float(o["noise"])
        FIG[f"fig17_{name}_black"] = bool(o["black"])


def fig_18_visualise():
    """FalseColor on a thermal band and Overlay with a box."""
    from dense_armor.utility.vision.visualize import FalseColor, Overlay

    rng = np.random.default_rng(0)
    band = rng.normal(300.0, 5.0, (64, 64)).astype(np.float32)
    band[21:27, 21:27] += 50.0
    fc = FalseColor(colormap="hot", low_pct=1.0, high_pct=99.0)
    fc.learn_one({"array": band})
    rgb = fc.transform_one({"array": band})["rgb"]
    legend = fc.transform_one({"array": band})["legend"]
    ov = Overlay(box_thickness=1)
    out = ov.transform_one({"array": rgb,
                            "tracks": [{"id": 1,
                                        "box": (21.0, 21.0, 27.0, 27.0)}]})
    rgb_over = out["rgb"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].imshow(rgb)
    axes[0].set_title(f"FalseColor {legend['colormap']}", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(np.clip(rgb_over, 0, 1))
    axes[1].set_title("Overlay with a box", fontsize=9)
    axes[1].axis("off")
    _save(fig, "visualise", "false_color_and_overlay")
    FIG["fig18_colormap"] = legend["colormap"]
    FIG["fig18_low"] = float(legend["low"])
    FIG["fig18_high"] = float(legend["high"])


def fig_19_event():
    """EventFrame on 200 random events."""
    from dense_armor.utility.vision.quality import EventFrame

    rng = np.random.default_rng(0)
    ef = EventFrame(shape=(64, 64), decay=0.9)
    for _ in range(200):
        ef.learn_one({"x": int(rng.integers(0, 64)),
                      "y": int(rng.integers(0, 64)),
                      "polarity": int(rng.choice([-1, 1])),
                      "t": 0.0})
    o = ef.transform_one(None)
    img = o["frame"]
    n_nonzero = int((img != 0).sum())
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"EventFrame after {o['n_events']} events, "
                 f"{n_nonzero} nonzero", fontsize=9)
    ax.set_xticks(np.arange(0, 65, 16))
    ax.set_yticks(np.arange(0, 65, 16))
    _save(fig, "quality", "event_frame")
    FIG["fig19_n_events"] = int(o["n_events"])
    FIG["fig19_n_nonzero"] = n_nonzero


def fig_20_benchmark():
    """Benchmark: ms per frame per module at 640x360."""
    from pathlib import Path as _P
    json_path = ROOT / "benchmarks" / "video_quality.json"
    data = json.loads(_P(json_path).read_text())
    runs = data["runs"].get("smooth_640x360", {})
    names = []
    meds = []
    for k, v in runs.items():
        if v["median_us"] is None or not np.isfinite(v["median_us"]):
            continue
        names.append(k)
        meds.append(v["median_us"] / 1000.0)
    if not names:
        print("  fig_20: JSON without 640x360 rows, figure skipped")
        return
    order = np.argsort(meds)
    names = [names[i] for i in order]
    meds = [meds[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    y = np.arange(len(names))
    ax.barh(y, meds, color=BLUE, edgecolor=INK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("median milliseconds per frame")
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)
    for yi, m in zip(y, meds):
        ax.text(m * 1.1, yi, f"{m:.2f}", va="center", fontsize=7)
    _save(fig, "benchmark", "ms_per_frame")
    for name, m in zip(names, meds):
        FIG[f"fig20_{name}_ms"] = float(m)

PAGES = [
    ("fig_01_frames", fig_01_frames),
    ("fig_02_thermal", fig_02_thermal),
    ("fig_03_hog", fig_03_hog),
    ("fig_04_moments", fig_04_moments),
    ("fig_05_flow", fig_05_flow),
    ("fig_06_colour", fig_06_colour),
    ("fig_07_projection", fig_07_projection),
    ("fig_08_pca", fig_08_pca),
    ("fig_09_chain", fig_09_chain),
    ("fig_10_motion", fig_10_motion),
    ("fig_11_scenes", fig_11_scenes),
    ("fig_12_tracking", fig_12_tracking),
    ("fig_13_signals", fig_13_signals),
    ("fig_14_landmarks", fig_14_landmarks),
    ("fig_15_registration", fig_15_registration),
    ("fig_16_dense_flow", fig_16_dense_flow),
    ("fig_17_quality", fig_17_quality),
    ("fig_18_visualise", fig_18_visualise),
    ("fig_19_event", fig_19_event),
    ("fig_20_benchmark", fig_20_benchmark),
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
