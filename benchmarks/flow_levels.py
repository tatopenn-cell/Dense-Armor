"""Optical flow levels and intensity scale: the numbers behind the guide.

Table 1 (``levels``): for shifts ``(1, 1)``, ``(3, 2)``, ``(6, 4)`` and
``(10, 6)`` pixels, on a smooth texture of 64 and of 128 pixels, the
median flow of :class:`FrameFeatures` against the true shift, for
``flow_levels`` 1 to 4, median and interquartile range over the seeds.

Table 2 (``intensity``): the endpoint error of :class:`DenseFlow` against
the true shift, at intensity scales 1, 10 and 255, with and without
``normalize_intensity``, on the same smooth texture.

The texture is a sum of three sinusoids with the same frequency in both
axes, phases drawn from the seed. The frame ``b`` is a crop of the same
texture shifted by ``(dx, dy)``, so ``b[y, x] = a[y + dy, x + dx]`` (no
wrap-around at the border): the apparent motion from ``a`` to ``b`` is
``(dx, dy)``.

Command, from the repository root:

    python benchmarks/flow_levels.py

    python benchmarks/flow_levels.py --smoke
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dense_armor.utility.vision.dense import DenseFlow
from dense_armor.utility.vision.features import FrameFeatures

SHIFTS = [(1, 1), (3, 2), (6, 4), (10, 6)]
SIZES = [64, 128]
LEVELS = [1, 2, 3, 4]
SCALES = [1, 10, 255]
MAX_DY = max(dy for _, dy in SHIFTS)
MAX_DX = max(dx for dx, _ in SHIFTS)


def _texture(size, seed):
    """Smooth 2-D texture, sum of three sinusoids at different periods."""
    rng = np.random.default_rng(seed)
    h = size + MAX_DY
    w = size + MAX_DX
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((h, w), dtype=np.float64)
    for period in (8, 16, 32):
        fy = float(rng.uniform(0.0, 2.0 * np.pi))
        fx = float(rng.uniform(0.0, 2.0 * np.pi))
        out += np.sin(2.0 * np.pi * y / period + fy)
        out += np.sin(2.0 * np.pi * x / period + fx)
    out -= out.min()
    out /= out.max()
    return out


def _crop_pair(texture, size, dx, dy):
    """Return ``(a, b)`` such that ``b[y, x] = a[y + dy, x + dx]``."""
    a = texture[MAX_DY - dy:MAX_DY - dy + size,
                MAX_DX - dx:MAX_DX - dx + size]
    b = texture[MAX_DY:MAX_DY + size, MAX_DX:MAX_DX + size]
    return a.astype(np.float32), b.astype(np.float32)


def _table_levels(seeds):
    """Median and IQR of the median flow for each (shift, size, levels)."""
    out = {}
    for size in SIZES:
        for dx, dy in SHIFTS:
            for levels in LEVELS:
                us, vs = [], []
                for seed in range(seeds):
                    tex = _texture(size, seed)
                    a, b = _crop_pair(tex, size, dx, dy)
                    ff = FrameFeatures(flow_cells=(1, 1), flow_levels=levels)
                    ff.learn_one(a)
                    f = ff.transform_one(b)
                    us.append(float(f["flow_u_0_0"]))
                    vs.append(float(f["flow_v_0_0"]))
                key = f"size={size} shift=({dx},{dy}) levels={levels}"
                out[key] = {
                    "u_median": float(np.median(us)),
                    "u_iqr": float(np.percentile(us, 75) - np.percentile(us, 25)),
                    "v_median": float(np.median(vs)),
                    "v_iqr": float(np.percentile(vs, 75) - np.percentile(vs, 25)),
                    "true_u": float(dx),
                    "true_v": float(dy),
                }
    return out


def _table_intensity(seeds):
    """Endpoint error of DenseFlow against the true shift, per intensity."""
    out = {}
    dx, dy = 3, 2
    size = 64
    for scale in SCALES:
        for normalize in (False, True):
            for seed in range(seeds):
                tex = _texture(size, seed) * float(scale)
                a, b = _crop_pair(tex, size, dx, dy)
                f = DenseFlow(alpha=1.0, n_iter=30,
                              normalize_intensity=normalize)
                f.learn_one(a)
                o = f.transform_one({"array": b})
                u_med = float(np.median(o["u"]))
                v_med = float(np.median(o["v"]))
                err = float(np.hypot(u_med - dx, v_med - dy))
                key = f"scale={scale} normalize={normalize}"
                out.setdefault(key, []).append(
                    {"u": u_med, "v": v_med, "error": err}
                )
    for k, v in list(out.items()):
        errs = np.array([r["error"] for r in v])
        us = np.array([r["u"] for r in v])
        vs = np.array([r["v"] for r in v])
        out[k] = {
            "u_median": float(np.median(us)),
            "v_median": float(np.median(vs)),
            "error_median": float(np.median(errs)),
            "error_iqr": float(np.percentile(errs, 75) - np.percentile(errs, 25)),
            "true_u": float(dx),
            "true_v": float(dy),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="benchmarks/flow_levels.json")
    a = ap.parse_args()
    seeds = 1 if a.smoke else a.seeds

    print(f"seeds = {seeds}, smoke = {a.smoke}")
    print()
    print("=== Table 1: median flow of FrameFeatures (u, v) vs true shift ===")
    print(f"{'size':>5} {'shift':>9} {'levels':>7} "
          f"{'u_med':>8} {'v_med':>8} {'true':>9}")
    t1 = _table_levels(seeds)
    for k, v in t1.items():
        print(f"  {k:40s}  u={v['u_median']:.2f} v={v['v_median']:.2f} "
              f"true=({v['true_u']:.0f},{v['true_v']:.0f})")

    print()
    print("=== Table 2: DenseFlow endpoint error at intensity scales ===")
    print(f"{'scale':>6} {'normalize':>10} {'u_med':>8} {'v_med':>8} "
          f"{'err_med':>8}")
    t2 = _table_intensity(seeds)
    for k, v in t2.items():
        print(f"  {k:30s}  u={v['u_median']:.3f} v={v['v_median']:.3f} "
              f"err={v['error_median']:.3f}  true=({v['true_u']:.0f},{v['true_v']:.0f})")

    out = Path(a.out)
    out.write_text(json.dumps(
        {"seeds": seeds, "smoke": a.smoke,
         "table_levels": t1, "table_intensity": t2}, indent=2))
    print()
    print(f"JSON: {out}  ({out.stat().st_size} B)")


if __name__ == "__main__":
    main()
