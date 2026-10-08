"""RAD benchmark: native PatchCore on real robot images.

Dataset: RAD, Chang et al. (2024), "RAD: A Realistic Multi-View
Benchmark for Pose-Agnostic Anomaly Detection", arXiv:2410.00713.
Dataset copyright (c) 2024-2026 RAD Authors, MIT License. Images are
not committed to this repository; this script expects the extracted
folder under ``RAD_ROOT``.

Compares four runs on the same 13-category protocol:

- reservoir mode, image size 160 x 120;
- reservoir mode, image size 320 x 240;
- coreset mode, image size 160 x 120;
- coreset mode, image size 320 x 240.

Baseline of ``vision_rad.py`` (global frame features, incremental PCA,
robust Mahalanobis): mean image-level AUROC 0.497. Paper number of
PatchCore (WideResNet50 backbone): 0.833.

The images are separate views of the same object, not a video: no
optical flow, no previous frame. Same parameters for every category,
fixed before looking at any test image.

Run from the repository root:

    python benchmarks/vision_rad_patches.py
"""

import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dense_armor.utility.vision import (
    ImageFolderStream,
    PatchFeatures,
    PatchMemory,
)

RAD_ROOT = Path("/content/rad/Anomaly_refine_msk")

CATEGORY_MAP = {
    "binderclip": ["binderclip", "binderclip2"],
    "bowl": ["bowl_upright"],
    "box": ["box"],
    "can": ["can"],
    "charger": ["charger"],
    "cup1": ["cup1_upright"],
    "cup2": ["cup2_upright", "cup2_upright2", "cup2_upright3"],
    "gluebottle": ["gluebottle", "gluebottle2"],
    "phonecase": ["phonecase", "phonecase2"],
    "rubberduck": ["rubberduck"],
    "spoon": ["spoon_upright"],
    "spraybottle": ["spraybottle2"],
    "tennisball": ["tennisball"],
}

PATCH = 16
STRIDE = 8
N_BINS = 9
NEIGHBOURHOOD = 3
MAX_SIZE = 20000
LATENCY_SAMPLE = 95

BASELINE = 0.497
PAPER = 0.833

RUNS: list[tuple[str, tuple[int, int], str]] = [
    ("reservoir", (160, 120), "reservoir"),
    ("reservoir", (320, 240), "reservoir"),
    ("coreset", (160, 120), "coreset"),
    ("coreset", (320, 240), "coreset"),
]

if os.environ.get("VISION_RAD_RUNS"):
    RUNS = []
    for token in os.environ["VISION_RAD_RUNS"].split(","):
        mode, size = token.split("@")
        w, h = size.split("x")
        RUNS.append((mode, (int(w), int(h)), mode))


def _new_components(mode):
    pf = PatchFeatures(
        patch=PATCH,
        stride=STRIDE,
        n_bins=N_BINS,
        neighbourhood=NEIGHBOURHOOD,
    )
    mem = PatchMemory(max_size=MAX_SIZE, seed=0, mode=mode)
    return pf, mem


def _iter_train(subdirs, size):
    for sd in subdirs:
        p = RAD_ROOT / sd / "train" / "good"
        if not p.exists():
            continue
        yield from ImageFolderStream(p, fps=None, size=size, gray=True)


def _iter_test(subdirs, size):
    for sd in subdirs:
        test_dir = RAD_ROOT / sd / "test"
        if not test_dir.exists():
            continue
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = 0 if sub.name == "good" else 1
            for frame in ImageFolderStream(sub, fps=None, size=size, gray=True):
                yield label, frame


def run_category(subdirs, size, mode):
    pf, mem = _new_components(mode)
    n_train = 0
    t0 = time.perf_counter()
    for frame in _iter_train(subdirs, size):
        mem.learn_one(pf.transform_one(frame))
        n_train += 1
    if mode == "coreset":
        mem.finalize()
    t_train = time.perf_counter() - t0
    labels = []
    scores = []
    for label, frame in _iter_test(subdirs, size):
        labels.append(label)
        scores.append(mem.score_one(pf.transform_one(frame)))
    return (
        np.asarray(labels),
        np.asarray(scores, dtype=float),
        n_train,
        mem.bank_size,
        t_train,
    )


def measure_latency(subdirs, size, mode):
    pf, mem = _new_components(mode)
    for frame in _iter_train(subdirs, size):
        mem.learn_one(pf.transform_one(frame))
    if mode == "coreset":
        mem.finalize()
    test_good = RAD_ROOT / subdirs[0] / "test" / "good"
    if not test_good.exists():
        return None
    samples = []
    for i, frame in enumerate(
        ImageFolderStream(test_good, fps=None, size=size, gray=True)
    ):
        if i >= LATENCY_SAMPLE:
            break
        t0 = time.perf_counter_ns()
        mem.score_one(pf.transform_one(frame))
        samples.append(time.perf_counter_ns() - t0)
    if not samples:
        return None
    a = np.asarray(samples, dtype=float) / 1e6
    return float(np.percentile(a, 50)), float(np.percentile(a, 99))


def run_one(mode, size):
    print("=" * 74)
    print(f"mode = {mode}   image size (w, h) = {size}")
    print("=" * 74)
    header = (
        f"{'category':<14} {'n_train':>7} {'n_test':>7} "
        f"{'pos':>5} {'bank':>6} {'AUROC':>7} {'train_s':>8}"
    )
    print(header)
    print("-" * len(header))
    aurocs = []
    for cat, subs in CATEGORY_MAP.items():
        print(f"[{mode} {size[0]}x{size[1]}] {cat} ...", flush=True)
        y, s, n_tr, n_bank, t_tr = run_category(subs, size, mode)
        n_te = len(y)
        n_pos = int(y.sum())
        if len(np.unique(y)) < 2:
            print(
                f"{cat:<14} {n_tr:>7} {n_te:>7} {n_pos:>5} "
                f"{n_bank:>6} {'n/a':>7} {t_tr:>8.2f}"
            )
            continue
        a = float(roc_auc_score(y, s))
        aurocs.append(a)
        print(
            f"{cat:<14} {n_tr:>7} {n_te:>7} {n_pos:>5} "
            f"{n_bank:>6} {a:>7.3f} {t_tr:>8.2f}"
        )
    print("-" * len(header))
    mean = float(np.mean(aurocs)) if aurocs else float("nan")
    print(f"{'mean':<14} {'':>7} {'':>7} {'':>5} {'':>6} {mean:>7.3f}")
    print()
    lat = measure_latency(CATEGORY_MAP["box"], size, mode)
    if lat is not None:
        p50, p99 = lat
        print(
            f"latency per image at {size[0]}x{size[1]} ({mode}, "
            f"score only, n={LATENCY_SAMPLE}):"
        )
        print(f"  p50 = {p50:.3f} ms   p99 = {p99:.3f} ms")
    print()
    return mean, lat


def main():
    print(f"RAD root: {RAD_ROOT}")
    print(
        f"pipeline: PatchFeatures(patch={PATCH}, stride={STRIDE}, "
        f"n_bins={N_BINS}, neighbourhood={NEIGHBOURHOOD}) | "
        f"PatchMemory(max_size={MAX_SIZE})"
    )
    print()
    summary = []
    for mode, size, _ in RUNS:
        mean, lat = run_one(mode, size)
        summary.append((mode, size, mean, lat))
    print("=" * 74)
    print("summary")
    print("=" * 74)
    print(f"{'mode':<12} {'size':<12} {'mean AUROC':>12} {'p50 ms':>10} {'p99 ms':>10}")
    print("-" * 58)
    for mode, size, mean, lat in summary:
        if lat is None:
            p50 = p99 = float("nan")
        else:
            p50, p99 = lat
        print(f"{mode:<12} {size!s:<12} {mean:>12.3f} {p50:>10.1f} {p99:>10.1f}")
    print()
    print(
        f"baseline vision_rad.py (global features + PCA + Mahalanobis): {BASELINE:.3f}"
    )
    print(f"paper PatchCore (WideResNet50 backbone): {PAPER:.3f}")


if __name__ == "__main__":
    main()
