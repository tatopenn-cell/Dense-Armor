"""RAD benchmark: native online vision on real robot images.

Dataset: RAD, Chang et al. (2024), "RAD: A Realistic Multi-View
Benchmark for Pose-Agnostic Anomaly Detection", arXiv:2410.00713.
Dataset copyright (c) 2024-2026 RAD Authors, MIT License. Images are
not committed to this repository; this script expects the extracted
folder under ``RAD_ROOT``.

The images are separate views of the same object, not a video. The
optical-flow features of ``FrameFeatures`` compare each frame to the
previous one; on unrelated views that comparison is noise, not motion.
For this reason ``ff.learn_one`` is not called here: no previous frame is
stored, the flow block is exactly zero and contributes nothing. The HOG,
the moments and the centroid do the work.

For each of the 13 semantic categories the pipeline
``FrameFeatures(flow_cells=(1, 1)) | IncrementalPCA(k=8) |
OnlineRobustMahalanobis`` is learned on the normal training views, then
scored on every test image without learning. Same parameters for every
category, fixed before looking at any test image. Image-level AUROC is
computed per category with ``sklearn.metrics.roc_auc_score`` (defect
counts as positive) and averaged over the 13 categories.

Reference numbers from the paper (image-level AUROC): PatchCore 0.833,
EfficientAD 0.803 (Chang et al. 2024, p. 10). Our numbers will almost
certainly be lower: the library uses native features, no pretrained
network.

Run from the repository root:

    python benchmarks/vision_rad.py
"""

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.vision import (
    FrameFeatures,
    ImageFolderStream,
    IncrementalPCA,
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

SIZE = (160, 120)
GREY = True
K = 8
FLOW_CELLS = (1, 1)
LATENCY_SAMPLE = 95

PAPER = (
    ("PatchCore", 0.833),
    ("EfficientAD", 0.803),
)


def _new_components():
    ff = FrameFeatures(flow_cells=FLOW_CELLS)
    pca = IncrementalPCA(k=K)
    det = OnlineRobustMahalanobis(feature_keys=[f"pc{j}" for j in range(K)])
    return ff, pca, det


def _learn(ff, pca, det, frame):
    h = ff.transform_one(frame)
    pca.learn_one(h)
    h2 = pca.transform_one(h)
    det.learn_one(h2)


def _score(ff, pca, det, frame):
    h = ff.transform_one(frame)
    h = pca.transform_one(h)
    return det.score_one(h)


def _iter_train(subdirs):
    for sd in subdirs:
        p = RAD_ROOT / sd / "train" / "good"
        if not p.exists():
            continue
        yield from ImageFolderStream(p, fps=None, size=SIZE, gray=GREY)


def _iter_test(subdirs):
    for sd in subdirs:
        test_dir = RAD_ROOT / sd / "test"
        if not test_dir.exists():
            continue
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = 0 if sub.name == "good" else 1
            for frame in ImageFolderStream(sub, fps=None, size=SIZE, gray=GREY):
                yield label, frame


def run_category(subdirs):
    ff, pca, det = _new_components()
    n_train = 0
    for frame in _iter_train(subdirs):
        _learn(ff, pca, det, frame)
        n_train += 1
    labels = []
    scores = []
    for label, frame in _iter_test(subdirs):
        labels.append(label)
        scores.append(_score(ff, pca, det, frame))
    return np.asarray(labels), np.asarray(scores, dtype=float), n_train


def measure_latency(subdirs):
    ff, pca, det = _new_components()
    for frame in _iter_train(subdirs):
        _learn(ff, pca, det, frame)
    test_good = RAD_ROOT / subdirs[0] / "test" / "good"
    if not test_good.exists():
        return None
    samples = []
    for i, frame in enumerate(
        ImageFolderStream(test_good, fps=None, size=SIZE, gray=GREY)
    ):
        if i >= LATENCY_SAMPLE:
            break
        t0 = time.perf_counter_ns()
        _score(ff, pca, det, frame)
        samples.append(time.perf_counter_ns() - t0)
    if not samples:
        return None
    a = np.asarray(samples, dtype=float) / 1e6
    return float(np.percentile(a, 50)), float(np.percentile(a, 99))



def check_flow_zero(subdirs):
    """Mean absolute value of ``flow_u_0_0`` over the test images (must be 0.0)."""
    ff, pca, det = _new_components()
    for frame in _iter_train(subdirs):
        _learn(ff, pca, det, frame)
    vals = [abs(ff.transform_one(frame)["flow_u_0_0"]) for _, frame in _iter_test(subdirs)]
    return float(np.mean(vals)) if vals else float("nan")

def main():
    print(f"RAD root: {RAD_ROOT}")
    print(
        f"pipeline: FrameFeatures(flow_cells={FLOW_CELLS}) "
        f"| IncrementalPCA(k={K}) | OnlineRobustMahalanobis"
    )
    print(f"image size (w, h) = {SIZE}, grey = {GREY}")
    print(
        "note: RAD images are separate views; ff.learn_one is not called, "
        "so no previous frame is stored and the flow features are exactly zero."
    )
    print()
    header = f"{'category':<14} {'n_train':>7} {'n_test':>7} {'pos':>5} {'AUROC':>7}"
    print(header)
    print("-" * len(header))

    aurocs = []
    per_cat = {}
    for cat, subs in CATEGORY_MAP.items():
        y, s, n_tr = run_category(subs)
        n_te = len(y)
        n_pos = int(y.sum())
        if len(np.unique(y)) < 2:
            print(f"{cat:<14} {n_tr:>7} {n_te:>7} {n_pos:>5} {'n/a':>7}")
            per_cat[cat] = None
            continue
        a = float(roc_auc_score(y, s))
        aurocs.append(a)
        per_cat[cat] = a
        print(f"{cat:<14} {n_tr:>7} {n_te:>7} {n_pos:>5} {a:>7.3f}")

    print("-" * len(header))
    if aurocs:
        print(f"{'mean':<14} {'':>7} {'':>7} {'':>5} {float(np.mean(aurocs)):>7.3f}")
    print()

    print("Reference (Chang et al. 2024, p. 10, image-level AUROC):")
    for name, val in PAPER:
        print(f"  {name:<14} {val:.3f}")
    print()

    flow_mean = check_flow_zero(CATEGORY_MAP["box"])
    print(f"flow check on box test images: mean |flow_u_0_0| = {flow_mean:.4f}")
    print()
    lat = measure_latency(CATEGORY_MAP["box"])
    if lat is not None:
        p50, p99 = lat
        print(
            f"latency per image at {SIZE[0]}x{SIZE[1]} "
            f"(score only, n={LATENCY_SAMPLE}):"
        )
        print(f"  p50 = {p50:.3f} ms   p99 = {p99:.3f} ms")


if __name__ == "__main__":
    main()
