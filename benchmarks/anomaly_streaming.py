"""Streaming anomaly detectors on the library's own streams.

Three evaluation protocols, all reported:

- ``learn-on-healthy``: the detector learns only from the healthy part
  of the stream and is scored on the whole stream.
- ``learn-on-all``: the detector sees the whole stream one sample at a
  time, scoring each before learning from it. This is the protocol the
  papers of HalfSpaceTrees, OnlineIsolationForest and LODA describe.
- ``learn-on-all-online-ranges``: same as ``learn-on-all`` but with the
  feature ranges learned online from the first ``range_init`` samples
  inside the detector, instead of being fixed in advance. It is only
  available for the detectors that accept ``feature_ranges=None``.

The ``learn-on-healthy`` and ``learn-on-all`` runs pass explicit
``feature_ranges`` (or nothing, for detectors that do not need them),
computed from the healthy part of the stream with a 10 % margin. That
computation uses the labels; it is equivalent to a calibration phase
in deployment, and is kept for comparability with the paper
protocols. The ``learn-on-all-online-ranges`` run is the fully
label-free variant.

Streams:

- SyntheticArm, payload fault: joints 0-3, input is the residual
  ``tau_meas - ds.nominal_torque``. ``y`` is 0 before the fault and 1
  after. 400 samples at 50 Hz with a 2 s period, fault at 2 s. Joint
  0 has no payload torque and is included as an uninformative control
  channel; the discriminative signal sits on joints 1-5.
- DriftStream, one sudden drift: the samples of the first concept are
  the healthy part, the samples of the second concept the anomalies.
  2000 samples, spacing 999 (sharp step at 0.5 of the stream).

Run from the repository root:

    python benchmarks/anomaly_streaming.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.anomaly.mahalanobis import (
    OnlineRobustMahalanobis,
)
from dense_armor.utility.anomaly.ocsvm import OneClassSGD
from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest
from dense_armor.utility.datasets import DriftStream, SyntheticArm

URDF = Path("test/fixtures/urdf/panda.urdf")

DETECTOR_NAMES = [
    "HalfSpaceTrees",
    "OnlineIsolationForest",
    "LODA",
    "OneClassSGD",
    "OnlineRobustMahalanobis",
]


def arm_stream():
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=4,
        fault_at_s=2.0,
        fault="payload",
        payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 5e-2),
        seed=0,
    )
    X = []
    y = []
    for sig, label in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        X.append(x)
        y.append(int(label))
    return X, np.asarray(y)


def drift_stream():
    ds = DriftStream(
        n_samples=2000,
        n_features=4,
        n_drifts=1,
        kind="sudden",
        spacing=999.0,
        seed=0,
    )
    X = []
    for sig, _ in ds.stream():
        X.append({k: float(sig[k]) for k in sig.names})
    concepts = ds.concepts_
    y = (concepts == 1).astype(int)
    return X, y


def healthy_ranges(X, y, margin=0.1):
    keys = list(X[0].keys())
    healthy = [x for x, lbl in zip(X, y) if lbl == 0]
    ranges = []
    for k in keys:
        vals = np.array([float(x[k]) for x in healthy])
        lo = float(vals.min())
        hi = float(vals.max())
        span = max(hi - lo, 1e-6)
        ranges.append((lo - margin * span, hi + margin * span))
    return ranges


def evaluate(detector, X, y, learn_on_all):
    n = len(X)
    scores = np.empty(n, dtype=float)
    healthy = y == 0
    t_learn = 0.0
    t_score = 0.0
    n_learned = 0
    for i, x in enumerate(X):
        t0 = time.perf_counter()
        scores[i] = detector.score_one(x)
        t_score += time.perf_counter() - t0
        if learn_on_all or healthy[i]:
            t0 = time.perf_counter()
            detector.learn_one(x)
            t_learn += time.perf_counter() - t0
            n_learned += 1
    n_learned = max(n_learned, 1)
    ms = (t_score + t_learn) / n_learned * 1e3
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    thr = float(getattr(detector, "threshold", 0.0))
    fa = float(np.mean(scores[healthy] > thr)) if healthy.any() else float("nan")
    return auc, fa, ms, float(scores.min()), float(scores.max())


def build_one(name, keys, ranges, online_ranges):
    fr = None if online_ranges else ranges
    if name == "HalfSpaceTrees":
        return HalfSpaceTrees(
            n_trees=25,
            depth=8,
            window=200,
            feature_ranges=fr,
            range_init=50,
            feature_keys=keys,
            threshold=4.0,
            seed=0,
        )
    if name == "OnlineIsolationForest":
        return OnlineIsolationForest(
            n_trees=16,
            window=200,
            max_leaf_samples=8,
            feature_keys=keys,
            threshold=0.6,
            seed=0,
        )
    if name == "LODA":
        return LODA(
            n_projections=100,
            n_bins=10,
            window=200,
            feature_ranges=fr,
            range_init=50,
            feature_keys=keys,
            threshold=4.0,
            seed=0,
        )
    if name == "OneClassSGD":
        return OneClassSGD(
            n_features_rff=64,
            nu=0.05,
            step=0.1,
            n_init=50,
            feature_keys=keys,
            threshold=0.0,
            seed=0,
        )
    if name == "OnlineRobustMahalanobis":
        return OnlineRobustMahalanobis(
            n_init=50,
            feature_keys=keys,
            threshold=7.0,
        )
    raise ValueError(f"unknown detector {name!r}")


def run_one(name, X, y):
    keys = list(X[0].keys())
    ranges = healthy_ranges(X, y)
    print("=" * 100)
    print(
        f"stream: {name}  n={len(X)}  features={len(keys)}  positives={int(np.sum(y))}"
    )
    print(
        f"healthy ranges (calibration, uses labels): "
        f"{[(round(lo, 3), round(hi, 3)) for lo, hi in ranges]}"
    )
    print("=" * 100)
    header = (
        f"{'detector':<26} {'protocol':<32} {'ROC-AUC':>8} {'FA@thr':>7} "
        f"{'ms/sample':>10} {'s_min':>8} {'s_max':>10}"
    )
    print(header)
    print("-" * len(header))
    for det_name in DETECTOR_NAMES:
        protocols = [
            ("learn-on-healthy", False, False),
            ("learn-on-all", True, False),
            ("learn-on-all-online-ranges", True, True),
        ]
        for protocol, learn_on_all, online_ranges in protocols:
            if online_ranges and det_name == "OnlineIsolationForest":
                continue
            det = build_one(det_name, keys, ranges, online_ranges=online_ranges)
            auc, fa, ms, smin, smax = evaluate(det, X, y, learn_on_all=learn_on_all)
            print(
                f"{det_name:<26} {protocol:<32} {auc:>8.3f} {fa:>7.3f} "
                f"{ms:>10.3f} {smin:>8.2f} {smax:>10.2f}"
            )
    print()


def main():
    print("streaming anomaly detectors on the library's own streams")
    print("three protocols: learn-on-healthy, learn-on-all, learn-on-all-online-ranges")
    print()
    X_arm, y_arm = arm_stream()
    run_one("SyntheticArm payload fault", X_arm, y_arm)
    X_dr, y_dr = drift_stream()
    run_one("DriftStream sudden drift", X_dr, y_dr)


if __name__ == "__main__":
    main()
