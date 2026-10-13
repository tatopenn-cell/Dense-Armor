"""Streaming anomaly detectors on the library's own streams.

Two evaluation protocols, both reported:

- ``learn-on-healthy``: the detector learns only from the healthy part
  of the stream and is scored on the whole stream. This is the
  protocol that matches a deployment after a calibration phase.
- ``learn-on-all``: the detector sees the whole stream one sample at a
  time, scoring each before learning from it. This is the protocol the
  papers of HalfSpaceTrees, OnlineIsolationForest and LODA describe.

Every detector is built with its own defaults: no ``threshold``,
``feature_ranges``, ``range_init``, ``n_init``, ``depth`` or ``window``
is passed. HalfSpaceTrees and LODA therefore learn their ranges from
the first samples they see; OnlineRobustMahalanobis keeps its default
``n_init``; and HalfSpaceTrees, OneClassSGD use their adaptive
thresholds (a quantile of the recent scores). False alarms are computed
by reading ``detector.threshold`` before every ``score_one`` and
comparing the score with the threshold in force at that moment. The
value reported in the ``thr_final`` column is the threshold at the last
healthy sample of the run.

Streams:

- SyntheticArm, payload fault: joints 0-3, input is the residual
  ``tau_meas - ds.nominal_torque``. ``y`` is 0 before the fault and 1
  after. 200 samples at 50 Hz with a 2 s period, fault at 2 s.
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
        n_cycles=2,
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


def evaluate(detector, X, y, learn_on_all):
    n = len(X)
    scores = np.empty(n, dtype=float)
    healthy = y == 0
    t_learn = 0.0
    t_score = 0.0
    n_learned = 0
    n_fa = 0
    n_healthy = 0
    thr_final = float(getattr(detector, "threshold", 0.0))
    for i, x in enumerate(X):
        thr_now = float(getattr(detector, "threshold", 0.0))
        t0 = time.perf_counter()
        scores[i] = detector.score_one(x)
        t_score += time.perf_counter() - t0
        if healthy[i]:
            n_healthy += 1
            if scores[i] > thr_now:
                n_fa += 1
            thr_final = thr_now
        if learn_on_all or healthy[i]:
            t0 = time.perf_counter()
            detector.learn_one(x)
            t_learn += time.perf_counter() - t0
            n_learned += 1
    n_learned = max(n_learned, 1)
    ms = (t_score + t_learn) / n_learned * 1e3
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    fa = float(n_fa) / float(n_healthy) if n_healthy else float("nan")
    return (
        auc,
        fa,
        ms,
        n_learned,
        n_healthy,
        thr_final,
        float(scores.min()),
        float(scores.max()),
    )


def build_one(name, keys):
    if name == "HalfSpaceTrees":
        return HalfSpaceTrees(feature_keys=keys, seed=0)
    if name == "OnlineIsolationForest":
        return OnlineIsolationForest(feature_keys=keys, seed=0)
    if name == "LODA":
        return LODA(feature_keys=keys, seed=0)
    if name == "OneClassSGD":
        return OneClassSGD(feature_keys=keys, seed=0)
    if name == "OnlineRobustMahalanobis":
        return OnlineRobustMahalanobis(feature_keys=keys)
    raise ValueError(f"unknown detector {name!r}")


def run_one(name, X, y):
    keys = list(X[0].keys())
    print("=" * 120)
    print(
        f"stream: {name}  n={len(X)}  features={len(keys)}  "
        f"healthy={int((y == 0).sum())}  positives={int((y == 1).sum())}"
    )
    print("=" * 120)
    header = (
        f"{'detector':<26} {'protocol':<20} {'ROC-AUC':>8} {'FA@thr':>7} "
        f"{'ms/sample':>10} {'n_learn':>8} {'n_healthy':>10} "
        f"{'thr_final':>10} {'s_min':>10} {'s_max':>10}"
    )
    print(header)
    print("-" * len(header))
    for det_name in DETECTOR_NAMES:
        for protocol, learn_on_all in [
            ("learn-on-healthy", False),
            ("learn-on-all", True),
        ]:
            det = build_one(det_name, keys)
            auc, fa, ms, n_learned, n_healthy, thr, smin, smax = evaluate(
                det, X, y, learn_on_all=learn_on_all
            )
            print(
                f"{det_name:<26} {protocol:<20} {auc:>8.3f} {fa:>7.3f} "
                f"{ms:>10.3f} {n_learned:>8} {n_healthy:>10} "
                f"{thr:>10.4f} {smin:>10.4f} {smax:>10.4f}"
            )
    print()


def main():
    print("streaming anomaly detectors on the library's own streams")
    print("protocols: learn-on-healthy, learn-on-all; detectors at their defaults")
    print()
    X_arm, y_arm = arm_stream()
    run_one("SyntheticArm payload fault", X_arm, y_arm)
    X_dr, y_dr = drift_stream()
    run_one("DriftStream sudden drift", X_dr, y_dr)


if __name__ == "__main__":
    main()
