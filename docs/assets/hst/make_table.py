"""Generate the HST numbers table in docs/anomaly/hst.md section 5.

Streams:
- SyntheticArm payload fault (5 kg at t=2s), residuals r0..r3.
- DriftStream(n_samples=2000, n_features=4, n_drifts=1, kind="sudden", seed=0).

Detector: HalfSpaceTrees(n_trees=25, depth=8, window=200, range_init=100, seed=0)
with adaptive threshold (quantile 0.95 of last 500 scores, warmup 50).
Tested before learn_one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from sklearn.metrics import roc_auc_score
from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.datasets import SyntheticArm, DriftStream

URDF = Path("test/fixtures/urdf/panda.urdf")


def arm_stream():
    ds = SyntheticArm(URDF, period_s=2.0, rate_hz=50.0, n_cycles=4,
                      fault_at_s=2.0, fault="payload", payload_mass=5.0,
                      noise_std=(1e-3, 5e-3, 5e-2), seed=0)
    for sig, fault in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        yield x, int(fault)


def drift_stream():
    ds = DriftStream(n_samples=2000, n_features=4, n_drifts=1,
                     kind="sudden", seed=0)
    for sig, y in ds.stream():
        x = {f"x{i}": float(sig[f"x{i}"]) for i in range(4)}
        yield x, int(y)


def run(stream):
    det = HalfSpaceTrees(n_trees=25, depth=8, window=200,
                         range_init=100, threshold=None, seed=0)
    scores, labels = [], []
    for x, y in stream:
        s = det.score_one(x)
        det.learn_one(x)
        scores.append(s); labels.append(y)
    return np.asarray(scores), np.asarray(labels)


def table(name, stream):
    scores, labels = run(stream)
    auc = roc_auc_score(labels, scores)
    healthy = labels == 0
    fa_fixed = float(np.mean(scores[healthy] > 4.0))

    from collections import deque
    q_win = deque(maxlen=500)
    flagged = []
    for s in scores:
        thr = 1e9 if len(q_win) < 50 else float(np.quantile(q_win, 0.95))
        flagged.append(s > thr)
        q_win.append(s)
    fa_adapt = float(np.mean(np.asarray(flagged)[healthy]))
    print(f"{name}: n={len(scores)} AUC={auc:.3f} fa_fixed={fa_fixed:.3f} fa_adapt={fa_adapt:.3f}")


print("Table for docs/anomaly/hst.md section 5")
table("SyntheticArm", arm_stream())
table("DriftStream", drift_stream())
