"""Generate the two figures for docs/anomaly/hst.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.datasets import DriftStream, SyntheticArm

OUT = Path(__file__).parent
URDF = Path("test/fixtures/urdf/panda.urdf")


def _cuts(det, tr, node, lo, hi, depth, segments):
    if depth == 0 or node >= len(det._feature[tr]):
        return
    f = int(det._feature[tr, node])
    m = float(det._midpoints[tr, node])
    other = 1 - f
    segments.append((f, m, max(lo[other], 0.0), min(hi[other], 1.0)))
    hi_l = hi.copy()
    hi_l[f] = m
    lo_r = lo.copy()
    lo_r[f] = m
    _cuts(det, tr, 2 * node + 1, lo, hi_l, depth - 1, segments)
    _cuts(det, tr, 2 * node + 2, lo_r, hi, depth - 1, segments)


def tree_cuts():
    det = HalfSpaceTrees(
        n_trees=3,
        depth=5,
        window=20,
        feature_ranges=[(0.0, 1.0), (0.0, 1.0)],
        seed=0,
    )
    det.learn_one({"a": 0.5, "b": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for tr, ax in enumerate(axes):
        lo, hi = det._work_lo[tr].copy(), det._work_hi[tr].copy()
        segs = []
        _cuts(det, tr, 0, lo, hi, 5, segs)
        for f, m, a, b in segs:
            if not (0.0 <= m <= 1.0):
                continue
            if f == 0:
                ax.plot([m, m], [a, b], color="steelblue", lw=0.8)
            else:
                ax.plot([a, b], [m, m], color="steelblue", lw=0.8)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=1.2))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        wl, wh = lo.round(2), hi.round(2)
        ax.set_title(
            f"tree {tr}: work space x [{wl[0]}, {wh[0]}], y [{wl[1]}, {wh[1]}]",
            fontsize=7,
        )
        ax.set_xlabel("r0 (normalised)")
        ax.set_ylabel("r1 (normalised)")
    fig.tight_layout()
    fig.savefig(OUT / "tree_cuts.png", dpi=140)
    plt.close(fig)


def _run(det, rows, learn_on_healthy=True):
    scores, thr = [], []
    for x, y in rows:
        thr.append(det.threshold)
        scores.append(det.score_one(x))
        if y == 0 or not learn_on_healthy:
            det.learn_one(x)
    return np.array(scores), np.array(thr)


def score_vs_time():
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
    rows, t_arm = [], []
    for sig, y in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        rows.append(
            ({f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}, int(y))
        )
        t_arm.append(sig.t)
    keys = list(rows[0][0])
    det = HalfSpaceTrees(feature_keys=keys, seed=0)
    s_arm, _ = _run(det, rows)

    dr = DriftStream(
        n_samples=2000, n_features=4, n_drifts=1, kind="sudden", spacing=999.0, seed=0
    )
    samples = [{k: float(sig[k]) for k in sig.names} for sig, _ in dr.stream()]
    rows_d = [(x, int(c == 1)) for x, c in zip(samples, dr.concepts_)]
    det_d = HalfSpaceTrees(feature_keys=list(rows_d[0][0]), seed=0)
    s_d, thr_d = _run(det_d, rows_d)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0))
    ax = axes[0]
    ax.plot(t_arm, s_arm, color="steelblue", lw=1.0, label="score")
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="0.5 (uniform mass)")
    ax.axvline(2.0, color="crimson", ls=":", lw=1.2, label="fault at t = 2 s")
    ax.set_ylabel("score")
    ax.set_xlabel("time (s)")
    ax.set_title("SyntheticArm, learning on the healthy part", fontsize=9)
    ax.legend(loc="center left", fontsize=8)
    ax = axes[1]
    ax.plot(s_d, color="steelblue", lw=0.6, label="score")
    shown = np.where(thr_d < 1e8, thr_d, np.nan)
    ax.plot(shown, color="darkorange", lw=1.2, label="adaptive threshold")
    ax.axvline(1000, color="crimson", ls=":", lw=1.2, label="drift")
    ax.set_ylabel("score")
    ax.set_xlabel("sample")
    ax.set_title("DriftStream, learning on the healthy part", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_time.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    tree_cuts()
    score_vs_time()
