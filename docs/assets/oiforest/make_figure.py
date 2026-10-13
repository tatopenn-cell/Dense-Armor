"""Generate the two figures for docs/anomaly/oiforest.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest
from dense_armor.utility.datasets import DriftStream, SyntheticArm

OUT = Path(__file__).parent
URDF = Path("test/fixtures/urdf/panda.urdf")


def _stream_arm():
    ds = SyntheticArm(
        URDF,
        period_s=2.0, rate_hz=50.0, n_cycles=2,
        fault_at_s=2.0, fault="payload", payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 5e-2), seed=0,
    )
    X, y = [], []
    for sig, label in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        X.append(x)
        y.append(int(label))
    return X, np.asarray(y)


def score_vs_time():
    X_arm, y_arm = _stream_arm()
    det_arm = OnlineIsolationForest(
        feature_keys=["r0", "r1", "r2", "r3"], seed=0,
    )
    s_arm = []
    for x in X_arm:
        s_arm.append(det_arm.score_one(x))
        det_arm.learn_one(x)
    s_arm = np.asarray(s_arm)

    ds_dr = DriftStream(n_samples=2000, n_features=4, n_drifts=1,
                        kind="sudden", spacing=999.0, seed=0)
    det_dr = OnlineIsolationForest(
        feature_keys=["x0", "x1", "x2", "x3"], seed=0,
    )
    s_dr = []
    for sig, _ in ds_dr.stream():
        x = {k: float(sig[k]) for k in sig.names}
        s_dr.append(det_dr.score_one(x))
        det_dr.learn_one(x)
    s_dr = np.asarray(s_dr)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=False)
    axes[0].plot(s_arm, lw=0.8, color="steelblue")
    axes[0].axvline(100, color="black", ls=":", lw=1.0,
                    label="fault at sample 100")
    axes[0].axhline(0.6, color="crimson", ls="--", lw=1.0,
                    label="threshold 0.6")
    axes[0].set_ylabel("OnlineIsolationForest score")
    axes[0].set_xlabel("sample (SyntheticArm, n_cycles=2)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(s_dr, lw=0.8, color="steelblue")
    axes[1].axvline(1000, color="black", ls=":", lw=1.0,
                    label="drift at sample 1000")
    axes[1].axhline(0.6, color="crimson", ls="--", lw=1.0,
                    label="threshold 0.6")
    axes[1].set_ylabel("OnlineIsolationForest score")
    axes[1].set_xlabel("sample (DriftStream)")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "score_vs_time.png", dpi=140)
    plt.close(fig)


def depth_distribution():
    X_arm, y_arm = _stream_arm()
    det = OnlineIsolationForest(
        feature_keys=["r0", "r1", "r2", "r3"], seed=0,
    )
    depths_healthy, depths_fault = [], []
    for i, x in enumerate(X_arm):
        if det._ready:
            v = det._to_vec(x)
            depths = []
            for tr in range(det.n_trees):
                cur = det._trees[tr]
                if cur is not None:
                    depths.append(det._depth_search(cur, v))
            d = float(np.mean(depths)) if depths else 0.0
        else:
            d = 0.0
        if y_arm[i] == 0:
            depths_healthy.append(d)
        else:
            depths_fault.append(d)
        det.learn_one(x)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.linspace(min(depths_healthy + depths_fault) - 0.5,
                       max(depths_healthy + depths_fault) + 0.5, 25)
    ax.hist(depths_healthy, bins=bins, alpha=0.6, color="steelblue",
            label="healthy")
    ax.hist(depths_fault, bins=bins, alpha=0.6, color="crimson",
            label="fault")
    ax.set_xlabel("leaf depth of the sample")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "depth_distribution.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    score_vs_time()
    depth_distribution()
    print("wrote", OUT / "score_vs_time.png")
    print("wrote", OUT / "depth_distribution.png")
