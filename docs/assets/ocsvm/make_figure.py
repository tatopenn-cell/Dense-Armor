"""Generate the two figures for docs/anomaly/ocsvm.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.anomaly.ocsvm import OneClassSGD
from dense_armor.utility.datasets import DriftStream, SyntheticArm

OUT = Path(__file__).parent
URDF = Path("test/fixtures/urdf/panda.urdf")


def score_vs_time():
    ds_arm = SyntheticArm(
        URDF,
        period_s=2.0, rate_hz=50.0, n_cycles=2,
        fault_at_s=2.0, fault="payload", payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 5e-2), seed=0,
    )
    det_arm = OneClassSGD(
        feature_keys=["r0", "r1", "r2", "r3"], seed=0,
    )
    s_arm = []
    for sig, _ in ds_arm.stream():
        q, qd, qdd = ds_arm.trajectory(sig.t)
        tau_nom = ds_arm.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        s_arm.append(det_arm.score_one(x))
        det_arm.learn_one(x)
    s_arm = np.asarray(s_arm)

    ds_dr = DriftStream(n_samples=2000, n_features=4, n_drifts=1,
                        kind="sudden", spacing=999.0, seed=0)
    det_dr = OneClassSGD(feature_keys=["x0", "x1", "x2", "x3"], seed=0)
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
    axes[0].axhline(0.0, color="crimson", ls="--", lw=1.0,
                    label="zero boundary")
    axes[0].set_ylabel("OneClassSGD score")
    axes[0].set_xlabel("sample (SyntheticArm, n_cycles=2)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(s_dr, lw=0.8, color="steelblue")
    axes[1].axvline(1000, color="black", ls=":", lw=1.0,
                    label="drift at sample 1000")
    axes[1].axhline(0.0, color="crimson", ls="--", lw=1.0,
                    label="zero boundary")
    axes[1].set_ylabel("OneClassSGD score")
    axes[1].set_xlabel("sample (DriftStream)")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "score_vs_time.png", dpi=140)
    plt.close(fig)


def convergence():
    rng = np.random.default_rng(0)
    n_total = 20000
    x_healthy = {"a": 0.5, "b": 0.5}
    det = OneClassSGD(
        n_features_rff=128, nu=0.05, n_init=20, seed=0,
    )
    scores = []
    for i in range(n_total):
        if i % 500 == 0:
            scores.append((i, det.score_one(x_healthy)))
        det.learn_one({
            "a": float(rng.normal(0.5, 0.05)),
            "b": float(rng.normal(0.5, 0.05)),
        })
    scores.append((n_total, det.score_one(x_healthy)))

    xs = [p[0] for p in scores]
    ys = [p[1] for p in scores]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(xs, ys, "o-", lw=1.2, color="steelblue")
    ax.axhline(0.0, color="crimson", ls="--", lw=1.0, label="boundary (0)")
    ax.set_xscale("log")
    ax.set_xlabel("samples learned")
    ax.set_ylabel("score on the fixed healthy point")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    score_vs_time()
    convergence()
    print("wrote", OUT / "score_vs_time.png")
    print("wrote", OUT / "convergence.png")
