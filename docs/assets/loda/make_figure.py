"""Generate the two figures for docs/anomaly/loda.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.datasets import DriftStream, SyntheticArm

OUT = Path(__file__).parent
URDF = Path("test/fixtures/urdf/panda.urdf")


def histogram_one_projection():
    rng = np.random.default_rng(0)
    det = LODA(n_projections=100, n_bins=10, window=200, range_init=50, seed=0)
    for _ in range(300):
        det.learn_one({
            "a": float(rng.normal(0.5, 0.05)),
            "b": float(rng.normal(0.5, 0.05)),
        })
    x_healthy = {"a": 0.5, "b": 0.5}
    x_far = {"a": 5.0, "b": 5.0}
    proj_h = float(det._W[0] @ det._to_norm(x_healthy))
    proj_f = float(det._W[0] @ det._to_norm(x_far))
    bin_h = int(det._bin_index(np.array([proj_h]))[0])
    bin_f = int(det._bin_index(np.array([proj_f]))[0])
    pmin = float(det._pmin[0])
    pmax = float(det._pmax[0])
    edges = np.linspace(pmin, pmax, det.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(centers, det._counts[0], width=widths * 0.9, color="steelblue",
           edgecolor="black", linewidth=0.4, label="window count")
    ax.axvline(proj_h, color="green", lw=2.0, label=f"healthy (bin {bin_h})")
    ax.axvline(proj_f, color="crimson", lw=2.0, label=f"far (bin {bin_f})")
    ax.set_xlabel("projection value (projection 0 of 100)")
    ax.set_ylabel("window count")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "histogram.png", dpi=140)
    plt.close(fig)


def score_vs_time():
    ds_arm = SyntheticArm(
        URDF,
        period_s=2.0, rate_hz=50.0, n_cycles=2,
        fault_at_s=2.0, fault="payload", payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 5e-2), seed=0,
    )
    det_arm = LODA(
        n_projections=100, n_bins=10, window=256, range_init=20,
        feature_keys=["r0", "r1", "r2", "r3"], threshold=4.0, seed=0,
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
    det_dr = LODA(
        n_projections=100, n_bins=10, window=256, range_init=20,
        feature_keys=["x0", "x1", "x2", "x3"], threshold=4.0, seed=0,
    )
    s_dr = []
    for sig, _ in ds_dr.stream():
        x = {k: float(sig[k]) for k in sig.names}
        s_dr.append(det_dr.score_one(x))
        det_dr.learn_one(x)
    s_dr = np.asarray(s_dr)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=False)
    axes[0].plot(s_arm, lw=0.8, color="steelblue")
    axes[0].axvline(100, color="black", ls=":", lw=1.0, label="fault at sample 100")
    axes[0].axhline(4.0, color="crimson", ls="--", lw=1.0, label="threshold 4.0")
    axes[0].set_ylabel("LODA score")
    axes[0].set_xlabel("sample (SyntheticArm)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylim(0.0, 6.0)

    axes[1].plot(s_dr, lw=0.8, color="steelblue")
    axes[1].axvline(1000, color="black", ls=":", lw=1.0, label="drift at sample 1000")
    axes[1].axhline(4.0, color="crimson", ls="--", lw=1.0, label="threshold 4.0")
    axes[1].set_ylabel("LODA score")
    axes[1].set_xlabel("sample (DriftStream)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_ylim(0.0, 6.0)

    fig.tight_layout()
    fig.savefig(OUT / "score_vs_time.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    histogram_one_projection()
    score_vs_time()
    print("wrote", OUT / "histogram.png")
    print("wrote", OUT / "score_vs_time.png")
