"""Generate the four PNGs of docs/assets/lof/.

Run from the repository root:

    python docs/assets/lof/make_figure.py

Every figure comes from the library's own classes on the same streams
the tests use. No external data, no hard-coded numbers.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

_roc_auc_score = None

def _roc_auc_score(y_true, y_score):
    """ROC-AUC via the Mann-Whitney U statistic (no external dependency)."""
    pairs = sorted(zip(y_score, y_true))
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (rank + (rank + j - i)) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg
        rank += j - i + 1
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.anomaly.lof import LocalOutlierFactor
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.anomaly.ocsvm import OneClassSGD
from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest
from dense_armor.utility.datasets import DriftStream, SyntheticArm

HERE = Path(__file__).resolve().parent
URDF = ROOT / "test/fixtures/urdf/panda.urdf"


def fig_paper_example():
    """The dense / sparse / near-point example of the paper's introduction."""
    rng = np.random.default_rng(0)
    C1 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(400, 2))
    C2 = rng.normal(loc=(3.0, 3.0), scale=0.15, size=(100, 2))
    o1 = np.array([3.0 + 0.8, 3.0 + 0.8])

    model = LocalOutlierFactor(n_neighbors=20, window=550)
    for p in C1:
        model.learn_one({"a": float(p[0]), "b": float(p[1])})
    for p in C2:
        model.learn_one({"a": float(p[0]), "b": float(p[1])})

    lof_C1 = np.array(
        [model.score_one({"a": float(p[0]), "b": float(p[1])}) for p in C1]
    )
    lof_C2 = np.array(
        [model.score_one({"a": float(p[0]), "b": float(p[1])}) for p in C2]
    )
    lof_o1 = model.score_one({"a": float(o1[0]), "b": float(o1[1])})

    fig, ax = plt.subplots(figsize=(9, 7))
    sc1 = ax.scatter(
        C1[:, 0],
        C1[:, 1],
        c=lof_C1,
        cmap="viridis",
        vmin=0.9,
        vmax=max(lof_C1.max(), lof_o1),
        s=12,
        alpha=0.8,
        label="sparse cluster (400)",
    )
    ax.scatter(
        C2[:, 0],
        C2[:, 1],
        c=lof_C2,
        cmap="viridis",
        vmin=0.9,
        vmax=max(lof_C1.max(), lof_o1),
        s=20,
        alpha=0.9,
        label="dense cluster (100)",
    )
    ax.scatter(
        [o1[0]],
        [o1[1]],
        c=[lof_o1],
        cmap="viridis",
        vmin=0.9,
        vmax=max(lof_C1.max(), lof_o1),
        s=200,
        edgecolor="red",
        linewidth=2,
        label=f"o1, LOF={lof_o1:.1f}",
    )
    ax.set_title(
        "LOF on the paper's introduction example (k=20)", fontsize=12, weight="bold"
    )
    ax.set_xlabel("feature a")
    ax.set_ylabel("feature b")
    ax.legend(loc="upper left", fontsize=10)
    fig.colorbar(sc1, ax=ax, label="LOF")
    fig.tight_layout()
    fig.savefig(HERE / "lof_paper_example.png", dpi=140)
    plt.close(fig)
    print(
        f"  lof_paper_example.png: o1 LOF={lof_o1:.3f}, max C1={lof_C1.max():.3f}, max C2={lof_C2.max():.3f}"
    )


def fig_arm_trajectory():
    """LOF score over the SyntheticArm payload fault, joint residuals."""
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
    X, y = [], []
    for sig, label in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        X.append(x)
        y.append(int(label))
    y = np.asarray(y)
    model = LocalOutlierFactor(n_neighbors=20, window=100, threshold=1.5)
    scores = np.empty(len(X))
    t_list = np.empty(len(X))
    for i, (x, (sig, _)) in enumerate(zip(X, ds.stream())):
        scores[i] = model.score_one(x)
        model.learn_one(x)
        t_list[i] = sig.t

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_list, scores, color="#2c6fa6", lw=1.5, label="LOF score")
    ax.axhline(1.5, color="#c0392b", ls="--", lw=1.5, label="threshold 1.5")
    ax.axvline(2.0, color="#27ae60", ls=":", lw=2, label="fault (payload 5 kg)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("LOF")
    ax.set_yscale("log")
    ax.set_title(
        "LOF over the SyntheticArm payload fault (k=20, window=100)",
        fontsize=12,
        weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "lof_arm_trajectory.png", dpi=140)
    plt.close(fig)
    auc = _roc_auc_score(y, scores)
    print(f"  lof_arm_trajectory.png: AUC={auc:.3f}")


def _comparison_run():
    """Run 6 detectors x 2 protocols x 5 seeds on DriftStream sudden."""
    import time

    builders = {
        "HST": lambda k: HalfSpaceTrees(feature_keys=k, seed=0),
        "LODA": lambda k: LODA(feature_keys=k, seed=0),
        "OIForest": lambda k: OnlineIsolationForest(feature_keys=k, seed=0),
        "OCSVM": lambda k: OneClassSGD(feature_keys=k, seed=0),
        "Mahalanobis": lambda k: OnlineRobustMahalanobis(feature_keys=k),
        "LOF": lambda k: LocalOutlierFactor(feature_keys=k, seed=0),
    }
    out = {n: {"learn-on-healthy": [], "learn-on-all": []} for n in builders}
    for seed in range(5):
        ds = DriftStream(
            n_samples=2000, n_features=4, n_drifts=1,
            kind="sudden", spacing=999.0, seed=seed,
        )
        X = [{k: float(sig[k]) for k in sig.names} for sig, _ in ds.stream()]
        y = (ds.concepts_ == 1).astype(int)
        keys = list(X[0].keys())
        healthy = y == 0
        for name, make in builders.items():
            for proto, on_all in (
                ("learn-on-healthy", False), ("learn-on-all", True)
            ):
                det = make(keys)
                sc = np.empty(len(X))
                for i, xi in enumerate(X):
                    sc[i] = det.score_one(xi)
                    if on_all or healthy[i]:
                        det.learn_one(xi)
                out[name][proto].append(float(_roc_auc_score(y, sc)))
    return out


def fig_comparison():
    """AUC mediana su 5 seed, 6 detector x 2 protocolli, DriftStream sudden."""
    comp_seeds = _comparison_run()
    names = ["LOF", "HST", "LODA", "OIForest", "OCSVM", "Mahalanobis"]

    def _med(name, proto):
        vals = comp_seeds[name][proto]
        return float(np.median(vals)), float(np.percentile(vals, 75) - np.percentile(vals, 25))

    h_med = [_med(n, "learn-on-healthy")[0] for n in names]
    a_med = [_med(n, "learn-on-all")[0] for n in names]
    h_iqr = [_med(n, "learn-on-healthy")[1] for n in names]
    a_iqr = [_med(n, "learn-on-all")[1] for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(
        x - w / 2,
        h_med,
        w,
        yerr=h_iqr,
        capsize=5,
        color="#2c6fa6",
        label="learn-on-healthy",
    )
    ax.bar(
        x + w / 2,
        a_med,
        w,
        yerr=a_iqr,
        capsize=5,
        color="#c0392b",
        label="learn-on-all",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("ROC-AUC (median of 5 seeds, IQR bar)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_title(
        "Anomaly detectors on DriftStream, sudden drift, 2000 samples",
        fontsize=12,
        weight="bold",
    )
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "lof_comparison.png", dpi=140)
    plt.close(fig)
    print(f"  lof_comparison.png: LOF healthy={h_med[0]:.3f}, LOF all={a_med[0]:.3f}")


def fig_budget():
    """p50 / p99 / max vs window, on 3000 samples."""
    import time

    rng = np.random.default_rng(0)
    X = [
        {"a": float(rng.normal()), "b": float(rng.normal()),
         "c": float(rng.normal()), "d": float(rng.normal())}
        for _ in range(3000)
    ]
    W = [25, 50, 100, 200, 400, 800]
    p50 = []
    p99 = []
    mx = []
    for w in W:
        model = LocalOutlierFactor(n_neighbors=20, window=w)
        lat = np.empty(len(X))
        for i, xi in enumerate(X):
            t0 = time.perf_counter()
            model.score_one(xi)
            model.learn_one(xi)
            lat[i] = (time.perf_counter() - t0) * 1e6
        p50.append(float(np.percentile(lat, 50)))
        p99.append(float(np.percentile(lat, 99)))
        mx.append(float(lat.max()))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(W, p50, "-o", color="#2c6fa6", lw=2, label="p50")
    ax.plot(W, p99, "-s", color="#c0392b", lw=2, label="p99")
    ax.plot(W, mx, "-^", color="#27ae60", lw=2, label="max")
    ax.axhline(10000, color="gray", ls="--", lw=1.5, label="budget_s = 10 000 us")
    ax.set_xlabel("window")
    ax.set_ylabel("time per score+learn (us)")
    ax.set_title(
        "LOF cost on 3000 samples (d=4, n_neighbors=20)", fontsize=12, weight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "lof_budget.png", dpi=140)
    plt.close(fig)
    print(f"  lof_budget.png: p99 at W=25 is {p99[0]:.0f} us")


if __name__ == "__main__":
    print("Generating docs/assets/lof/")
    fig_paper_example()
    fig_arm_trajectory()
    fig_comparison()
    fig_budget()
    print("Done.")
