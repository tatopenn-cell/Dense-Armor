"""Quality benchmark: the ensemble package, 20 seeds, one drifted stream.

Classifiers: online bagging, online boosting, voting and stacking against
one ``HoeffdingTreeClassifier``, on ``DriftStream`` with one sudden change,
4 features, 2000 samples, prequential protocol (predict first, then learn).
Voting and stacking get members that differ from each other.

Regressor: ``ExponentiallyWeightedAverageRegressor`` against its own best
member, on a synthetic regression stream with a known linear rule and a
drift at half. The measured cumulative loss of the ensemble is reported
next to ``bound()``; the bound of Freund and Schapire (1997) assumes the
loss is in ``[0, 1]``, so the loss is clipped to that interval for the
comparison and the clipping is reported.

Costs: milliseconds per sample of each class against one tree.

Reported per learner: median loss (or accuracy), IQR, ms per sample.
Reported per pair: two-sided paired Wilcoxon p-value, the A12 effect size
with a bootstrap 95% interval, the seeds on which each side is better. A
gain is called a gain only when p < 0.05 and the A12 interval is above
0.5.

Run: ``python benchmarks/ensemble_quality.py`` in Colab (about 15 minutes
on a 2-core CPU). Fast smoke test: ``--quick``.
"""

import argparse
import json
import multiprocessing as mp
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.ensemble import (
    ExponentiallyWeightedAverageRegressor,
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.roles import Regressor
from dense_armor.utility.neighbors import KNNRegressor
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier

N = 2000
D = 4
N_SEEDS = 20
CLIP = 1.0
PAIRS = [
    ("bagging", "tree"),
    ("boosting", "tree"),
    ("voting", "tree"),
    ("stacking", "tree"),
]
LEARNER_NAMES = ["tree", "bagging", "boosting", "voting", "stacking"]


def _stream(seed: int):
    ds = DriftStream(
        n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed
    )
    return [(x.to_dict(), int(y)) for x, y in ds.stream()]


def _single_tree():
    return HoeffdingTreeClassifier(grace_period=50)


def _make_classifiers():
    return {
        "tree": _single_tree,
        "bagging": lambda: OnlineBaggingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50),
            n_models=5,
            seed=0,
        ),
        "boosting": lambda: OnlineBoostingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50),
            n_models=5,
            seed=0,
        ),
        "voting": lambda: VotingClassifier(
            [
                lambda: HoeffdingTreeClassifier(grace_period=20),
                lambda: HoeffdingTreeClassifier(grace_period=50),
                lambda: HoeffdingTreeClassifier(grace_period=100),
            ],
            mode="hard",
        ),
        "stacking": lambda: StackingClassifier(
            [
                lambda: HoeffdingTreeClassifier(grace_period=20),
                lambda: HoeffdingTreeClassifier(grace_period=100),
            ],
            lambda: HoeffdingTreeClassifier(grace_period=20),
        ),
    }


def _run_classifiers(seed: int):
    data = _stream(seed)
    out = {}
    for name, make in _make_classifiers().items():
        model = make()
        err = np.empty(len(data))
        t0 = time.perf_counter()
        for i, (x, y) in enumerate(data):
            p = model.predict_one(x)
            err[i] = float(p != y)
            model.learn_one(x, y)
        dt = (time.perf_counter() - t0) / len(data)
        out[name] = {
            "accuracy": float(1.0 - err.mean()),
            "ms_per_sample": 1e3 * dt,
        }
    return seed, out


def _regression_stream(seed: int):
    """A non-linear rule with a sudden change and additive noise.

    The rule is ``y = a . x + 0.5 * sin(2 * x0) + noise`` before the
    change and ``a_after . x + 0.5 * sin(2 * x0) + noise`` after it.
    The sine makes the rule non-linear, so no member fits it exactly;
    the noise (sigma 0.1) removes the perfect-fit corner. Every member
    is finite and worse than the rule itself, which is what the
    weighted average needs.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, D))
    half = N // 2
    a = np.array([1.0, 2.0, -1.0, 0.5])
    a_after = np.array([1.0, 2.0, 1.0, 0.5])
    noise = rng.normal(0.0, 0.1, size=N)
    y = np.empty(N)
    y[:half] = X[:half] @ a + 0.5 * np.sin(2.0 * X[:half, 0])
    y[half:] = X[half:] @ a_after + 0.5 * np.sin(2.0 * X[half:, 0])
    y += noise
    return X, y


def _make_regressors():
    """Three members on different scales: local, smooth, and constant.

    The local one adapts quickly and is noisy; the smooth one is more
    stable and worse right after the change; the constant one is the
    weakest and exists so the Hedge weights have to push it down. The
    spread between the three is what the weighted average has to
    exploit.
    """
    return {
        "knn_small": lambda: KNNRegressor(window_size=30, n_neighbors=3),
        "knn_large": lambda: KNNRegressor(window_size=200, n_neighbors=10),
        "mean": lambda: _MeanRegressor(),
    }


class _MeanRegressor(Regressor):
    """A constant regressor: the running mean of the targets seen so far.

    It is the weakest member in the ensemble. It exists so the
    exponentially weighted average has a member that is plainly wrong
    on the linear rule, and the Hedge weights have to push it down.
    """

    def __init__(self) -> None:
        self.mean_ = 0.0
        self.n_seen_ = 0

    def learn_one(self, x, y, t=None):
        self.n_seen_ += 1
        self.mean_ += (float(y) - self.mean_) / self.n_seen_
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self.mean_





def _run_regressor(seed: int):
    X, y = _regression_stream(seed)
    names = [f"x{i}" for i in range(D)]
    factories = _make_regressors()
    losses = {name: 0.0 for name in factories}
    ms = {name: 0.0 for name in factories}
    ewa = ExponentiallyWeightedAverageRegressor(
        [factories[name] for name in factories], eta=0.5
    )
    ewa_loss = 0.0
    t0_ewa = time.perf_counter()
    for i in range(N):
        d = dict(zip(names, map(float, X[i])))
        yt = float(y[i])
        p_ewa = float(ewa.predict_one(d))
        clipped = float(np.clip((p_ewa - yt) ** 2, 0.0, CLIP))
        ewa_loss += clipped
        ewa.learn_one(d, yt)
        for name, model in zip(factories, ewa.members_):
            p = float(model.predict_one(d))
            losses[name] += float(np.clip((p - yt) ** 2, 0.0, CLIP))
    dt_ewa = (time.perf_counter() - t0_ewa) / N
    best_name = min(losses, key=losses.get)
    return seed, {
        "ewa_cumulative_loss": ewa_loss,
        "ewa_bound": float(ewa.bound()),
        "best_member": best_name,
        "best_member_loss": float(losses[best_name]),
        "per_member_loss": {k: float(v) for k, v in losses.items()},
        "ewa_ms_per_sample": 1e3 * dt_ewa,
    }


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    """Probability that a random a beats a random b; higher is better for a.

    Scores are oriented so that a larger number is a better result
    (accuracy for classifiers, negative loss for the regressor). A12
    above 0.5 means a is better than b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _a12_ci(a, b, rng, n_boot: int = 2000):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = _a12(a[idx], b[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _pairwise(a_vals, b_vals, label_a: str, label_b: str) -> dict:
    a = np.asarray(a_vals)
    b = np.asarray(b_vals)
    try:
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    a12 = _a12(a, b)
    lo, hi = _a12_ci(a, b, np.random.default_rng(0))
    diff = a - b
    return {
        "a": label_a,
        "b": label_b,
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "p_two_sided": p,
        "a12": a12,
        "a12_ci95": [lo, hi],
        "seeds_a_better": int(np.sum(diff > 0)),
        "seeds_b_better": int(np.sum(diff < 0)),
        "seeds_tie": int(np.sum(diff == 0)),
    }


def _machine() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": mp.cpu_count(),
    }


def main(seeds: int, workers: int, out_path: Path) -> None:
    print("machine:", _machine(), flush=True)
    print(f"seeds: {seeds}, samples: {N}, features: {D}", flush=True)
    t0 = time.perf_counter()

    clf_results = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(_run_classifiers, range(seeds)), 1
            ):
                clf_results.append(res)
                print(f"  clf [{i:3d}/{seeds}]  seed={res[0]:2d}  "
                      f"elapsed={time.perf_counter()-t0:6.1f}s", flush=True)
    else:
        for i, seed in enumerate(range(seeds), 1):
            res = _run_classifiers(seed)
            clf_results.append(res)
            print(f"  clf [{i:3d}/{seeds}]  seed={seed:2d}  "
                  f"elapsed={time.perf_counter()-t0:6.1f}s", flush=True)

    reg_results = []
    for i, seed in enumerate(range(seeds), 1):
        res = _run_regressor(seed)
        reg_results.append(res)
        print(f"  reg [{i:3d}/{seeds}]  seed={seed:2d}  "
              f"elapsed={time.perf_counter()-t0:6.1f}s", flush=True)

    total_s = time.perf_counter() - t0
    print(f"\ntotal: {total_s:.1f}s = {total_s/60:.1f} min")

    clf = defaultdict(lambda: defaultdict(list))
    ms = defaultdict(list)
    for seed, res in clf_results:
        for name, r in res.items():
            clf[name]["accuracy"].append(r["accuracy"])
            ms[name].append(r["ms_per_sample"])

    print()
    header = (f"{'learner':10s} {'acc-med':>9s} {'acc-IQR':>9s} "
              f"{'ms/sample':>10s}")
    print(header)
    print("-" * len(header))
    for name in LEARNER_NAMES:
        if name not in clf:
            continue
        a = clf[name]["accuracy"]
        m = ms[name]
        print(f"{name:10s} {np.median(a):9.4f} "
              f"{np.percentile(a, 75) - np.percentile(a, 25):9.4f} "
              f"{np.median(m):10.2f}")

    tests = []
    print()
    print("=" * 100)
    print("PAIRWISE, accuracy over the whole stream")
    print("=" * 100)
    for a_name, b_name in PAIRS:
        r = _pairwise(
            clf[a_name]["accuracy"], clf[b_name]["accuracy"], a_name, b_name
        )
        tests.append(r)
        n = (r["seeds_a_better"] + r["seeds_b_better"] + r["seeds_tie"])
        print(f"  {a_name:10s} vs {b_name:8s}  "
              f"medA={r['median_a']:.4f}  medB={r['median_b']:.4f}  "
              f"p={r['p_two_sided']:.4f}  "
              f"A12={r['a12']:.3f} "
              f"[{r['a12_ci95'][0]:.3f},{r['a12_ci95'][1]:.3f}]  "
              f"A-better {r['seeds_a_better']}/{n}")

    ewa_losses = np.array([r["ewa_cumulative_loss"] for _, r in reg_results])
    best_losses = np.array([r["best_member_loss"] for _, r in reg_results])
    bounds = np.array([r["ewa_bound"] for _, r in reg_results])
    ewa_ms = np.array([r["ewa_ms_per_sample"] for _, r in reg_results])
    print()
    print("=" * 100)
    print("EWA vs its own best member (cumulative clipped loss over the stream)")
    print("=" * 100)
    print(f"  ewa median={np.median(ewa_losses):.2f}  "
          f"best-member median={np.median(best_losses):.2f}  "
          f"bound median={np.median(bounds):.2f}  "
          f"ms/sample median={np.median(ewa_ms):.4f}")
    for seed, r in reg_results:
        print(f"    seed={seed}: ewa={r['ewa_cumulative_loss']:.2f}  "
              f"bound={r['ewa_bound']:.2f}  "
              f"members={r['per_member_loss']}  best={r['best_member']}")
    reg_pair = _pairwise(-ewa_losses, -best_losses, "ewa", "best_member")
    n = (reg_pair["seeds_a_better"] + reg_pair["seeds_b_better"]
         + reg_pair["seeds_tie"])
    print(f"  ewa vs best: medEwa={np.median(ewa_losses):.2f}  "
          f"medBest={np.median(best_losses):.2f}  "
          f"p={reg_pair['p_two_sided']:.4f}  "
          f"A12={reg_pair['a12']:.3f} "
          f"[{reg_pair['a12_ci95'][0]:.3f},{reg_pair['a12_ci95'][1]:.3f}]  "
          f"A-better {reg_pair['seeds_a_better']}/{n}")

    payload = {
        "machine": _machine(),
        "seeds": seeds,
        "samples": N,
        "features": D,
        "stream": "DriftStream sudden, one change, 4 features, 2000 samples",
        "loss_clipped_to": [0.0, CLIP],
        "classifiers": {
            name: {
                "accuracy": clf[name]["accuracy"],
                "accuracy_median": float(np.median(clf[name]["accuracy"])),
                "accuracy_iqr": float(
                    np.percentile(clf[name]["accuracy"], 75)
                    - np.percentile(clf[name]["accuracy"], 25)
                ),
                "ms_per_sample": ms[name],
                "ms_per_sample_median": float(np.median(ms[name])),
            }
            for name in LEARNER_NAMES
            if name in clf
        },
        "pairwise_accuracy": tests,
        "ewa": {
            "cumulative_loss": ewa_losses.tolist(),
            "bound": bounds.tolist(),
            "best_member_loss": best_losses.tolist(),
            "best_member_names": [r["best_member"] for _, r in reg_results],
            "cumulative_loss_median": float(np.median(ewa_losses)),
            "best_member_loss_median": float(np.median(best_losses)),
            "bound_median": float(np.median(bounds)),
            "ms_per_sample_median": float(np.median(ewa_ms)),
        },
        "pairwise_ewa": reg_pair,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"JSON: {out_path}  ({out_path.stat().st_size} B)")


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/ensemble_quality.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 2
        args.workers = 1
        args.out = "benchmarks/ensemble_quality_quick.json"
    main(args.seeds, args.workers, Path(args.out))
