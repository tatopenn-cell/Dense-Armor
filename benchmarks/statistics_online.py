"""Benchmark the three N10 models on three synthetic domains.

For each model (Student-t, Gaussian mixture, GARCH(1,1)) and each
domain (robot, drone, finance), generate a synthetic stream, learn
one sample at a time, measure the median per-sample latency, and
compare with a batch reference where the batch reference is
available (scipy.stats, scipy.optimize, sklearn.mixture).

The synthetic domains are generated inside this script, with a fixed
seed, and are labelled "synthetic" in the JSON and in the printed
table:

- **robot**: a joint position in radians, a slow sinusoid with rare
  spikes and a mean step.
- **drone**: an IMU acceleration in m/s², a random walk with Student-t
  noise and a variance change.
- **finance**: a return stream, Student-t noise with df=3 as in
  ``benchmarks/finance/drift_vs_noise.py``, with two volatility
  regimes.

``--smoke`` runs a smaller configuration. ``--crunchdao`` is a second
block that runs only if the CrunchDAO DataCrunch 2 data is available in
``/content/data/``: it measures whether the three models as feature
extractors add signal beyond the ``Feature_*`` columns. See the module
docstring of ``_crunchdao`` below.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dense_armor.utility.stats import OnlineGaussianMixture, RunningMoments, StudentT  # noqa: E402
from dense_armor.utility.timeseries import GARCH11  # noqa: E402

SEED = 20261026
N_FULL = 10000
N_SMOKE = 2000


class DomainCfg(TypedDict):
    nu: float
    init_means: list[list[float]]


class Summary(TypedDict):
    median: float | None
    p90: float | None
    max: float | None
    n: int


def _summary(values: list[float]) -> Summary:
    if not values:
        return {"median": None, "p90": None, "max": None, "n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


def _robot(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * math.pi, n)
    q = 0.1 * np.sin(t)
    spikes = rng.random(n) < 0.02
    q[spikes] += rng.normal(0.0, 0.5, int(spikes.sum()))
    step = n // 2
    q[step:] += 0.2
    return q


def _drone(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(df=4.0, size=n) * 0.5
    walk = np.cumsum(rng.normal(0.0, 0.05, n))
    fault = n // 2
    walk[fault:] *= 3.0
    return walk + noise


def _finance(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.ones(n) * 0.01
    vol[n // 2 :] = 0.03
    return rng.standard_t(df=3.0, size=n) * vol


def _bimodal(n: int, seed: int, sep: float = 2.0, noise: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    left = rng.normal(-sep, noise, half)
    right = rng.normal(+sep, noise, half)
    x = np.concatenate([left, right])
    rng.shuffle(x)
    return x


def _timed_learn(model, xs: np.ndarray) -> list[float]:
    samples = [{"x": float(v)} for v in xs]
    lat = []
    for s in samples:
        t0 = time.perf_counter()
        model.learn_one(s)
        lat.append((time.perf_counter() - t0) * 1e6)
    return lat


def _student_t_domain(
    xs: np.ndarray, nu: float = 5.0, learning_rate: float = 0.05
) -> dict:
    m = StudentT(nu=nu, learning_rate=learning_rate)
    lat = _timed_learn(m, xs)
    naive = RunningMoments()
    for v in xs:
        naive.learn_one({"x": float(v)})
    out = {
        "n_": m.n_,
        "mu_": float(m.mu_),
        "sigma_": float(m.sigma_),
        "naive_mean": float(naive.mean),
        "naive_std": float(naive.std),
        "latency_us": _summary(lat),
    }
    try:
        from scipy.stats import t as _t

        params = _t.fit(xs)
        out["batch_scipy"] = {
            "df": float(params[0]),
            "loc": float(params[1]),
            "scale": float(params[2]),
        }
    except Exception:
        out["batch_scipy"] = "not computed"
    return out


def _mixture_domain(
    xs: np.ndarray, init_means: list[list[float]] | None = None
) -> dict:
    m = OnlineGaussianMixture(n_components=2, learning_rate=0.05, init_means=init_means)
    lat = _timed_learn(m, xs)
    means = sorted(float(v) for v in m.means_.ravel())
    weights = [float(v) for v in m.weights_]
    out = {
        "n_": m.n_,
        "means_": means,
        "weights_": weights,
        "latency_us": _summary(lat),
    }
    try:
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=2, random_state=SEED, n_init=1).fit(
            xs.reshape(-1, 1)
        )
        out["batch_sklearn"] = {
            "means": sorted(float(v) for v in gm.means_.ravel()),
            "weights": [float(v) for v in gm.weights_],
        }
    except Exception:
        out["batch_sklearn"] = "not computed"
    return out


def _garch_domain(xs: np.ndarray) -> dict:
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    lat = _timed_learn(m, xs)
    out = {
        "n_": m.n_,
        "sigma2_": float(m.sigma2_),
        "epsilon_": float(m.epsilon_),
        "latency_us": _summary(lat),
    }
    return out


def _run_domain(
    name: str,
    xs: np.ndarray,
    nu: float,
    init_means: list[list[float]],
    learning_rate: float = 0.05,
) -> dict:
    return {
        "domain": name,
        "n_samples": int(xs.size),
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "learning_rate": learning_rate,
        "student_t": _student_t_domain(xs, nu=nu, learning_rate=learning_rate),
        "mixture": _mixture_domain(xs, init_means=init_means),
        "garch": _garch_domain(xs),
    }


def _crunchdao() -> dict:
    """Measure whether the three features add signal on CrunchDAO data.

    Needs ``/content/data/X_train.parquet`` and ``y_train.parquet``
    (the quick starter of CrunchDAO DataCrunch 2 downloads them). For
    each row, compute the three features, fit a ``LinearRegression``
    on ``Feature_*`` alone (baseline) and on ``Feature_*`` + the three
    features (augmented), measure the Pearson correlation per ``moon``
    on the last 20% of the moons, report both.

    N10 stops at the feature vector and the measurement. It does not
    build the regressor that competes on the leaderboard.
    """
    data = Path("/content/data")
    x_path = data / "X_train.parquet"
    y_path = data / "y_train.parquet"
    if not (x_path.exists() and y_path.exists()):
        return {"status": "not computed, data not in /content/data/"}
    try:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
    except ImportError as e:
        return {"status": f"not computed, missing dependency: {e}"}

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    df = X.merge(y, on=["moon", "id"], how="inner")
    feature_cols = [c for c in df.columns if c.startswith("Feature_")]
    df = df.sort_values(["moon", "id"]).reset_index(drop=True)

    st = StudentT(nu=5.0, learning_rate=0.05)
    mx = OnlineGaussianMixture(n_components=2, learning_rate=0.05)
    ga = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)

    rows = []
    for _, row in df.iterrows():
        s = {"x": float(row[feature_cols[0]])}
        st.learn_one(s)
        mx.learn_one(s)
        ga.learn_one(s)
        rows.append(
            {
                "st_logpdf": -st.score_one(s),
                "mx_resp0": float(mx.transform_one(s)["responsibilities"][0]),
                "garch_sigma2": float(ga.transform_one(s)["sigma2"]),
            }
        )
    feat = pd.DataFrame(rows)
    for c in feat.columns:
        df[c] = feat[c].values

    moons = sorted(df["moon"].unique())
    cut = moons[int(len(moons) * 0.8)]
    train = df[df["moon"] < cut]
    test = df[df["moon"] >= cut]
    if len(test) == 0:
        return {"status": "not computed, no test moons after the split"}

    def _pearson(cols: list[str]) -> float:
        model = LinearRegression().fit(train[cols], train["target"])
        pred = model.predict(test[cols])
        if float(np.std(pred)) == 0.0:
            return 0.0
        return float(np.corrcoef(pred, test["target"])[0, 1])

    base = _pearson(feature_cols)
    aug = _pearson(feature_cols + ["st_logpdf", "mx_resp0", "garch_sigma2"])
    return {
        "status": "computed",
        "baseline_pearson": base,
        "augmented_pearson": aug,
        "delta": aug - base,
        "test_moons": int(test["moon"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smaller configuration")
    ap.add_argument(
        "--robust",
        action="store_true",
        help="StudentT with learning_rate=0.005 and the bimodal domain",
    )
    ap.add_argument(
        "--crunchdao", action="store_true", help="also run the CrunchDAO block"
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "benchmarks/statistics_online.json"
    )
    args = ap.parse_args()

    n = N_SMOKE if args.smoke else N_FULL
    lr = 0.005 if args.robust else 0.05
    out: dict = {
        "seed": SEED,
        "n_samples": n,
        "synthetic": True,
        "robust": bool(args.robust),
        "learning_rate": lr,
        "domains": [],
    }
    per_domain: dict[str, DomainCfg] = {
        "robot": {"nu": 3.0, "init_means": [[0.0], [0.2]]},
        "drone": {"nu": 4.0, "init_means": [[-1.0], [1.0]]},
        "finance": {"nu": 3.0, "init_means": [[-0.02], [0.02]]},
    }
    plan = [("robot", _robot), ("drone", _drone), ("finance", _finance)]
    if args.robust:
        plan.append(("bimodal", _bimodal))
        per_domain["bimodal"] = {"nu": 5.0, "init_means": [[-2.0], [2.0]]}
    for name, gen in plan:
        xs = gen(n, SEED)
        cfg = per_domain[name]
        res = _run_domain(
            name, xs, nu=cfg["nu"], init_means=cfg["init_means"], learning_rate=lr
        )
        out["domains"].append(res)
        print(f"--- {name} (n={n}, lr={lr}) ---")
        print(
            f"  student_t: mu={res['student_t']['mu_']:+.4f}, "
            f"sigma={res['student_t']['sigma_']:.4f}, "
            f"naive mu={res['student_t']['naive_mean']:+.4f}, "
            f"naive std={res['student_t']['naive_std']:.4f}, "
            f"lat={res['student_t']['latency_us']['median']:.1f} us"
        )
        print(
            f"  mixture:   means={[round(m, 3) for m in res['mixture']['means_']]}, "
            f"weights={[round(w, 3) for w in res['mixture']['weights_']]}, "
            f"lat={res['mixture']['latency_us']['median']:.1f} us"
        )
        print(
            f"  garch:     sigma2={res['garch']['sigma2_']:.3e}, "
            f"lat={res['garch']['latency_us']['median']:.1f} us"
        )

    if args.crunchdao:
        print("\n--- crunchdao ---")
        c = _crunchdao()
        out["crunchdao"] = c
        if c["status"] == "computed":
            print(
                f"  baseline={c['baseline_pearson']:.4f}, "
                f"augmented={c['augmented_pearson']:.4f}, "
                f"delta={c['delta']:+.4f}"
            )

    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
