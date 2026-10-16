import json
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT = "/tmp/da"
OUT = "/kaggle/working"
BRANCH = os.environ.get("ZG_BRANCH", "taskD-docs")
SMOKE = os.environ.get("ZG_SMOKE") == "1"


def sh(c):
    return subprocess.run(c, shell=True, check=True, capture_output=True, text=True)


if not SMOKE:
    sh(f"git clone -q --depth 1 -b {BRANCH} https://github.com/tatopenn-cell/Dense-Armor {ROOT}")
    sh(f"pip install -q -e {ROOT} scipy")
else:
    ROOT = "."
sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.forest import AggregatedMondrianForestRegressor as AMF, OnlineExtraTreesRegressor as OXT
from dense_armor.utility.tree import MondrianForestRegressor as MON

SEEDS = 2 if SMOKE else 20
N = 300 if SMOKE else 1000


def make(regime, seed):
    rng = np.random.default_rng(seed)
    if regime == "d20_2relevant":
        X = rng.random((N, 20)); y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=N)
    elif regime == "d100_2relevant":
        X = rng.random((N, 100)); y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=N)
    elif regime == "piecewise_one_feature":
        X = rng.random((N, 4)); y = np.floor(4 * X[:, 0]) / 4 + 0.15 * rng.normal(size=N)
    elif regime == "outliers_5pct":
        X = rng.random((N, 4)); y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=N)
        m = rng.random(N) < 0.05
        y[m] += rng.choice([-5.0, 5.0], size=m.sum())
    elif regime == "cost_limit_1ms":
        X = rng.random((N, 4)); y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=N)
    return X, y


def learners(cls, n, seed):
    if cls == "AMF":
        return AMF(n_trees=n, low=0.0, high=1.0, seed=seed)
    if cls == "OXT":
        return OXT(n_models=n, seed=seed)
    return MON(n_trees=n, seed=seed)


def prequential(est, X, y):
    names = [f"x{i}" for i in range(X.shape[1])]
    err = np.empty(len(y))
    t0 = time.perf_counter()
    for i, (xi, yi) in enumerate(zip(X, y)):
        d = dict(zip(names, map(float, xi)))
        p = est.predict_one(d)
        p = 0.0 if p is None else float(p)
        err[i] = (p - float(yi)) ** 2
        est.learn_one(d, float(yi))
    ms = 1e3 * (time.perf_counter() - t0) / len(y)
    return float(err[len(y) // 2:].mean()), ms


def a12(a, b):
    d = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return float((d < 0).mean() + 0.5 * (d == 0).mean())


def a12_ci(a, b, rng, nb=1000):
    a, b = np.asarray(a), np.asarray(b)
    v = [a12(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(nb))]
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def pick_cost_limit():
    chosen = {}
    X, y = make("cost_limit_1ms", 0)
    for cls in ("AMF", "OXT", "MON"):
        best = 1
        for n in (1, 2, 3, 5, 10, 20):
            _, ms = prequential(learners(cls, n, 0), X[:300], y[:300])
            if ms <= 1.0:
                best = n
        chosen[cls] = best
    return chosen


def cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def cpu_detail():
    d = {}
    keep = ("Vendor ID", "Model name", "CPU family", "Model", "Stepping", "CPU(s)", "Thread(s) per core",
            "Core(s) per socket", "Socket(s)", "CPU MHz", "L1d cache", "L2 cache", "L3 cache",
            "Hypervisor vendor", "BogoMIPS")
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in keep:
                d[k] = v
            if k == "Flags":
                d["avx2"] = " avx2" in " " + v
                d["avx512f"] = "avx512f" in v
    except Exception:
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                d["MemTotal_kB"] = int(line.split()[1])
                break
    except OSError:
        pass
    return d


report = {"machine": {"cpu": cpu(), "cores": os.cpu_count(), "python": sys.version.split()[0], "cpu_detail": cpu_detail()},
          "samples": N, "seeds": SEEDS, "metric": "mean squared error over the second half of the stream, prequential",
          "rule": "GO: lower error than the Mondrian forest with two-sided paired Wilcoxon p < 0.05 and the 95% bootstrap interval of A12 above 0.5, or the same error (interval contains 0.5) at no more than half the ms per sample",
          "regimes": {}}
regimes = ["d20_2relevant", "d100_2relevant", "piecewise_one_feature", "outliers_5pct", "cost_limit_1ms"]
rng_b = np.random.default_rng(0)
try:
    limit = pick_cost_limit()
    report["cost_limit_trees"] = limit
except Exception:
    limit = {"AMF": 3, "OXT": 3, "MON": 5}
    report["cost_limit_failure"] = traceback.format_exc()[-300:]
for regime in regimes:
    t0 = time.time()
    data = {c: {"err": [], "ms": []} for c in ("AMF", "OXT", "MON")}
    try:
        for s in range(SEEDS):
            X, y = make(regime, s)
            for cls in data:
                n = limit[cls] if regime == "cost_limit_1ms" else 10
                e, ms = prequential(learners(cls, n, s), X, y)
                data[cls]["err"].append(e)
                data[cls]["ms"].append(ms)
        rows = {}
        for cls in ("AMF", "OXT"):
            a, b = data[cls]["err"], data["MON"]["err"]
            try:
                p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            lo, hi = a12_ci(a, b, rng_b)
            ms_a, ms_b = float(np.median(data[cls]["ms"])), float(np.median(data["MON"]["ms"]))
            lower = float(np.median(a)) < float(np.median(b))
            go_err = (p < 0.05) and lower and lo > 0.5
            same = (lo <= 0.5 <= hi) and ms_a <= 0.5 * ms_b
            rows[cls] = {"median_error": float(np.median(a)), "mondrian_median_error": float(np.median(b)),
                         "p_two_sided": p, "a12": a12(a, b), "a12_ci95": [lo, hi],
                         "seeds_better": int(sum(x < y for x, y in zip(a, b))),
                         "ms_per_sample": ms_a, "mondrian_ms_per_sample": ms_b,
                         "verdict": "GO" if (go_err or same) else "NO-GO"}
        report["regimes"][regime] = {"pairs": rows, "minutes": (time.time() - t0) / 60}
    except Exception:
        report["regimes"][regime] = {"failure": traceback.format_exc()[-400:]}
    json.dump(report, open(f"{OUT}/zg_report.json" if not SMOKE else "zg_smoke.json", "w"), indent=1)
    print(regime, json.dumps({c: (r["verdict"], round(r["p_two_sided"], 4), round(r["a12"], 3), round(r["ms_per_sample"], 2)) for c, r in report["regimes"][regime].get("pairs", {}).items()}), flush=True)
print(json.dumps(report, indent=1)[:3000])
