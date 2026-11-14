"""Speed harness: 5 components x 20 seeds x 2000 samples, 200 warm-up."""

import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

SMOKE = os.environ.get("SPEED_SMOKE") == "1"
N = 300 if SMOKE else 2000
SEEDS = 1 if SMOKE else 20
WARM = 50 if SMOKE else 200
OUT = os.environ.get("SPEED_OUT", "/content/speed.json")
ROOT = os.environ.get("SPEED_ROOT", "/content/Dense-Armor")


def sh(c):
    subprocess.run(c, shell=True, check=True)


sys.path.insert(0, ROOT)

import numpy as np

from dense_armor.core.online.anomaly.hst import HalfSpaceTrees as OurHST
from dense_armor.core.online.drift.adwin import ADWIN as OurADWIN
from dense_armor.core.online.forest.arf import AdaptiveRandomForestClassifier as OurARF
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB as OurNB
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier as OurHT
from dense_armor.core.signals.datasets import DriftStream

D = 4


def stream(seed):
    ds = DriftStream(n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed)
    return [(x.to_dict(), y) for x, y in ds.stream()]


def squash(x):
    return {k: 1.0 / (1.0 + math.exp(-v / 3.0)) for k, v in x.items()}


def models(seed):
    m = {
        "gaussian_nb": (OurNB(), "clf"),
        "hoeffding_tree": (OurHT(), "clf"),
        "adaptive_random_forest": (OurARF(n_models=10, seed=seed), "clf"),
        "half_space_trees": (
            OurHST(n_trees=25, depth=15, window=50, feature_ranges=[(0.0, 1.0)] * D, seed=seed),
            "anom",
        ),
    }
    return m


def run_clf(model, data):
    p_ns, l_ns = [], []
    for x, y in data:
        t0 = time.perf_counter_ns()
        model.predict_one(x)
        t1 = time.perf_counter_ns()
        model.learn_one(x, y)
        t2 = time.perf_counter_ns()
        p_ns.append(t1 - t0)
        l_ns.append(t2 - t1)
    return p_ns, l_ns


def run_anom(model, data):
    p_ns, l_ns = [], []
    for x, _ in data:
        x = squash(x)
        t0 = time.perf_counter_ns()
        model.score_one(x)
        t1 = time.perf_counter_ns()
        model.learn_one(x)
        t2 = time.perf_counter_ns()
        p_ns.append(t1 - t0)
        l_ns.append(t2 - t1)
    return p_ns, l_ns


def stats(p_ns, l_ns):
    p = np.array(p_ns[WARM:], float) / 1e3
    l = np.array(l_ns[WARM:], float) / 1e3
    t = p + l
    first = (p_ns[WARM] + l_ns[WARM]) / 1e3 if len(p_ns) > WARM else 0.0
    return {
        "predict_us_median": float(np.median(p)),
        "predict_us_p99": float(np.percentile(p, 99)),
        "predict_us_p999": float(np.percentile(p, 99.9)),
        "predict_us_max": float(p.max()),
        "learn_us_median": float(np.median(l)),
        "learn_us_p99": float(np.percentile(l, 99)),
        "learn_us_p999": float(np.percentile(l, 99.9)),
        "learn_us_max": float(l.max()),
        "total_us_median": float(np.median(t)),
        "total_us_p99": float(np.percentile(t, 99)),
        "total_us_p999": float(np.percentile(t, 99.9)),
        "total_us_max": float(t.max()),
        "first_call_us": float(first),
    }


def adwin_run(det, errs):
    ns = []
    for e in errs:
        t0 = time.perf_counter_ns()
        det.update(e)
        ns.append(time.perf_counter_ns() - t0)
    a = np.array(ns[WARM:], float) / 1e3
    first = ns[WARM] / 1e3 if len(ns) > WARM else 0.0
    return {
        "update_us_median": float(np.median(a)),
        "update_us_p99": float(np.percentile(a, 99)),
        "update_us_p999": float(np.percentile(a, 99.9)),
        "update_us_max": float(a.max()),
        "update_us_first_call": float(first),
    }


def cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


res = {
    "machine": {"cpu": cpu(), "cores": os.cpu_count(),
                "python": sys.version.split()[0], "platform": platform.platform()},
    "n_samples": N, "seeds": SEEDS, "warmup_discarded": WARM,
    "versions": {}, "runs": {}, "errors": {},
}
try:
    import jax
    res["versions"]["jax"] = jax.__version__
except Exception:
    pass
res["versions"]["numpy"] = np.__version__

def _save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

# resume: se il JSON esiste, riusa i run
done_seeds = set()
if os.path.exists(OUT):
    try:
        prev = json.loads(open(OUT).read())
        res["runs"] = prev.get("runs", {})
        res["errors"] = prev.get("errors", {})
        for key, rows in res["runs"].items():
            done_seeds.add(len(rows) - 1)
        print("resume: seed gia' completati", sorted(done_seeds), flush=True)
    except Exception:
        pass

_save()

for seed in range(SEEDS):
    if seed in done_seeds:
        continue
    data = stream(seed)
    for name, (model, kind) in models(seed).items():
        try:
            p, l = run_clf(model, data) if kind == "clf" else run_anom(model, data)
            res["runs"].setdefault(name, []).append(stats(p, l))
        except Exception:
            res["errors"][f"{name}|seed{seed}"] = traceback.format_exc()[-400:]
    rng = np.random.default_rng(seed)
    errs = [float(v) for v in np.concatenate([rng.random(N // 2) < 0.1, rng.random(N - N // 2) < 0.5])]
    try:
        res["runs"].setdefault("adwin", []).append(adwin_run(OurADWIN(), errs))
    except Exception:
        res["errors"][f"adwin|seed{seed}"] = traceback.format_exc()[-400:]
    _save()
    print("seed", seed, "done", flush=True)

summary = {}
for key, rows in res["runs"].items():
    fields = rows[0].keys()
    summary[key] = {f: float(np.median([r[f] for r in rows])) for f in fields}
res["summary_median_over_seeds"] = summary

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({"summary": summary, "errors": list(res["errors"])}, indent=1)[:3500])
