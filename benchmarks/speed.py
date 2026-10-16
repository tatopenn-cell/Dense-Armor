"""Speed harness: 5 components x 20 seeds x 2000 samples, 200 warm-up."""

import importlib
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
REF_NAME = os.environ.get("SPEED_REF", "")


def sh(c):
    subprocess.run(c, shell=True, check=True)


sys.path.insert(0, ROOT)

import numpy as np

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.drift.adwin import ADWIN as OurADWIN
from dense_armor.utility.anomaly.hst import HalfSpaceTrees as OurHST
from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier as OurARF
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB as OurNB
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier as OurHT

HAVE_REF = False
if REF_NAME:
    try:
        _ref = importlib.import_module(REF_NAME)
        _r_drift = importlib.import_module(REF_NAME + ".drift")
        _r_forest = importlib.import_module(REF_NAME + ".forest")
        _r_nb = importlib.import_module(REF_NAME + ".naive_bayes")
        _r_tree = importlib.import_module(REF_NAME + ".tree")
        _r_anom = importlib.import_module(REF_NAME + ".anomaly")
        HAVE_REF = True
    except Exception:
        HAVE_REF = False

D = 4


def stream(seed):
    ds = DriftStream(n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed)
    return [(x.to_dict(), y) for x, y in ds.stream()]


def squash(x):
    return {k: 1.0 / (1.0 + math.exp(-v / 3.0)) for k, v in x.items()}


def models(seed):
    m = {
        "gaussian_nb": (OurNB(), (lambda: _r_nb.GaussianNB()) if HAVE_REF else None, "clf"),
        "hoeffding_tree": (
            OurHT(),
            (lambda: _r_tree.HoeffdingTreeClassifier(leaf_prediction="nb")) if HAVE_REF else None,
            "clf",
        ),
        "adaptive_random_forest": (
            OurARF(n_models=10, seed=seed),
            (lambda: _r_forest.ARFClassifier(n_models=10, seed=seed)) if HAVE_REF else None,
            "clf",
        ),
        "half_space_trees": (
            OurHST(n_trees=25, depth=15, window=50, feature_ranges=[(0.0, 1.0)] * D, seed=seed),
            (lambda: _r_anom.HalfSpaceTrees(n_trees=25, height=15, window_size=50, seed=seed)) if HAVE_REF else None,
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
    for name, (ours, ref_f, kind) in models(seed).items():
        for lib, mk in (("ours", ours), ("reference", ref_f)):
            if mk is None:
                continue
            key = f"{name}|{lib}"
            try:
                m = mk if lib == "ours" else mk()
                p, l = run_clf(m, data) if kind == "clf" else run_anom(m, data)
                res["runs"].setdefault(key, []).append(stats(p, l))
            except Exception:
                res["errors"][f"{key}|seed{seed}"] = traceback.format_exc()[-400:]
    rng = np.random.default_rng(seed)
    errs = [float(v) for v in np.concatenate([rng.random(N // 2) < 0.1, rng.random(N - N // 2) < 0.5])]
    for lib, mk in (("ours", lambda: OurADWIN()), ("reference", (lambda: _r_drift.ADWIN()) if HAVE_REF else None)):
        if mk is None:
            continue
        key = f"adwin|{lib}"
        try:
            res["runs"].setdefault(key, []).append(adwin_run(mk(), errs))
        except Exception:
            res["errors"][f"{key}|seed{seed}"] = traceback.format_exc()[-400:]
    _save()
    print("seed", seed, "done", flush=True)

summary = {}
for key, rows in res["runs"].items():
    fields = rows[0].keys()
    summary[key] = {f: float(np.median([r[f] for r in rows])) for f in fields}
res["summary_median_over_seeds"] = summary

try:
    from scipy.stats import wilcoxon
    comp = {}
    for name in ("gaussian_nb", "hoeffding_tree", "adaptive_random_forest",
                 "half_space_trees", "adwin"):
        fld = "update_us_median" if name == "adwin" else "total_us_median"
        a = [r[fld] for r in res["runs"].get(f"{name}|ours", [])]
        b = [r[fld] for r in res["runs"].get(f"{name}|reference", [])]
        if len(a) == len(b) and len(a) >= 6:
            w = wilcoxon(a, b)
            gt = sum(x < y for x, y in zip(a, b))
            comp[name] = {
                "ratio_ours_over_reference": float(np.median(np.array(a) / np.array(b))),
                "ours_faster_on_seeds": int(gt),
                "wilcoxon_two_sided_p": float(w.pvalue),
            }
    res["comparison"] = comp
except Exception:
    res["errors"]["comparison"] = traceback.format_exc()[-400:]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({"summary": summary, "comparison": res.get("comparison"),
                  "errors": list(res["errors"])}, indent=1)[:3500])
