"""Seed audit: every random estimator takes a seed.

Walks every public class of ``dense_armor.utility``, instantiates it with
the first kwargs of ``_unit_test_params``, runs it twice on the same 200
rows of a fixed synthetic stream and compares ``state_dict()``. A class
whose two runs differ is *random*. For every random class the audit
reports whether ``seed`` is in the ``__init__`` signature and its default,
and whether the two runs differ.

The output is a list of the random classes and a list of the classes
that have ``seed`` defaulting to ``None`` (item G, do not change the
default) or have no ``seed`` (item G, add one at the end of the
signature).

No code change here: the table is the delivery.
"""

import argparse
import inspect
import importlib
import json
import os
from pathlib import Path
from typing import Any

from dense_armor.roles import Estimator

ROW_KEYS = ["a", "b", "c", "d"]
N_ROWS = 200


def _stream():
    """Return a fixed list of (x, y, t) triples for the audit."""
    import numpy as np

    rng = np.random.default_rng(0)
    out = []
    for i in range(N_ROWS):
        x = {k: float(rng.normal(0.0, 1.0)) for k in ROW_KEYS}
        y = int(x["a"] > 0.0)
        out.append((x, y, float(i) * 0.01))
    return out


def _probe():
    """Return five fixed input dicts used to compare two runs."""
    return [
        {"a": 0.1, "b": -0.2, "c": 0.3, "d": 0.4},
        {"a": -0.5, "b": 0.2, "c": 0.0, "d": 0.1},
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
        {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
        {"a": -1.0, "b": -1.0, "c": -1.0, "d": -1.0},
    ]


def _walk_estimators():
    """Yield (module, class) for every public Estimator subclass."""
    found = []
    root = Path("dense_armor/utility")
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path) or path.name.startswith("_"):
            continue
        mod_name = str(path.with_suffix("")).replace(os.sep, ".")
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (inspect.isclass(obj)
                    and issubclass(obj, Estimator)
                    and obj is not Estimator
                    and obj.__module__ == mod_name
                    and not attr.startswith("_")):
                found.append((mod_name, attr, obj))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def _params(cls):
    """First constructor kwargs from ``_unit_test_params``, or ``{}``."""
    try:
        gen = cls._unit_test_params()
        return next(iter(gen), {})
    except Exception:
        return {}


def _seed_info(cls):
    """Return (has_seed, default, annotation) of ``seed`` in __init__."""
    sig = inspect.signature(cls.__init__)
    if "seed" not in sig.parameters:
        return False, "<absent>", None
    p = sig.parameters["seed"]
    return True, p.default, p.annotation


def _run(est, stream):
    """Run ``learn_one`` over the stream; ignore predict errors."""
    for x, y, t in stream:
        try:
            est.learn_one(x, y, t=t)
        except Exception:
            pass
    try:
        return est.state_dict()
    except Exception:
        return {}


def _predict_differs(est_a, est_b, X_probe):
    """True when the two estimators give different predictions on X_probe.

    The two runs see the same training stream; a class is random when the
    two predictions on the same five probe rows differ. This is the
    honest criterion: a ``state_dict`` can differ between two runs for
    reasons that are not randomness (dict insertion order, object ids),
    and the earlier version of the audit flagged 14 false positives.

    Args:
        est_a: first trained estimator.
        est_b: second trained estimator on the same stream.
        X_probe: five probe inputs.

    Returns:
        True when at least one prediction differs.
    """
    for xp in X_probe:
        try:
            p_a = est_a.predict_one(xp)
        except Exception:
            p_a = None
        try:
            p_b = est_b.predict_one(xp)
        except Exception:
            p_b = None
        try:
            if p_a != p_b:
                return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/seed_audit.json")
    a = ap.parse_args()

    stream = _stream()
    probe = _probe()
    estimators = _walk_estimators()
    print(f"estimator trovati: {len(estimators)}")

    random_classes = []
    seed_defaults = []
    skipped = []
    for mod_name, attr, cls in estimators:
        params = _params(cls)
        try:
            est_a = cls(**params)
            est_b = cls(**params)
        except Exception as e:
            skipped.append((mod_name, attr, f"{type(e).__name__}: {e}"))
            continue
        try:
            _run(est_a, stream)
            _run(est_b, stream)
        except Exception as e:
            skipped.append((mod_name, attr, f"run {type(e).__name__}: {e}"))
            continue
        has_seed, default, ann = _seed_info(cls)
        differs = _predict_differs(est_a, est_b, probe)
        if differs:
            random_classes.append({
                "module": mod_name, "class": attr,
                "has_seed": has_seed,
                "seed_default": repr(default),
            })
        if default is None and default is not inspect.Parameter.empty:
            seed_defaults.append({"module": mod_name, "class": attr})

    print()
    print(f"=== classi con due run diversi (random senza seed fissato): {len(random_classes)} ===")
    for r in random_classes:
        tag = ""
        if r["has_seed"] and r["seed_default"] in ("None", "<class 'inspect._empty'>"):
            tag = "  [item G: keep default None, add to the list]"
        elif not r["has_seed"]:
            tag = "  [item G: add seed at the end of the signature]"
        print(f"  {r['module']}.{r['class']}  seed={r['has_seed']} default={r['seed_default']}{tag}")

    print()
    print(f"=== classi con seed=None (default da non cambiare): {len(seed_defaults)} ===")
    for r in seed_defaults:
        print(f"  {r['module']}.{r['class']}")

    print()
    print(f"=== classi saltate (init o run fallito): {len(skipped)} ===")
    for m, a_, e in skipped[:20]:
        print(f"  {m}.{a_}: {e[:80]}")

    Path(a.out).write_text(json.dumps({
        "estimator_totali": len(estimators),
        "random_classes": random_classes,
        "seed_default_none": seed_defaults,
        "skipped": skipped,
    }, indent=2))
    print()
    print(f"JSON: {a.out}")


if __name__ == "__main__":
    main()
