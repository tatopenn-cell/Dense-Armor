"""Audit of the public classes of ``dense_armor/ml/`` against scikit-learn tools.

One row per public class of ``dense_armor.ml``: which of ``fit``, ``predict``,
``predict_proba``, ``transform`` it has, the signature of ``fit_transform``,
whether it has ``score``, whether ``sklearn.base.clone`` works, and whether
it runs in a ``sklearn.pipeline.Pipeline`` of two steps on a synthetic table.

Usage: ``python tools/audit_ml_sklearn.py``. Exit code 0 when the audit ran.
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np

import dense_armor.ml as ml
from dense_armor.ml._sklearn_compat import SklearnCompat, estimator_kind
from dense_armor.ml.base import MLBase

warnings.filterwarnings("ignore")

CELLS = ("fit", "predict", "predict_proba", "transform", "fit_transform", "score")


def _classes():
    seen = {}
    for m in pkgutil.iter_modules(ml.__path__):
        mod = importlib.import_module(f"dense_armor.ml.{m.name}")
        for name, obj in vars(mod).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, MLBase)
                and obj is not MLBase
                and not name.startswith("_")
                and obj.__module__ == mod.__name__
            ):
                seen[f"{m.name}.{name}"] = obj
    return seen


def _table():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    return X, (X[:, 0] > 0).astype(int), X @ np.array([1.0, 2.0, 3.0])


def _has(cls, cell):
    own = any(cell in vars(k) for k in cls.__mro__ if k not in (MLBase, SklearnCompat))
    if cell == "score":
        return own or estimator_kind(cls) in ("classifier", "regressor")
    if cell == "fit_transform":
        return own or hasattr(cls, "transform")
    return hasattr(cls, cell)


def _fit_transform_sig(cls):
    if not _has(cls, "fit_transform"):
        return "-"
    f = getattr(cls, "fit_transform")
    params = [p for p in inspect.signature(f).parameters if p != "self"]
    return "(" + ", ".join(params) + ")"


def _clone_ok(est):
    from sklearn.base import clone

    try:
        clone(est)
        return "yes"
    except Exception as e:
        return "no:" + type(e).__name__


def _pipeline_ok(cls):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    X, yc, _ = _table()
    try:
        est = cls()
    except TypeError:
        return "needs args"
    if not hasattr(est, "transform"):
        return "-"
    try:
        make_pipeline(est, LogisticRegression()).fit(X, yc)
        return "yes"
    except Exception as e:
        return "no:" + type(e).__name__


def main():
    rows = []
    for key, cls in sorted(_classes().items()):
        cells = {c: ("yes" if _has(cls, c) else "-") for c in CELLS}
        cells["fit_transform"] = _fit_transform_sig(cls)
        try:
            est = cls()
            cl = _clone_ok(est)
        except TypeError:
            cl = "needs args"
        rows.append((key, cells, cl, _pipeline_ok(cls)))
    head = "| class | " + " | ".join(CELLS) + " | clone | pipeline |"
    print(head)
    print("|" + "---|" * (len(CELLS) + 3))
    for key, cells, cl, pl in rows:
        print(f"| {key} | " + " | ".join(cells[c] for c in CELLS) + f" | {cl} | {pl} |")
    n = len(rows)
    print(f"\nclasses: {n}")
    for c in CELLS:
        print(f"with {c}: {sum(1 for r in rows if r[1][c] != '-')}")
    print("clone works:", sum(1 for r in rows if r[2] == "yes"))
    print("pipeline works:", sum(1 for r in rows if r[3] == "yes"))
    print("pipeline fails:", sum(1 for r in rows if r[3].startswith("no")))


if __name__ == "__main__":
    main()
