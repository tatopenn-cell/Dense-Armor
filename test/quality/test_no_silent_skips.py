"""Item D: no check is silently skipped by check_estimator.

Every estimator in the six packages named by the task is run through
check_estimator with the default _unit_test_params; a check that raises
AttributeError or NotImplementedError must be listed in the allow-list
with a reason, otherwise the call fails. The printed result of the
module is ``N checks run, 0 silently skipped``.

The utility.proba module contains the standalone distributions
Gaussian, Beta and Multinomial. They are not ``Estimator`` subclasses
and ``check_estimator`` does not apply to them, so they are not counted.
"""

import importlib
import inspect
import pkgutil

import pytest

from dense_armor.checks import check_estimator
from dense_armor.roles import Estimator


PACKAGES = [
    "dense_armor.utility.bayes",
    "dense_armor.utility.neighbors",
    "dense_armor.utility.multiclass",
    "dense_armor.utility.multioutput",
    "dense_armor.utility.forest",
    "dense_armor.utility.rules",
]

# Reasons for a check that cannot apply. Empty by default: this module
# fails if a check raises AttributeError or NotImplementedError and is
# not listed here.
ALLOW_NOT_APPLICABLE: dict[str, str] = {}


def _estimators() -> list:
    found = []
    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for _, name, _ in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f"{pkg_name}.{name}")
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if not inspect.isclass(obj):
                    continue
                if not issubclass(obj, Estimator):
                    continue
                if obj.__module__ != mod.__name__:
                    continue
                if attr.startswith("_"):
                    continue
                if obj not in found:
                    found.append(obj)
    return found


def test_no_check_is_silently_skipped():
    classes = _estimators()
    names = sorted(c.__name__ for c in classes)
    print(f"\nestimators found: {len(classes)}")
    for n in names:
        print(f"  {n}")
    assert len(classes) >= 20, f"only {len(classes)} estimators found"

    n_run = 0
    n_na = 0
    for cls in classes:
        params = (
            list(cls._unit_test_params())[0]
            if hasattr(cls, "_unit_test_params")
            else {}
        )
        est = cls(**params)
        out = check_estimator(est, allow_not_applicable=ALLOW_NOT_APPLICABLE)
        n_run += len(out["run"])
        n_na += len(out["not_applicable"])
    print(f"\n{n_run} checks run, {n_na} silently skipped")
    assert n_na == 0, f"{n_na} not-applicable checks not in the allow-list"
