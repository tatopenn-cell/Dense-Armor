# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/utility/online_classifiers.py."""
import doctest
import importlib
import sys
import time

import numpy as np
import pytest

pytest.importorskip("river")

import dense_armor.utility.online_classifiers as online_classifiers  # noqa: E402
from dense_armor.utility.online_classifiers import (  # noqa: E402
    OnlineGaussianNB, OnlineSoftmaxRegression, DriftAdaptiveClassifier,
)
from dense_armor.utility.river_drift import CUSUMDriftDetector  # noqa: E402


def _batch_nb(X, y):
    classes = sorted(set(y.tolist()))
    means = {c: X[y == c].mean(axis=0) for c in classes}
    vars_ = {c: ((X[y == c] - means[c]) ** 2).mean(axis=0) for c in classes}
    priors = {c: float(np.sum(y == c)) / len(y) for c in classes}
    return classes, means, vars_, priors


def _batch_nb_proba(x, classes, means, vars_, priors, eps=1e-12):
    log_p = {}
    for c in classes:
        var = np.maximum(vars_[c], eps)
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - means[c]) ** 2 / var)
        log_p[c] = float(ll) + float(np.log(priors[c]))
    mx = max(log_p.values())
    ex = {c: float(np.exp(v - mx)) for c, v in log_p.items()}
    s = sum(ex.values())
    return {c: p / s for c, p in ex.items()}


def test_gaussian_nb_matches_batch_at_alpha_zero():
    rng = np.random.default_rng(0)
    n, d = 500, 4
    X = rng.standard_normal((n, d)) + rng.integers(0, 3, n)[:, None] * 2.0
    y = rng.integers(0, 3, n)
    nb = OnlineGaussianNB(alpha=0.0)
    for i in range(n):
        xi = {f"f{j}": float(X[i, j]) for j in range(d)}
        _ = nb.learn_one(xi, int(y[i]))
    classes, means, vars_, priors = _batch_nb(X, y)
    max_diff = 0.0
    for _ in range(20):
        xq = rng.standard_normal(d)
        xi = {f"f{j}": float(xq[j]) for j in range(d)}
        online = nb.predict_proba_one(xi)
        batch = _batch_nb_proba(xq, classes, means, vars_, priors)
        for c in classes:
            max_diff = max(max_diff, abs(online[c] - batch[c]))
    print(f"\nNB vs batch, max prob diff: {max_diff:.2e}")
    assert max_diff < 1e-10


def test_softmax_learns_separable_two_class():
    rng = np.random.default_rng(0)
    sr = OnlineSoftmaxRegression(eta=0.5)
    for _ in range(200):
        a = float(rng.standard_normal())
        _ = sr.learn_one({"a": a - 2.0}, 0)
        _ = sr.learn_one({"a": a + 2.0}, 1)
    assert sr.predict_one({"a": -3.0}) == 0
    assert sr.predict_one({"a": 3.0}) == 1


def test_drift_adaptive_on_synthetic_shift():
    rng = np.random.default_rng(0)
    clf = DriftAdaptiveClassifier(
        OnlineGaussianNB(alpha=0.01),
        CUSUMDriftDetector(reference="adaptive", radius=10, ref_mult=3),
        smooth_window=20, window=50,
    )
    X = np.concatenate([rng.standard_normal((500, 4)) + 0.0,
                        rng.standard_normal((500, 4)) + 5.0])
    y = np.array([0] * 500 + [1] * 500)
    for i in range(len(X)):
        xi = {f"f{j}": float(X[i, j]) for j in range(4)}
        _ = clf.learn_one(xi, int(y[i]))
    assert clf.predict_one({f"f{j}": 0.0 for j in range(4)}) == 0
    assert clf.predict_one({f"f{j}": 5.0 for j in range(4)}) == 1


def _simulated_robot_signal(seed=42, n=6000, d=4):
    """Three robot states with a change of the contact signature at n / 2.

    States follow a Markov chain that stays in the current state with
    probability 0.95 and otherwise jumps to one of the other two (mean dwell
    20 samples). Joint-feature means: free (0, 0, 0, 0), contact
    (2, 2, 0, 0), collision (4, 4, 4, 4); unit noise. From sample n / 2 the
    contact load moves to the other two joints, (0, 0, 2, 2).
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = (states[t - 1] if rng.random() < 0.95
                     else (states[t - 1] + rng.integers(1, 3)) % 3)
    sig = np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0],
                    [4.0, 4.0, 4.0, 4.0]])[:, :d]
    means = sig[states]
    after = (np.arange(n) >= n // 2) & (states == 1)
    means[after] = np.array([0.0, 0.0, 2.0, 2.0])[:d]
    X = means + rng.standard_normal((n, d))
    return X, states


def _detector():
    return CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5,
                              two_sided=False)


def _prequential_hits(clf, X, y):
    hits = np.zeros(len(X))
    for i in range(len(X)):
        xi = {f"f{j}": float(X[i, j]) for j in range(X.shape[1])}
        pred = clf.predict_one(xi)
        hits[i] = 1.0 if pred == y[i] else 0.0
        _ = clf.learn_one(xi, int(y[i]))
    return hits


def test_simulated_robot_signal():
    X, y = _simulated_robot_signal()
    n = len(X)
    print("\nSimulated robot signal (contact signature change at 3000)")
    print(f"{'Classifier':<44} {'pre':>6} {'post':>6} {'late':>6} {'rec':>6}")

    configs = [
        ("OnlineGaussianNB(alpha=0)",
         lambda: OnlineGaussianNB(alpha=0.0)),
        ("OnlineGaussianNB(alpha=0.01)",
         lambda: OnlineGaussianNB(alpha=0.01)),
        ("OnlineGaussianNB(alpha=0.1)",
         lambda: OnlineGaussianNB(alpha=0.1)),
        ("OnlineSoftmaxRegression(eta=0.1,l2=1e-4)",
         lambda: OnlineSoftmaxRegression(eta=0.1, l2=1e-4)),
        ("DriftAdaptive(NB alpha=0)",
         lambda: DriftAdaptiveClassifier(
             OnlineGaussianNB(alpha=0.0),
             _detector(), smooth_window=50, window=50)),
        ("DriftAdaptive(SR)",
         lambda: DriftAdaptiveClassifier(
             OnlineSoftmaxRegression(eta=0.1, l2=1e-4),
             _detector(), smooth_window=50, window=50)),
    ]

    results = {}
    for name, factory in configs:
        clf = factory()
        hits = _prequential_hits(clf, X, y)
        pre = float(hits[500:3000].mean())
        post = float(hits[3000:3200].mean())
        late = float(hits[5000:5900].mean())
        rec = None
        for t in range(3000, n - 200):
            if hits[t:t + 200].mean() >= pre - 0.05:
                rec = t - 3000
                break
        results[name] = (pre, post, late, rec)
        print(f"{name:<44} {pre:>6.3f} {post:>6.3f} {late:>6.3f} {str(rec):>6}")
    plain = results["OnlineGaussianNB(alpha=0)"]
    wrapped = results["DriftAdaptive(NB alpha=0)"]
    assert wrapped[0] >= plain[0] - 0.01
    assert wrapped[3] is not None and wrapped[3] < plain[3]
    assert wrapped[2] > plain[2]


def test_per_sample_time():
    X, y = _simulated_robot_signal(n=2000)
    print("\nPer-sample time (us):")
    for name, clf in [
        ("OnlineGaussianNB", OnlineGaussianNB(alpha=0.01)),
        ("OnlineSoftmaxRegression",
         OnlineSoftmaxRegression(eta=0.1, l2=1e-4)),
        ("DriftAdaptive(NB)", DriftAdaptiveClassifier(
            OnlineGaussianNB(alpha=0.01),
            _detector(), smooth_window=50, window=50)),
    ]:
        t0 = time.perf_counter()
        for i in range(len(X)):
            xi = {f"f{j}": float(X[i, j]) for j in range(X.shape[1])}
            _ = clf.learn_one(xi, int(y[i]))
        dt = time.perf_counter() - t0
        print(f"  {name:<30}: {dt / len(X) * 1e6:>8.2f}")


def test_platt_wraps_binary_gaussian_nb():
    from dense_armor.utility.calibration import OnlinePlattScaling
    base = OnlineGaussianNB(alpha=0.0)
    wrapped = OnlinePlattScaling(base)
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = float(rng.standard_normal())
        b = float(rng.standard_normal())
        _ = wrapped.learn_one({"a": a - 2.0, "b": b - 2.0}, 0)
        _ = wrapped.learn_one({"a": a + 2.0, "b": b + 2.0}, 1)
    p = wrapped.predict_proba_one({"a": -3.0, "b": -3.0})
    assert p[True] < p[False]


def test_doctest():
    assert doctest.testmod(online_classifiers).failed == 0


def test_missing_river_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules,
                       "dense_armor.utility.online_classifiers",
                       raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.utility.online_classifiers")


def test_check_estimator():
    from river.checks import check_estimator
    check_estimator(OnlineGaussianNB())
    check_estimator(OnlineSoftmaxRegression())
    check_estimator(DriftAdaptiveClassifier(
        OnlineGaussianNB(), CUSUMDriftDetector(reference="adaptive")))
