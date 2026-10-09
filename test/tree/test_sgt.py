import math

import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.metrics import MeanSquaredError
from dense_armor.utility.tree.sgt import (
    SGTClassifier,
    SGTRegressor,
    _student_t_sf,
)


def _pendulum_tau(q, qd, qdd, m=1.0, l=1.0, g=9.81, b=0.1):
    return m * l * l * qdd + b * qd + m * g * l * math.sin(q)


def test_sgt_t_cdf_against_scipy():
    scipy = __import__("pytest").importorskip("scipy.stats")
    for t in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.5):
        for df in (1, 5, 30):
            got = _student_t_sf(t, df)
            want = float(scipy.t.sf(t, df))
            assert abs(got - want) < 1e-3, (t, df, got, want)


def test_sgt_regressor_beats_running_mean():
    rng = np.random.default_rng(0)
    n = 4000
    xs = [
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        for _ in range(n)
    ]
    ys = [_pendulum_tau(q, qd, qdd) + rng.normal(0.0, 0.05) for q, qd, qdd in xs]
    sgt = SGTRegressor(min_samples_split=50, n_bins=16, bin_samples=200)
    run = MeanSquaredError()
    sgt_mse = MeanSquaredError()
    running = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = {"q": x[0], "qd": x[1], "qdd": x[2]}
        run.update(y, running)
        sgt_mse.update(y, sgt.predict_one(d))
        sgt.learn_one(d, y=y)
        running += (y - running) / (i + 1)
    print(f"SGT MSE = {sgt_mse.get():.4f}")
    print(f"run MSE = {run.get():.4f}")
    assert sgt_mse.get() < run.get()


def test_sgt_does_not_split_on_pure_noise():
    rng = np.random.default_rng(3)
    s = SGTRegressor(min_samples_split=50, grace_period=50, n_bins=16, bin_samples=200)
    for _ in range(2000):
        d = {"x0": float(rng.normal(0.0, 1.0))}
        y = float(rng.normal(0.0, 1.0))
        s.learn_one(d, y=y)
    print(f"n_nodes on pure noise: {s.n_nodes_}")
    assert s.n_nodes_ <= 3


def test_sgt_classifier_smoke():
    rng = np.random.default_rng(1)
    h = SGTClassifier(min_samples_split=30, n_bins=16, bin_samples=100)
    for _ in range(800):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        h.learn_one({"x0": float(rng.normal(c, 0.4))}, y=y)
    p0 = h.predict_one({"x0": 0.0})
    p1 = h.predict_one({"x0": 5.0})
    assert p0 == 0 and p1 == 1


def test_sgt_nan_counted():
    s = SGTRegressor(min_samples_split=10, bin_samples=5)
    s.learn_one({"x0": 0.0}, y=0.0)
    s.learn_one({"x0": float("nan")}, y=0.0)
    assert s.n_missing == 1


def test_check_estimator_sgt_regressor():
    check_estimator(SGTRegressor(min_samples_split=20, bin_samples=50))


def test_check_estimator_sgt_classifier():
    check_estimator(SGTClassifier(min_samples_split=20, bin_samples=50))


def test_merge_of_bins_equals_concatenated_moments():
    from dense_armor.utility.tree.sgt import _BinStats, _merge

    rng = np.random.default_rng(3)
    g = rng.normal(size=300)
    h = rng.random(300) + 0.5 * g
    bins = [_BinStats() for _ in range(4)]
    for i, (gi, hi) in enumerate(zip(g, h)):
        bins[i % 4 if i < 200 else 3].update(float(gi), float(hi))
    n, mg, m2g, mh, m2h, c = _merge(bins)
    assert n == 300
    assert math.isclose(mg, g.mean(), rel_tol=1e-12)
    assert math.isclose(mh, h.mean(), rel_tol=1e-12)
    assert math.isclose(m2g / (n - 1), g.var(ddof=1), rel_tol=1e-10)
    assert math.isclose(m2h / (n - 1), h.var(ddof=1), rel_tol=1e-10)
    assert math.isclose(c / (n - 1), np.cov(g, h)[0, 1], rel_tol=1e-10)
