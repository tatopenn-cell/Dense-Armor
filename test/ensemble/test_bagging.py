"""Online bagging: the Poisson(1) resample and the vote."""

import numpy as np

from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _clf():
    return HoeffdingTreeClassifier(grace_period=20)


def test_poisson_draw_mean_and_variance():
    """``k ~ Poisson(1)`` has mean and variance one."""
    rng = np.random.default_rng(0)
    k = rng.poisson(1.0, 100000)
    assert abs(k.mean() - 1.0) < 0.02
    assert abs(k.var() - 1.0) < 0.03


def test_learns_a_simple_split():
    rng = np.random.default_rng(0)
    clf = OnlineBaggingClassifier(_clf, n_models=3, seed=0)
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        y = int(x > 2.0)
        _ = clf.learn_one({"x": x}, y)
    assert clf.predict_one({"x": 0.1}) == 0
    assert clf.predict_one({"x": 4.0}) == 1


def test_nan_sample_is_skipped_and_counted():
    clf = OnlineBaggingClassifier(_clf, n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    _ = clf.learn_one({"x": float("inf")}, 1)
    _ = clf.learn_one({}, 0)
    _ = clf.learn_one({"x": 1.0}, None)
    assert clf.n_missing == 4
    assert clf.n_seen_ == 1


def test_clone_preserves_constructor():
    clf = OnlineBaggingClassifier(_clf, n_models=5, seed=42)
    c = clf.clone()
    assert type(c) is OnlineBaggingClassifier
    assert c.n_models == 5
    assert c.seed == 42
    assert c.get_params() == clf.get_params()


def test_no_nan_in_weights_after_a_nan_sample():
    clf = OnlineBaggingClassifier(_clf, n_models=2, seed=0)
    for _ in range(20):
        _ = clf.learn_one({"x": 0.5}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    for _ in range(20):
        _ = clf.learn_one({"x": 0.5}, 0)
    p = clf.predict_proba_one({"x": 0.5})
    for v in p.values():
        assert np.isfinite(v)
