"""Online boosting: the lambda accumulators, the vote, NaN handling."""

import numpy as np

from dense_armor.utility.ensemble.boosting import OnlineBoostingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _clf():
    return HoeffdingTreeClassifier(grace_period=20)


def test_learns_a_simple_split():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_clf, n_models=3, seed=0)
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        y = int(x > 2.0)
        _ = clf.learn_one({"x": x}, y)
    assert clf.predict_one({"x": 0.1}) == 0
    assert clf.predict_one({"x": 4.0}) == 1


def test_lambda_accumulators_are_finite():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_clf, n_models=3, seed=0)
    for _ in range(200):
        x = float(rng.normal(0.0, 1.0))
        y = int(x > 0.0)
        _ = clf.learn_one({"x": x}, y)
    for v in clf.lambda_sc_:
        assert np.isfinite(v)
    for v in clf.lambda_sw_:
        assert np.isfinite(v)


def test_nan_is_skipped_and_counted():
    clf = OnlineBoostingClassifier(_clf, n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    assert clf.n_missing == 1
    assert clf.n_seen_ == 1


def test_clone_preserves_constructor():
    clf = OnlineBoostingClassifier(_clf, n_models=4, seed=1)
    c = clf.clone()
    assert c.n_models == 4
    assert c.seed == 1
    assert c.get_params() == clf.get_params()
