"""Online boosting: lambda accumulators, vote, NaN handling."""

import numpy as np

from dense_armor.utility.ensemble import OnlineBoostingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _NoPredict:
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")


def test_learns_a_simple_split():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_tree, n_models=3, seed=0)
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        y = int(x > 2.0)
        _ = clf.learn_one({"x": x}, y)
    assert clf.predict_one({"x": 0.1}) == 0
    assert clf.predict_one({"x": 4.0}) == 1


def test_lambda_accumulators_are_finite():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_tree, n_models=3, seed=0)
    for _ in range(200):
        x = float(rng.normal(0.0, 1.0))
        y = int(x > 0.0)
        _ = clf.learn_one({"x": x}, y)
    assert all(np.isfinite(v) for v in clf.lambda_sc_)
    assert all(np.isfinite(v) for v in clf.lambda_sw_)


def test_nan_is_skipped_and_counted():
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    assert clf.n_missing == 1
    assert clf.n_seen_ == 1


def test_clone_preserves_constructor():
    clf = OnlineBoostingClassifier(_tree, n_models=4, seed=1)
    c = clf.clone()
    assert c.n_models == 4
    assert c.seed == 1
    assert c.get_params() == clf.get_params()


def test_beta_is_one_before_any_accumulator():
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    clf.members_ = [clf._new_member() for _ in range(2)]
    clf.lambda_sc_ = [0.0, 0.0]
    clf.lambda_sw_ = [0.0, 0.0]
    assert clf._beta(0) == 1.0


def test_predict_before_learning():
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_member_without_predict_returns_none():
    clf = OnlineBoostingClassifier(lambda: _NoPredict(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_predict_proba_sums_to_one_after_learning():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    for _ in range(200):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        _ = clf.learn_one({"x": x}, int(x > 2.0))
    p = clf.predict_proba_one({"x": 0.5})
    if p:
        assert abs(sum(p.values()) - 1.0) < 1e-9


def test_rejects_zero_models():
    try:
        OnlineBoostingClassifier(_tree, n_models=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_skip_set_is_empty():
    assert (
        OnlineBoostingClassifier(_tree, n_models=2, seed=0)._unit_test_skips() == set()
    )
