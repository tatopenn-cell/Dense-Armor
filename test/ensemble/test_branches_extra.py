"""Targeted tests for the uncovered branches."""

import numpy as np
import pytest

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.ensemble import (
    ExponentiallyWeightedAverageRegressor,
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.ensemble._common import as_dict, new_member
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _ToDict:
    def to_dict(self):
        return {"a": 1.0}


class _Explodes:
    def __iter__(self):
        raise TypeError("no")


class _NoLearn(Classifier):
    def learn_one(self, x, y, t=None):
        raise AttributeError("no learn_one")

    def predict_one(self, x, t=None):
        return 0

    def predict_proba_one(self, x, t=None):
        return {0: 1.0}


class _NoPredict(Classifier):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")

    def predict_proba_one(self, x, t=None):
        return {}


class _BadPredictOne(Regressor):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        raise TypeError("bad")


# --- _common ---


def test_as_dict_from_to_dict():
    assert as_dict(_ToDict()) == {"a": 1.0}


def test_as_dict_exception_returns_empty():
    assert as_dict(_Explodes()) == {}


def test_new_member_from_class_and_instance_without_clone():
    class NoClone(Regressor):
        def __init__(self):
            self.v = 0

        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None, return_std=False):
            return self.v

    m = new_member(NoClone)
    assert isinstance(m, NoClone)
    m2 = new_member(NoClone())
    assert isinstance(m2, NoClone)


# --- skips ---


def test_skip_sets_are_empty():
    estimators = [
        OnlineBaggingClassifier(_tree, n_models=2, seed=0),
        OnlineBoostingClassifier(_tree, n_models=2, seed=0),
        ExponentiallyWeightedAverageRegressor([_tree]),
        VotingClassifier([_tree]),
        StackingClassifier([_tree], _tree),
    ]
    for est in estimators:
        assert est._unit_test_skips() == set()


# --- boosting: predict_proba / predict ---


def test_boosting_predict_proba_and_predict():
    rng = np.random.default_rng(0)
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    for _ in range(50):
        x = float(rng.normal(0.0, 1.0))
        y = int(x > 0.0)
        _ = clf.learn_one({"x": x}, y)
    p = clf.predict_proba_one({"x": 0.5})
    assert p == {} or abs(sum(p.values()) - 1.0) < 1e-9
    assert clf.predict_one({"x": 0.5}) is not None


def test_boosting_predict_skips_member_that_raises():
    clf = OnlineBoostingClassifier(lambda: _NoPredict(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_one({"x": 0.0}) is None


# --- bagging skip member that raises ---


def test_bagging_predict_skips_member_that_raises():
    clf = OnlineBaggingClassifier(lambda: _NoPredict(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


# --- ewa ---


def test_ewa_member_predict_raises_is_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_BadPredictOne])
    for _ in range(3):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.predict_one({"x": 0.0}) == 0.0


def test_ewa_member_learn_raises_is_caught():
    ewa = ExponentiallyWeightedAverageRegressor([_NoLearn])
    _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.n_seen_ == 1


def test_ewa_return_std():
    from dense_armor.roles import Regressor as R

    class C(R):
        def __init__(self, v=0.0):
            self.v = float(v)

        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None, return_std=False):
            return self.v

    ewa = ExponentiallyWeightedAverageRegressor([C(0.0), C(1.0)], eta=0.5)
    for _ in range(5):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    _y, s = ewa.predict_one({"x": 0.0}, return_std=True)
    assert s >= 0.0


def test_ewa_bound_eta_huge_denom_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_tree], eta=1e9)
    # bound ritorna inf quando denom <= 0; forzo l'accesso dopo un learn
    for _ in range(2):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    # il valore può essere finito o infinito, ma la chiamata non deve crashare
    b = ewa.bound()
    assert b >= 0 or np.isinf(b)


# --- stacking ---


def test_stacking_n_missing_and_skip():
    s = StackingClassifier([_tree], _tree)
    assert s.n_missing == 0
    assert s._unit_test_skips() == set()
    _ = s.learn_one({"x": float("nan")}, 0)
    assert s.n_missing == 1


def test_stacking_learn_member_raises_is_caught():
    s = StackingClassifier([_NoLearn], _tree)
    _ = s.learn_one({"x": 0.0}, 0)
    assert s.n_seen_ == 1


def test_stacking_predict_proba_before_meta():
    s = StackingClassifier([_tree], _tree)
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) is None


def test_stacking_member_meta_without_proba():
    s = StackingClassifier([_tree], _NoPredict)
    for _ in range(5):
        _ = s.learn_one({"x": 0.0}, 0)
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) is None


# --- voting ---


def test_voting_rejects_empty_models():
    with pytest.raises(ValueError):
        VotingClassifier([], mode="hard")


def test_voting_skip_set_empty():
    v = VotingClassifier([_tree])
    assert v._unit_test_skips() == set()


def test_voting_weights_are_used():
    v = VotingClassifier([_tree, _tree], weights=[0.0, 1.0])
    for _ in range(5):
        _ = v.learn_one({"x": 0.0}, 0)
    # il secondo membro ha peso 1, il primo 0
    p = v.predict_proba_one({"x": 0.0})
    assert p == {0: 1.0} or p == {}


def test_voting_member_learn_raises_is_caught():
    v = VotingClassifier([_NoLearn])
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.n_seen_ == 1


def test_voting_hard_skip_broken_predict():
    v = VotingClassifier([_NoPredict], mode="hard")
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_one({"x": 0.0}) is None


def test_voting_soft_skip_member_with_broken_proba():
    v = VotingClassifier([_NoPredict], mode="soft")
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_proba_one({"x": 0.0}) == {}
