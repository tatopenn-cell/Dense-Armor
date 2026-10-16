"""Branch coverage for the ensemble classes and the shared helpers."""

import pytest

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.ensemble import (
    ExponentiallyWeightedAverageRegressor,
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.ensemble._common import as_dict, finite, new_member
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _NoProba(Classifier):
    """A classifier whose ``predict_proba_one`` returns an empty dict."""

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 0

    def predict_proba_one(self, x, t=None):
        return {}


def test_as_dict_fallback_on_unconvertible():
    class Weird:
        def __iter__(self):
            raise TypeError("no")

        def keys(self):
            raise ValueError("no")

    assert as_dict(Weird()) == {}
    assert as_dict({}) == {}
    assert as_dict({"a": 1}) == {"a": 1}


def test_finite_on_string_and_nan():
    assert finite({"a": 1.0})
    assert not finite({"a": None})
    assert not finite({"a": "hello"})
    assert not finite({"a": float("nan")})
    assert not finite({"a": float("inf")})


def test_new_member_from_instance_without_clone_uses_deepcopy():
    class NoClone(Regressor):
        def __init__(self):
            self.v = 0.0

        def learn_one(self, x, y, t=None):
            self.v += 1.0
            return self

        def predict_one(self, x, t=None, return_std=False):
            return self.v

    base = NoClone()
    m = new_member(base)
    assert m is not base
    assert m.v == base.v == 0.0


def test_bagging_rejects_zero_models():
    with pytest.raises(ValueError):
        OnlineBaggingClassifier(_tree, n_models=0)


def test_boosting_rejects_zero_models():
    with pytest.raises(ValueError):
        OnlineBoostingClassifier(_tree, n_models=0)


def test_ewa_rejects_bad_eta():
    with pytest.raises(ValueError):
        ExponentiallyWeightedAverageRegressor([_NoProba], eta=0.0)
    with pytest.raises(ValueError):
        ExponentiallyWeightedAverageRegressor([], eta=1.0)


def test_voting_rejects_bad_mode():
    with pytest.raises(ValueError):
        VotingClassifier([_tree], mode="median")
    with pytest.raises(ValueError):
        VotingClassifier([_tree], weights=[1.0, 2.0])


def test_stacking_rejects_empty_models():
    with pytest.raises(ValueError):
        StackingClassifier([], _tree)


def test_bagging_predict_before_learning_returns_none():
    clf = OnlineBaggingClassifier(_tree, n_models=2, seed=0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_boosting_predict_before_learning_returns_none():
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_voting_hard_with_member_without_proba():
    v = VotingClassifier([_NoProba, _NoProba], mode="hard")
    for _ in range(5):
        _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_one({"x": 0.0}) == 0


def test_voting_soft_skips_members_without_proba():
    v = VotingClassifier([_NoProba, _NoProba], mode="soft")
    for _ in range(5):
        _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_proba_one({"x": 0.0}) == {}


def test_ewa_predict_before_learning_returns_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_NoProba])
    assert ewa.predict_one({"x": 0.0}) == 0.0


def test_ewa_bound_before_learning_is_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_NoProba])
    assert ewa.bound() == 0.0


def test_stacking_predict_before_learning_returns_none():
    s = StackingClassifier([_tree, _tree], _tree)
    assert s.predict_one({"x": 0.0}) is None
    assert s.predict_proba_one({"x": 0.0}) == {}
