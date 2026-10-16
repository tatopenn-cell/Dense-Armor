"""Cover the last uncovered branches: as_dict paths, member failures,
degenerate beta, EWA eta huge, stacking meta without proba."""

import numpy as np

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
    def __init__(self):
        self.d = {"a": 1.0}

    def to_dict(self):
        return self.d


class _NotDictable:
    def __iter__(self):
        raise TypeError("no")

    def keys(self):
        raise ValueError("no")


class _NoPredictClassifier(Classifier):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")

    def predict_proba_one(self, x, t=None):
        raise AttributeError("no proba")


class _OnlyPredictClassifier(Classifier):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 0

    def predict_proba_one(self, x, t=None):
        return {}


class _NoCloneNoLearnReg(Regressor):
    """Without clone, has learn_one, so new_member must deepcopy."""

    def __init__(self):
        self.calls = 0

    def learn_one(self, x, y, t=None):
        self.calls += 1
        return self

    def predict_one(self, x, t=None, return_std=False):
        return float(self.calls)


# --- as_dict entrambe le diramazioni ---


def test_as_dict_paths_covered():
    assert as_dict(_ToDict()) == {"a": 1.0}
    assert as_dict(_NotDictable()) == {}
    assert as_dict({"a": 1}) == {"a": 1}


# --- new_member deepcopy ---


def test_new_member_deepcopy_when_no_clone():
    base = _NoCloneNoLearnReg()
    m = new_member(base)
    assert m is not base
    assert m.calls == 0
    m.learn_one({}, 0.0)
    assert base.calls == 0


# --- bagging member without predict ---


def test_bagging_member_without_predict_counts_missing():
    clf = OnlineBaggingClassifier(lambda: _NoPredictClassifier(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


# --- boosting beta degenerate + predict_proba with broken member ---


def test_boosting_beta_degenerate():
    clf = OnlineBoostingClassifier(_tree, n_models=2, seed=0)
    # senza ancora aver appreso, i lambda sono zero: beta deve ritornare 1.0
    clf.members_ = [clf._new_member() for _ in range(2)]
    clf.lambda_sc_ = [0.0, 0.0]
    clf.lambda_sw_ = [0.0, 0.0]
    assert clf._beta(0) == 1.0


def test_boosting_predict_proba_with_broken_member():
    clf = OnlineBoostingClassifier(lambda: _NoPredictClassifier(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_proba_one({"x": 0.0}) == {}
    assert clf.predict_one({"x": 0.0}) is None


# --- ewa eta grande e z<=0 ---


def test_ewa_eta_huge_bound_inf():
    ewa = ExponentiallyWeightedAverageRegressor([_NoCloneNoLearnReg], eta=1e12)
    for _ in range(3):
        _ = ewa.learn_one({"x": 0.0}, 0.0)
    b = ewa.bound()
    assert np.isinf(b) or b >= 0


# --- stacking paths ---


def test_stacking_member_without_predict():
    s = StackingClassifier([lambda: _NoPredictClassifier()], lambda: _tree())
    _ = s.learn_one({"x": 0.0}, 0)
    # il meta esiste, ma _meta_input è vuoto perché il membro fallisce
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) is None


def test_stacking_meta_without_proba():
    s = StackingClassifier([lambda: _tree()], lambda: _OnlyPredictClassifier())
    for _ in range(3):
        _ = s.learn_one({"x": 0.0}, 0)
    # meta_ ha predict_one ma non predict_proba_one: la chiamata deve
    # cadere nel ramo AttributeError
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) in (0, None)


def test_stacking_predict_one_before_learn():
    s = StackingClassifier([_tree], _tree)
    assert s.predict_one({"x": 0.0}) is None
    assert s.predict_proba_one({"x": 0.0}) == {}


# --- voting paths ---


def test_voting_hard_zero_weight_sum():
    v = VotingClassifier([_tree], mode="hard", weights=[0.0])
    for _ in range(3):
        _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_proba_one({"x": 0.0}) == {}


def test_voting_member_without_predict():
    v = VotingClassifier([lambda: _NoPredictClassifier()], mode="hard")
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_one({"x": 0.0}) is None
    v2 = VotingClassifier([lambda: _NoPredictClassifier()], mode="soft")
    _ = v2.learn_one({"x": 0.0}, 0)
    assert v2.predict_proba_one({"x": 0.0}) == {}
