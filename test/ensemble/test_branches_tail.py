"""Cover the last branches: stacking before learn, voting before learn,
new_member with an object that has no clone."""

import numpy as np

from dense_armor.utility.ensemble import (
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.ensemble._common import new_member
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _BareModel:
    """Not a Root subclass, no clone, no learn_one-needed factory."""

    def __init__(self):
        self.calls = 0

    def learn_one(self, x, y, t=None):
        self.calls += 1
        return self

    def predict_one(self, x, t=None):
        return 0


def test_new_member_deepcopy_bare_object():
    base = _BareModel()
    m = new_member(base)
    assert m is not base
    assert m.calls == 0
    m.learn_one({}, 0)
    assert base.calls == 0


def test_stacking_predict_before_learn():
    s = StackingClassifier([_tree], _tree)
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) is None


def test_stacking_meta_without_proba():
    class _OnlyPredict:
        def __init__(self):
            pass

        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None):
            return 0

        def clone(self):
            return _OnlyPredict()

    s = StackingClassifier([_tree], _OnlyPredict)
    for _ in range(3):
        _ = s.learn_one({"x": 0.0}, 0)
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) == 0


def test_voting_predict_before_learn():
    v = VotingClassifier([_tree], mode="hard")
    assert v.predict_proba_one({"x": 0.0}) == {}


def test_ewa_bound_without_inf_guard():
    from dense_armor.roles import Regressor
    from dense_armor.utility.ensemble import ExponentiallyWeightedAverageRegressor

    class C(Regressor):
        def __init__(self):
            self.v = 0.0

        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None, return_std=False):
            return self.v

    ewa = ExponentiallyWeightedAverageRegressor([C], eta=0.5)
    for _ in range(3):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    b = ewa.bound()
    assert np.isfinite(b) and b >= 0
