"""Voting classifier: hard vote and soft vote."""

from dense_armor.roles import Classifier
from dense_armor.utility.ensemble.voting import VotingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree_factory():
    return HoeffdingTreeClassifier(grace_period=20)


class _Fixed(Classifier):
    def __init__(self, value=0, proba=None):
        self.value = value
        self.proba = proba or {value: 1.0}

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return self.value

    def predict_proba_one(self, x, t=None):
        return dict(self.proba)


def test_hard_vote_takes_the_majority():
    v = VotingClassifier([_Fixed(0), _Fixed(1), _Fixed(1)], mode="hard")
    assert v.predict_one({"x": 0.0}) == 1
    p = v.predict_proba_one({"x": 0.0})
    assert abs(p[1] - 2 / 3) < 1e-9
    assert abs(p[0] - 1 / 3) < 1e-9


def test_soft_vote_averages_probabilities():
    a = _Fixed(0, proba={0: 0.8, 1: 0.2})
    b = _Fixed(1, proba={0: 0.1, 1: 0.9})
    v = VotingClassifier([a, b], mode="soft")
    p = v.predict_proba_one({"x": 0.0})
    assert abs(p[0] - 0.45) < 1e-9
    assert abs(p[1] - 0.55) < 1e-9
    assert v.predict_one({"x": 0.0}) == 1


def test_nan_counted():
    v = VotingClassifier([_Fixed(0)], mode="hard")
    _ = v.learn_one({"x": float("nan")}, 0)
    assert v.n_missing == 1


def test_clone_preserves_constructor():
    v = VotingClassifier([HoeffdingTreeClassifier()], mode="soft")
    c = v.clone()
    assert type(c) is VotingClassifier
    assert c.mode == "soft"
    assert c.weights == v.weights
    assert len(c.models) == len(v.models)
