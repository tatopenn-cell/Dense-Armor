"""Voting classifier: hard vote and soft vote."""


from dense_armor.roles import Classifier
from dense_armor.utility.ensemble import VotingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
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


class _NoPredict(Classifier):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")

    def predict_proba_one(self, x, t=None):
        raise AttributeError("no proba")


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


def test_weights_are_used():
    v = VotingClassifier([_Fixed(0), _Fixed(1)], weights=[0.0, 1.0])
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


def test_rejects_bad_mode_and_empty_models():
    try:
        VotingClassifier([_tree], mode="median")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        VotingClassifier([], mode="hard")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rejects_bad_weights_length():
    try:
        VotingClassifier([_tree], weights=[1.0, 2.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_member_without_predict():
    v = VotingClassifier([_NoPredict], mode="hard")
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.predict_one({"x": 0.0}) is None
    v2 = VotingClassifier([_NoPredict], mode="soft")
    _ = v2.learn_one({"x": 0.0}, 0)
    assert v2.predict_proba_one({"x": 0.0}) == {}


def test_predict_before_learning():
    v = VotingClassifier([_tree], mode="hard")
    assert v.predict_proba_one({"x": 0.0}) == {}


def test_member_learn_raises_is_caught():
    class _BadLearn(Classifier):
        def learn_one(self, x, y, t=None):
            raise AttributeError("no learn")

        def predict_one(self, x, t=None):
            return 0

    v = VotingClassifier([_BadLearn])
    _ = v.learn_one({"x": 0.0}, 0)
    assert v.n_seen_ == 1


def test_skip_set_is_empty():
    assert VotingClassifier([_tree])._unit_test_skips() == set()
