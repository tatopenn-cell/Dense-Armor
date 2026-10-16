"""Stacking: the meta-model must not see the label before predicting."""

from dense_armor.roles import Classifier
from dense_armor.utility.ensemble import StackingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _Spy(Classifier):
    def __init__(self):
        self.calls: list = []

    def learn_one(self, x, y, t=None):
        self.calls.append(("learn", dict(x), y))
        return self

    def predict_one(self, x, t=None):
        self.calls.append(("predict", dict(x)))
        return 0

    def predict_proba_one(self, x, t=None):
        return {0: 1.0}


class _NoPredict(Classifier):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")

    def predict_proba_one(self, x, t=None):
        raise AttributeError("no proba")


class _BadLearn(Classifier):
    def learn_one(self, x, y, t=None):
        raise AttributeError("no learn")

    def predict_one(self, x, t=None):
        return 0

    def predict_proba_one(self, x, t=None):
        return {0: 1.0}


def test_no_leakage_meta_learns_before_members():
    s = StackingClassifier([_Spy], _Spy)
    _ = s.learn_one({"x": 1.0}, 0)
    assert "learn" in [c[0] for c in s.meta_.calls]
    assert [c[0] for c in s.members_[0].calls] == ["predict", "learn"]


def test_meta_sees_members_predictions():
    s = StackingClassifier([_Spy], _Spy)
    _ = s.learn_one({"x": 1.0}, 1)
    learn_call = next(c for c in s.meta_.calls if c[0] == "learn")
    assert "m0" in learn_call[1]


def test_clone():
    s = StackingClassifier([_Spy], _Spy)
    assert type(s.clone()) is StackingClassifier


def test_rejects_empty_models():
    try:
        StackingClassifier([], _tree)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_nan_counted():
    s = StackingClassifier([_tree], _tree)
    _ = s.learn_one({"x": float("nan")}, 0)
    assert s.n_missing == 1


def test_predict_before_learning():
    s = StackingClassifier([_tree], _tree)
    assert s.predict_one({"x": 0.0}) is None
    assert s.predict_proba_one({"x": 0.0}) == {}


def test_member_without_predict():
    s = StackingClassifier([_NoPredict], _tree)
    _ = s.learn_one({"x": 0.0}, 0)
    assert s.predict_one({"x": 0.0}) is None
    assert s.predict_proba_one({"x": 0.0}) == {}


def test_meta_without_proba():
    class _OnlyPredict(Classifier):
        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None):
            return 0

    s = StackingClassifier([_tree], _OnlyPredict)
    for _ in range(3):
        _ = s.learn_one({"x": 0.0}, 0)
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) == 0


def test_member_learn_raises_is_caught():
    s = StackingClassifier([_BadLearn], _tree)
    _ = s.learn_one({"x": 0.0}, 0)
    assert s.n_seen_ == 1


def test_skip_set_is_empty():
    assert StackingClassifier([_tree], _tree)._unit_test_skips() == set()


def test_meta_predict_one_raises_not_implemented():
    class _NoPredict(Classifier):
        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None):
            raise NotImplementedError("no predict")

        def predict_proba_one(self, x, t=None):
            return {0: 1.0}

    s = StackingClassifier([_tree], _NoPredict)
    for _ in range(3):
        _ = s.learn_one({"x": 0.0}, 0)
    assert s.predict_one({"x": 0.0}) is None
