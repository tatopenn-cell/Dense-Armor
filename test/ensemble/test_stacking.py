"""Stacking: the meta-model must not see the label before predicting."""

from dense_armor.roles import Classifier
from dense_armor.utility.ensemble.stacking import StackingClassifier


class _Spy(Classifier):
    """Records the order of learn/predict calls to detect leakage."""

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


def test_no_leakage_meta_learns_before_members():
    """The meta-model sees the members' predictions before them."""
    s = StackingClassifier([_Spy], _Spy)
    _ = s.learn_one({"x": 1.0}, 0)
    base_kinds = [c[0] for c in s.members_[0].calls]
    meta_kinds = [c[0] for c in s.meta_.calls]
    assert "learn" in meta_kinds
    # il base vede predict (fatto dal meta) prima di learn
    assert base_kinds == ["predict", "learn"]


def test_meta_sees_members_predictions():
    s = StackingClassifier([_Spy], _Spy)
    _ = s.learn_one({"x": 1.0}, 1)
    learn_call = next(c for c in s.meta_.calls if c[0] == "learn")
    assert "m0" in learn_call[1]


def test_clone():
    s = StackingClassifier([_Spy], _Spy)
    c = s.clone()
    assert type(c) is StackingClassifier
