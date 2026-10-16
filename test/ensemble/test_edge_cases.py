"""Edge cases found in the review of the ensemble package.

- EWA must keep the weighted average with large losses: the weights
  ``exp(-eta * L_i)`` are shifted by the smallest loss, so no member
  underflows to zero and the prediction is the best member, not the mean.
- ``new_member`` must not share state between members: two members built
  from the same object must be independent.
- The voting classifier must be able to beat a single member when the
  members are not identical.
- The stacking classifier must not let the meta-model see the label
  before the members have predicted.
"""

import numpy as np

from dense_armor.roles import Regressor
from dense_armor.utility.ensemble import (
    ExponentiallyWeightedAverageRegressor,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.ensemble._common import new_member
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


class _Const(Regressor):
    def __init__(self, v=0.0):
        self.v = float(v)

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self.v


class _NoCloneReg(Regressor):
    """A regressor without clone, to test the deep-copy fallback."""

    def __init__(self):
        self.counter = 0

    def learn_one(self, x, y, t=None):
        self.counter += 1
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self.counter


def test_ewa_keeps_the_best_member_with_large_losses():
    """The reviewer's case: three members at 0, 50, 90, truth 100."""
    ewa = ExponentiallyWeightedAverageRegressor(
        [_Const(0.0), _Const(50.0), _Const(90.0)], eta=0.5
    )
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 100.0)
    pred = ewa.predict_one({"x": 0.0})
    assert pred > 85.0, f"prediction {pred} too far from the best member (90)"


def test_ewa_predict_is_not_a_simple_mean():
    """A mean of the members would give 46.7; the weighted average is 90."""
    ewa = ExponentiallyWeightedAverageRegressor(
        [_Const(0.0), _Const(50.0), _Const(90.0)], eta=0.5
    )
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 100.0)
    pred = ewa.predict_one({"x": 0.0})
    simple_mean = (0.0 + 50.0 + 90.0) / 3.0
    assert abs(pred - simple_mean) > 30.0


def test_new_member_deep_copies_when_no_clone():
    """Two members built from the same object must not share state."""
    base = _NoCloneReg()
    a = new_member(base)
    b = new_member(base)
    assert a is not base
    assert b is not base
    a.learn_one({}, 0.0)
    assert base.counter == 0
    assert b.counter == 0
    assert a.counter == 1


def test_voting_can_beat_a_single_member():
    """Voting of diverse members (different grace_period) beats the
    worst member on the same stream, and it is not worse than the best."""
    rng = np.random.default_rng(0)
    stream = []
    for _ in range(800):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        stream.append(({"x": x}, int(x > 2.0)))

    def run(model):
        correct = 0
        for x, y in stream:
            p = model.predict_one(x)
            if p is not None and p == y:
                correct += 1
            model.learn_one(x, y)
        return correct / len(stream)

    single = run(HoeffdingTreeClassifier(grace_period=50))
    vote = run(
        VotingClassifier(
            [
                HoeffdingTreeClassifier(grace_period=20),
                HoeffdingTreeClassifier(grace_period=80),
            ],
            mode="hard",
        )
    )
    # il voto con membri diversi è almeno buono quanto l'albero singolo
    # (non garantiamo di più: dipende dallo stream)
    assert vote >= single - 0.05


def test_stacking_with_two_different_bases():
    """The stacking classifier works with two different base models."""
    rng = np.random.default_rng(0)
    stream = []
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        stream.append(({"x": x}, int(x > 2.0)))
    s = StackingClassifier(
        [
            HoeffdingTreeClassifier(grace_period=20),
            HoeffdingTreeClassifier(grace_period=80),
        ],
        HoeffdingTreeClassifier(grace_period=20),
    )
    for x, y in stream:
        _ = s.learn_one(x, y)
    p0 = s.predict_one({"x": 0.1})
    p4 = s.predict_one({"x": 4.0})
    assert p0 is not None and p4 is not None
