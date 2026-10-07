"""Unit tests for dense_armor/utility/calibration.py (Online Platt scaling)."""
import doctest
import importlib
import sys

import math
import random

import pytest

pytest.importorskip("river")

from river import metrics  # noqa: E402
from dense_armor.base import Classifier  # noqa: E402

import dense_armor.utility.learn.calibration as calibration  # noqa: E402
from dense_armor.utility.learn.calibration import OnlinePlattScaling  # noqa: E402
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB  # noqa: E402


class Fixed(Classifier):
    def __init__(self, p: float = 0.7):
        self.p = p

    def learn_one(self, x, y):
        pass

    def predict_proba_one(self, x):
        return {False: 1.0 - self.p, True: self.p}


class Overconfident(Classifier):
    def __init__(self, power: float = 4.0):
        self.power = power

    def learn_one(self, x, y):
        pass

    def predict_proba_one(self, x):
        z = self.power * x["s"]
        p = 1.0 / (1.0 + math.exp(-z))
        return {False: 1.0 - p, True: p}


def stream(n, seed=42):
    rng = random.Random(seed)
    for _ in range(n):
        s = rng.gauss(0, 1)
        yield {"s": s}, rng.random() < 1.0 / (1.0 + math.exp(-s))


def test_first_prediction_is_base_probability():
    m = OnlinePlattScaling(Fixed(0.7))
    assert m.predict_proba_one({})[True] == pytest.approx(0.7)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_extreme_base_probabilities_are_clipped(p):
    m = OnlinePlattScaling(Fixed(p))
    q = m.predict_proba_one({})[True]
    assert 0.01 - 1e-12 <= q <= 0.99 + 1e-12
    for y in [True, False, True]:
        m.learn_one({}, y)
    q = m.predict_proba_one({})[True]
    assert math.isfinite(q) and 0.0 < q < 1.0


def test_probabilities_sum_to_one():
    m = OnlinePlattScaling(Overconfident())
    for x, y in stream(200):
        proba = m.predict_proba_one(x)
        assert proba[True] + proba[False] == pytest.approx(1.0)
        m.learn_one(x, y)


def test_theta_stays_in_ball():
    m = OnlinePlattScaling(Fixed(0.99), radius=2.0)
    for _ in range(500):
        m.learn_one({}, False)
    assert float(m._theta @ m._theta) <= 2.0**2 + 1e-9


def test_int_labels_match_bool_labels():
    a = OnlinePlattScaling(Overconfident())
    b = OnlinePlattScaling(Overconfident())
    for x, y in stream(300):
        a.learn_one(x, y)
        b.learn_one(x, int(y))
    x = {"s": 0.3}
    assert a.predict_proba_one(x)[True] == pytest.approx(b.predict_proba_one(x)[True])


def test_calibrates_overconfident_model():
    raw, ops = metrics.LogLoss(), metrics.LogLoss()
    base_model = Overconfident()
    m = OnlinePlattScaling(Overconfident())
    for x, y in stream(5000):
        raw.update(y, base_model.predict_proba_one(x)[True])
        ops.update(y, m.predict_proba_one(x)[True])
        m.learn_one(x, y)
    assert ops.get() < raw.get()
    assert m._theta[0] == pytest.approx(0.25, abs=0.1)


def test_docstring_example():
    assert doctest.testmod(calibration).failed == 0



def test_projection_onto_small_ball():
    m = OnlinePlattScaling(Fixed(0.9), radius=0.5)
    for y in [False, True, False, False]:
        m.learn_one({}, y)
        assert float(m._theta @ m._theta) ** 0.5 <= 0.5 + 1e-9


def test_check_estimator():
    from dense_armor.checks import check_estimator

    check_estimator(OnlinePlattScaling(OnlineGaussianNB()))
