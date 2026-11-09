"""Unit tests for dense_armor/utility/calibration.py (Online Platt scaling)."""

import doctest
import math
import random

import pytest

from dense_armor.utility.evaluation.metrics import LogLoss  # noqa: E402

import dense_armor.utility.online.learn.calibration as calibration  # noqa: E402
from dense_armor.roles import Classifier  # noqa: E402
from dense_armor.utility.online.learn.calibration import OnlinePlattScaling  # noqa: E402
from dense_armor.utility.online.learn.online_classifiers import OnlineGaussianNB  # noqa: E402


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
    raw, ops = LogLoss(), LogLoss()
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


class Late(Classifier):
    def __init__(self):
        self.seen = []

    def learn_one(self, x, y):
        self.seen.append(y)

    def predict_proba_one(self, x):
        if not self.seen:
            return {}
        if all(not v for v in self.seen):
            return {False: 1.0}
        return {False: 0.3, True: 0.7}


def test_no_opinion_leaves_calibration_untouched():
    cal = OnlinePlattScaling(Late())
    cal.learn_one({"x": 1.0}, False)
    assert cal.classifier.seen == [False]
    assert cal._theta.tolist() == [1.0, 0.0]
    assert cal.predict_proba_one({"x": 1.0})[True] < 0.05


def test_online_platt_scaling_handles_nan():
    class NaNToxicBase:
        def predict_proba_one(self, x):
            return {True: float("nan"), False: float("nan")}

        def learn_one(self, x, y):
            pass

    cal = OnlinePlattScaling(NaNToxicBase())
    res = cal.predict_proba_one({"x": 1.0})

    assert not any(isinstance(v, float) and math.isnan(v) for v in res.values())
    assert not any(isinstance(v, float) and math.isinf(v) for v in res.values())
    assert res[True] == 0.5
    assert res[False] == 0.5

def test_wrapped_hoeffding_tree_with_no_opinion():
    """River PR #1988: a real tree returns {} until it has seen a class.
    The wrapper must not invent a label, and must not move _theta while
    there is nothing to calibrate. Mirrors river's test_wrapped_model_with_no_opinion
    but with the real primitive, not a fake classifier.
    """
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier

    cal = OnlinePlattScaling(HoeffdingTreeClassifier())
    x = {"a": 1.0, "b": 2.0}

    # before anything is learned: no opinion in, no opinion out
    assert cal.predict_proba_one(x) == {}
    assert cal.predict_one(x) is None

    # the sample reaches the wrapped tree, and _theta is untouched
    cal.learn_one(x, True)
    assert cal._theta.tolist() == [1.0, 0.0]
    assert float(cal._A[0, 0]) == pytest.approx(100.0)
    assert float(cal._A[1, 1]) == pytest.approx(100.0)

    # once the tree has an opinion, calibration resumes
    for _ in range(5):
        cal.learn_one(x, True)
        cal.learn_one({"a": -1.0, "b": -2.0}, False)
    proba = cal.predict_proba_one(x)
    assert set(proba) == {False, True}
    assert abs(sum(proba.values()) - 1.0) < 1e-9
    # _theta has moved from the identity
    assert not (cal._theta[0] == 1.0 and cal._theta[1] == 0.0)


def test_score_is_sigmoid_of_theta_dot_u():
    """The returned probability is exactly sigmoid(_theta @ u), where
    u = [logit(clip(p)), 1] and p is the base's probability of True.
    This pins the formula, not just the behaviour.
    """
    class Known(Classifier):
        def __init__(self, p):
            self.p = p
        def learn_one(self, x, y):
            pass
        def predict_proba_one(self, x):
            return {False: 1.0 - self.p, True: self.p}

    for base_p in (0.2, 0.5, 0.8, 0.95, 0.01):
        cal = OnlinePlattScaling(Known(base_p))
        # force a non-identity theta so the formula is not trivially the identity
        cal._theta = __import__("numpy").array([0.7, -0.3])
        # base p clipped to [clip, 1-clip]
        p_clip = min(max(base_p, cal.clip), 1.0 - cal.clip)
        logit_p = math.log(p_clip / (1.0 - p_clip))
        expected = 1.0 / (1.0 + math.exp(-(cal._theta[0] * logit_p + cal._theta[1] * 1.0)))
        got = cal.predict_proba_one({})[True]
        assert got == pytest.approx(expected, abs=1e-12)


def test_learn_one_does_not_use_current_sample_for_scoring():
    """The score fed to the sigmoid is the base's prediction BEFORE it learns.
    So a single learn_one on a fresh tree leaves _theta at the identity, even
    though the tree itself has moved. This is the out-of-sample flavour from
    River's docstring line 21-22.
    """
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier

    cal = OnlinePlattScaling(HoeffdingTreeClassifier())
    x = {"a": 1.0, "b": 2.0}
    cal.learn_one(x, True)
    # the tree has learned (opinion now exists), but _theta did not move
    assert cal.classifier.predict_proba_one(x) != {}
    assert cal._theta.tolist() == [1.0, 0.0]
