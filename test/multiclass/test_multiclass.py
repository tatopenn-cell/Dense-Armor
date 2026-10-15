"""Tests for the binary-to-multiclass strategies."""

import doctest

import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from dense_armor.utility.multiclass import multiclass as mc_mod
from dense_armor.utility.optim import SGD


def factory():
    return LogisticRegression(optimizer=SGD(lr=0.5), intercept=True)


def test_doctests():
    r = doctest.testmod(mc_mod)
    assert r.failed == 0


def _three_class_stream():
    return [
        ({"x": 0.0, "y": 0.0}, "A"),
        ({"x": 0.1, "y": 0.1}, "A"),
        ({"x": 1.0, "y": 1.0}, "B"),
        ({"x": 1.1, "y": 0.9}, "B"),
        ({"x": -1.0, "y": 1.0}, "C"),
        ({"x": -0.9, "y": 1.1}, "C"),
    ]


def test_one_vs_rest_three_class():
    clf = OneVsRestClassifier(binary_factory=factory)
    for _ in range(20):
        for x, y in _three_class_stream():
            clf.learn_one(x, y)
    assert clf.predict_one({"x": 0.0, "y": 0.0}) == "A"
    assert clf.predict_one({"x": 1.0, "y": 1.0}) == "B"
    assert clf.predict_one({"x": -1.0, "y": 1.0}) == "C"


def test_one_vs_one_three_class():
    clf = OneVsOneClassifier(binary_factory=factory)
    for _ in range(20):
        for x, y in _three_class_stream():
            clf.learn_one(x, y)
    assert clf.predict_one({"x": 0.0, "y": 0.0}) == "A"
    assert clf.predict_one({"x": 1.0, "y": 1.0}) == "B"
    assert clf.predict_one({"x": -1.0, "y": 1.0}) == "C"


def test_output_code_three_class():
    clf = OutputCodeClassifier(binary_factory=factory, code_size=8)
    for _ in range(20):
        for x, y in _three_class_stream():
            clf.learn_one(x, y)
    assert clf.predict_one({"x": 0.0, "y": 0.0}) == "A"


@pytest.mark.parametrize(
    "Cls", [OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier]
)
def test_nan_skip(Cls):
    clf = Cls(binary_factory=factory)
    for x, y in _three_class_stream():
        clf.learn_one(x, y)
    n0 = clf.n_missing
    clf.learn_one({"x": float("nan"), "y": 0.0}, "A")
    clf.learn_one({"y": 0.0}, "A")
    clf.learn_one({}, "A")
    assert clf.n_missing == n0 + 3


@pytest.mark.parametrize(
    "Cls", [OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier]
)
def test_check_estimator_passes(Cls):
    check_estimator(Cls(binary_factory=factory))


@pytest.mark.parametrize(
    "Cls", [OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier]
)
def test_predict_empty_before_learn(Cls):
    clf = Cls(binary_factory=factory)
    assert clf.predict_proba_one({"x": 1.0}) == {}
    assert clf.predict_one({"x": 1.0}) is None


def test_bad_binary_factory():
    with pytest.raises(TypeError, match="callable"):
        OneVsRestClassifier(binary_factory="not callable")


def test_output_code_bad_code_size():
    with pytest.raises(ValueError, match="code_size"):
        OutputCodeClassifier(binary_factory=factory, code_size=1)


def _stub(answer):
    from dense_armor.roles import Classifier

    class Stub(Classifier):
        def __init__(self, answer=None):
            self.answer = answer

        def learn_one(self, x, y, t=None):
            return self

        def predict_proba_one(self, x, t=None):
            return dict(self.answer or {})

    return lambda: Stub(answer)


def test_unit_test_params_build_binary_classifiers():
    for cls in (OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier):
        for params in cls._unit_test_params():
            assert params["binary_factory"]() is not None


def test_one_vs_rest_is_uniform_when_no_binary_classifier_says_yes():
    clf = OneVsRestClassifier(binary_factory=_stub({0: 1.0, 1: 0.0}))
    for c in ("a", "b", "c"):
        clf.learn_one({"x": 1.0}, c)
    p = clf.predict_proba_one({"x": 1.0})
    assert all(v == pytest.approx(1.0 / 3.0) for v in p.values())


def test_one_vs_one_skips_a_pair_that_gives_no_answer():
    clf = OneVsOneClassifier(binary_factory=_stub({}))
    for c in ("a", "b", "c"):
        clf.learn_one({"x": 1.0}, c)
    assert set(clf.predict_proba_one({"x": 1.0})) == {"a", "b", "c"}


def test_unusable_query_gives_an_empty_answer():
    for cls in (OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier):
        clf = cls(binary_factory=factory)
        for _ in range(5):
            clf.learn_one({"x": 0.0}, "a")
            clf.learn_one({"x": 1.0}, "b")
        assert clf.predict_proba_one({"x": "word"}) == {}
        assert clf.predict_proba_one({"y": 1.0}) == {}
        assert clf.predict_proba_one({"x": float("nan")}) == {}
