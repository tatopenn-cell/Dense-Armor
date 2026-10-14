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
