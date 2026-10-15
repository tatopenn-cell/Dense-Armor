"""Tests for the naive Bayes classifiers in bayes/.

Hand computation on four documents, check_estimator on every class,
NaN and missing key skip, clone not overwritten by learning.
"""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    MultinomialNB,
)
from dense_armor.utility.bayes import bernoulli as bern_mod
from dense_armor.utility.bayes import categorical as cat_mod
from dense_armor.utility.bayes import complement as comp_mod
from dense_armor.utility.bayes import multinomial as multi_mod

FIXTURE_DOCS = [
    ({"a": 2, "b": 1, "c": 0}, "X"),
    ({"a": 1, "b": 0, "c": 2}, "X"),
    ({"a": 0, "b": 2, "c": 1}, "Y"),
    ({"a": 0, "b": 3, "c": 1}, "Y"),
]


def test_doctests():
    for mod in (multi_mod, bern_mod, comp_mod, cat_mod):
        r = doctest.testmod(mod)
        assert r.failed == 0


# ------------------------------------------------------------------
# Multinomial: hand computation on four documents
# ------------------------------------------------------------------


def test_multinomial_hand_computation():
    """With alpha=1 the class-conditional probabilities are exact.

    Class X counts: a: 3, b: 1, c: 2, total 6.
    Class Y counts: a: 0, b: 5, c: 2, total 7.
    V = 3.

    theta_X = (3+1, 1+1, 2+1) / (6+3) = (4, 2, 3) / 9
    theta_Y = (0+1, 5+1, 2+1) / (7+3) = (1, 6, 3) / 10

    Priors: 1/2 each.

    log P(X | a=5, b=0, c=0) =
        log(1/2) + 5 log(4/9)
    log P(Y | a=5, b=0, c=0) =
        log(1/2) + 5 log(1/10)
    theta_X for "a" is 4/9 = 0.44, theta_Y is 1/10 = 0.10, so X wins.
    """
    nb = MultinomialNB(alpha=1.0)
    for x, y in FIXTURE_DOCS:
        nb.learn_one(x, y)
    proba = nb.predict_proba_one({"a": 5, "b": 0, "c": 0})
    log_px = np.log(0.5) + 5.0 * np.log(4.0 / 9.0)
    log_py = np.log(0.5) + 5.0 * np.log(1.0 / 10.0)
    px = np.exp(log_px)
    py = np.exp(log_py)
    s = px + py
    assert abs(proba["X"] - px / s) < 1e-12
    assert abs(proba["Y"] - py / s) < 1e-12
    assert nb.predict_one({"a": 5, "b": 0, "c": 0}) == "X"
    # a query with heavy "b" flips the answer: Y has b: 5, X has b: 1
    assert nb.predict_one({"a": 0, "b": 5, "c": 0}) == "Y"


# ------------------------------------------------------------------
# Bernoulli: hand computation
# ------------------------------------------------------------------


def test_bernoulli_hand_computation():
    """With alpha=1 the per-word present probability is exact.

    Class X: 2 docs, word a present in both (2), b in 1, c in 1.
        theta_X = (1 + (2,1,1)) / (2 + 2) = (3, 2, 2) / 4
    Class Y: 2 docs, a in none, b in both, c in both.
        theta_Y = (1 + (0,2,2)) / (2 + 2) = (1, 3, 3) / 4

    Query: {a:1, b:0, c:0}. Present factor for a, absent factor for b, c.
    log P(X | q) = log(1/2) + log(3/4) + log(1 - 2/4) + log(1 - 2/4)
    log P(Y | q) = log(1/2) + log(1/4) + log(1 - 3/4) + log(1 - 3/4)
    """
    nb = BernoulliNB(alpha=1.0)
    for x, y in FIXTURE_DOCS:
        nb.learn_one(x, y)
    log_px = np.log(0.5) + np.log(3.0 / 4.0) + 2.0 * np.log(1.0 - 2.0 / 4.0)
    log_py = np.log(0.5) + np.log(1.0 / 4.0) + 2.0 * np.log(1.0 - 3.0 / 4.0)
    px, py = np.exp(log_px), np.exp(log_py)
    s = px + py
    proba = nb.predict_proba_one({"a": 1, "b": 0, "c": 0})
    assert abs(proba["X"] - px / s) < 1e-12
    assert abs(proba["Y"] - py / s) < 1e-12


# ------------------------------------------------------------------
# Complement NB
# ------------------------------------------------------------------


def test_complement_sign_is_flipped():
    """CNB selects the class with the smallest log-complement score."""
    nb = ComplementNB(alpha=1.0)
    for x, y in FIXTURE_DOCS:
        nb.learn_one(x, y)
    p = nb.predict_proba_one({"a": 5, "b": 0, "c": 0})
    assert set(p.keys()) == {"X", "Y"}
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert nb.predict_one({"a": 5, "b": 0, "c": 0}) == "X"


# ------------------------------------------------------------------
# Categorical
# ------------------------------------------------------------------


def test_categorical_counts_categories():
    nb = CategoricalNB(alpha=1.0)
    for x, y in [
        ({"color": "red", "shape": "square"}, "X"),
        ({"color": "red", "shape": "circle"}, "X"),
        ({"color": "blue", "shape": "circle"}, "Y"),
        ({"color": "blue", "shape": "circle"}, "Y"),
    ]:
        nb.learn_one(x, y)
    assert nb.predict_one({"color": "blue", "shape": "circle"}) == "Y"
    assert nb.predict_one({"color": "red", "shape": "square"}) == "X"


# ------------------------------------------------------------------
# NaN and missing key
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls", [MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB]
)
def test_nan_and_missing_key_counted(Cls):
    nb = Cls()
    for x, y in (
        FIXTURE_DOCS
        if Cls is not CategoricalNB
        else [
            ({"a": "r", "b": "r", "c": "r"}, "X"),
            ({"a": "b", "b": "b", "c": "b"}, "Y"),
        ]
    ):
        nb.learn_one(x, y)
    n0 = nb.n_missing
    if Cls is CategoricalNB:
        nb.learn_one({"a": float("nan"), "b": "r", "c": "r"}, "X")
        nb.learn_one({"a": "r"}, "X")
    else:
        nb.learn_one({"a": float("nan"), "b": 1, "c": 0}, "X")
        nb.learn_one({"a": -1, "b": 1, "c": 0}, "X")
        nb.learn_one({"a": 1}, "X")
    assert nb.n_missing > n0


# ------------------------------------------------------------------
# check_estimator
# ------------------------------------------------------------------


def test_check_estimator_passes():
    check_estimator(MultinomialNB())
    check_estimator(BernoulliNB())
    check_estimator(ComplementNB())
    check_estimator(CategoricalNB())


# ------------------------------------------------------------------
# clone / get_params not overwritten by learning
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls", [MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB]
)
def test_clone_not_overwritten(Cls):
    nb = Cls(alpha=2.5)
    for x, y in (
        FIXTURE_DOCS
        if Cls is not CategoricalNB
        else [
            ({"a": "r", "b": "r", "c": "r"}, "X"),
            ({"a": "b", "b": "b", "c": "b"}, "Y"),
        ]
    ):
        nb.learn_one(x, y)
    c = nb.clone()
    assert c.alpha == 2.5
    assert c.get_params()["alpha"] == 2.5


# ------------------------------------------------------------------
# bad arguments
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls", [MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB]
)
def test_bad_alpha(Cls):
    with pytest.raises(ValueError, match="alpha"):
        Cls(alpha=0.0)


ALL_NB = (BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB)
COUNT_NB = (BernoulliNB, ComplementNB, MultinomialNB)


def _fitted(model):
    for x, y in FIXTURE_DOCS:
        model.learn_one(x, y)
    return model


def test_explicit_feature_keys_are_used():
    for cls in ALL_NB:
        assert _fitted(cls(feature_keys=["a", "b"]))._keys == ["a", "b"]


def test_empty_or_non_dict_sample_is_counted_in_n_missing():
    for cls in ALL_NB:
        m = cls()
        m.learn_one({}, "X")
        m.learn_one("a", "X")
        assert m.n_missing == 2


def test_prediction_is_empty_before_learning_and_on_unusable_input():
    for cls in COUNT_NB:
        assert cls().predict_proba_one({"a": 1}) == {}
        m = _fitted(cls())
        assert m.predict_proba_one({}) == {}
        assert m.predict_proba_one("a") == {}
        assert m.predict_proba_one({"a": "word", "b": 0, "c": 0}) == {}
        assert m.predict_proba_one({"a": -1.0, "b": 0, "c": 0}) == {}
        assert m.predict_proba_one({"a": float("nan"), "b": 0, "c": 0}) == {}


def test_categorical_prediction_is_empty_on_unusable_input():
    assert CategoricalNB().predict_one({"a": 1}) is None
    m = _fitted(CategoricalNB())
    assert m.predict_proba_one({}) == {}
    assert m.predict_proba_one({"a": 1}) == {}
    assert m.predict_proba_one({"a": 1, "b": None, "c": 1}) == {}
    assert m.predict_proba_one({"a": 1, "b": float("nan"), "c": 1}) == {}


def test_class_prior_replaces_the_observed_frequency():
    q = {"a": 1, "b": 1, "c": 1}
    for cls in (BernoulliNB, CategoricalNB, MultinomialNB):
        x_heavy = _fitted(cls(class_prior={"X": 0.9, "Y": 0.1}))
        y_heavy = _fitted(cls(class_prior={"X": 0.1, "Y": 0.9}))
        assert x_heavy.predict_proba_one(q)["X"] > y_heavy.predict_proba_one(q)["X"]
