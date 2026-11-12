"""The adapter that opens the primitives of the library to text."""

import pytest

from dense_armor.core.audio.adapter import NeedsLabels
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.core.signals.preprocessing.text_input import N_TERMS, TextInput, _Terms
from dense_armor.core.signals.preprocessing.vectorizer import TextVectorizer
from dense_armor.core.signals.stats.sdft import SDFT

POS = [
    "great fast reliable product",
    "love it works well",
    "very happy with the result",
]
NEG = [
    "terrible slow broken product",
    "hate it does not work",
    "very unhappy with the result",
]
DOCS = [(t, 1) for t in POS] + [(t, 0) for t in NEG]


class Scorer:
    def __init__(self):
        self.seen = []

    def score_one(self, x):
        self.seen.append(dict(x))
        return float(len(x))

    def learn_one(self, x):
        return self


class NeedsANumber:
    """Takes one number: a dict raises ``TypeError``."""

    def score_one(self, x):
        return float(x) + 1.0

    def learn_one(self, x):
        return self


class Learner:
    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        self.n += 1


def test_the_scores_of_a_document_read_a_missing_term_as_zero_without_adding_it():
    t = _Terms({"cat": 0.5, "mat": 0.5})
    assert t["dog"] == 0.0
    assert "dog" not in t
    assert t[N_TERMS] == 2.0
    assert N_TERMS not in t


def test_a_primitive_with_score_one_scores_and_learns_the_flat_scores():
    p = Scorer()
    a = TextInput(p)
    out = a.feed("the cat sat on the mat")
    assert a.mode == "score" and a.n == 1 and out == a.last
    assert all(isinstance(v, float) for v in p.seen[0].values())
    assert "text" not in p.seen[0]


def test_a_primitive_that_refuses_the_dict_is_given_the_number_of_terms():
    a = TextInput(NeedsANumber())
    out = a.feed("one two three")
    assert a.scalar is True
    assert out == len(a.vectorizer.transform_one("one two three")["scores"]) + 1.0


def test_a_term_can_be_the_number_given_to_a_scalar_primitive():
    a = TextInput(NeedsANumber(), key="cat")
    assert a.feed("a dog") == 1.0
    assert a.scalar is True


def test_a_classifier_predicts_and_learns_with_the_label():
    a = TextInput(OnlineGaussianNB())
    for text, y in DOCS * 2:
        a.feed(text, y)
    assert a.mode == "supervised" and a.n == 12


def test_a_classifier_without_a_label_says_so():
    a = TextInput(OnlineGaussianNB())
    with pytest.raises(NeedsLabels):
        a.feed("the cat")
    assert a.mode == "needs labels"


def test_a_regressor_learns_the_length_of_the_document():
    a = TextInput(RecursiveLeastSquares())
    for text, _ in DOCS * 2:
        a.feed(text, float(len(text)))
    assert a.mode == "supervised"


def test_a_drift_detector_gets_one_number_per_document():
    a = TextInput(ADWIN())
    for text, _ in DOCS * 3:
        a.feed(text)
    assert a.mode == "drift" and a.n == 18


def test_a_transformer_learns_and_transforms():
    a = TextInput(TextVectorizer())
    out = a.feed("the cat sat")
    assert a.mode == "transform" and "scores" in out


def test_a_primitive_that_only_learns_just_learns():
    p = Learner()
    a = TextInput(p)
    a.feed("the cat")
    assert a.mode == "learn" and p.n == 1


def test_a_primitive_of_samples_has_no_samples_in_a_document():
    with pytest.raises(ValueError, match="text has none"):
        TextInput(SDFT(4)).feed("the cat")


def test_an_object_without_methods_is_refused():
    with pytest.raises(ValueError, match="no known method"):
        TextInput(object()).feed("the cat")


def test_the_scores_can_be_given_and_the_vectorizer_shared():
    v = TextVectorizer()
    a = TextInput(Scorer(), vectorizer=v)
    a.feed("the cat sat")
    assert "cat" in v.transform_one("cat")["scores"]
    a.feed("ignored", x={"cat": 1.0})
    assert a.vectorizer is v and a.n == 2


def test_run_feeds_every_document_with_its_label():
    a = TextInput(OnlineGaussianNB())
    labels = dict(DOCS)
    out = a.run([t for t, _ in DOCS], label=lambda t: labels[t])
    assert len(out) == 6 and a.n == 6
