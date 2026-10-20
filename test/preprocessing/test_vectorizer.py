"""Tests for the text vectorizer."""

import json

import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.vectorizer import TextVectorizer

DOCS = ["the quick brown fox", "a quick brown dog", "the fox and the dog"]


def test_batch_and_online_end_in_the_same_state():
    a = TextVectorizer().fit(DOCS)
    b = TextVectorizer()
    for d in DOCS:
        b.learn_one(d)
    assert a.tfidf_.n_docs_ == b.tfidf_.n_docs_ == 3
    assert a.tfidf_.df_ == b.tfidf_.df_


def test_transform_equals_transform_one():
    v = TextVectorizer().fit(DOCS)
    many = v.transform(DOCS)
    one = [v.transform_one(d) for d in DOCS]
    for x, y in zip(many, one):
        assert x["scores"] == y["scores"]


def test_state_dict_round_trip():
    v = TextVectorizer().fit(DOCS)
    s = json.dumps(v.state_dict())
    w = TextVectorizer().load_state_dict(json.loads(s))
    for d in DOCS:
        assert w.transform_one(d)["scores"] == v.transform_one(d)["scores"]


def test_max_features_keeps_highest_df_alphabetical_tie():
    v = TextVectorizer(max_features=3).fit(DOCS)
    assert set(v.tfidf_.df_) == {"brown", "dog", "fox"}


def test_max_features_none_keeps_all():
    v = TextVectorizer().fit(DOCS)
    assert len(v.tfidf_.df_) > 3


def test_unseen_term_has_no_score():
    v = TextVectorizer().fit(DOCS)
    s = v.transform_one("zebra and the fox")["scores"]
    assert "zebra" not in s


def test_ngram_range_validation():
    with pytest.raises(ValueError, match="ngram_range"):
        TextVectorizer(ngram_range=(2, 1))
    with pytest.raises(ValueError, match="ngram_range"):
        TextVectorizer(ngram_range=(0, 2))


def test_max_features_validation():
    with pytest.raises(ValueError, match="max_features"):
        TextVectorizer(max_features=0)


def test_norm_validation():
    with pytest.raises(ValueError, match="norm"):
        TextVectorizer(norm="l1")


def test_stop_words_forwarded():
    v = TextVectorizer(stop_words=["the"]).fit(DOCS)
    assert all("the" not in t.split() for t in v.tfidf_.df_)


def test_check_estimator_passes():
    out = check_estimator(TextVectorizer())
    assert out["not_applicable"] == []
    assert len(out["skipped"]) == 0


def test_budget_above_measured_p99():
    import time

    import numpy as np
    v = TextVectorizer()
    ts = []
    for d in DOCS:
        t0 = time.perf_counter()
        v.learn_one(d)
        ts.append(time.perf_counter() - t0)
    assert float(np.percentile(ts, 99)) < v.budget_s


def test_as_text_other_type():
    v = TextVectorizer().fit(DOCS)
    s = v.transform_one(42)["scores"]
    assert isinstance(s, dict)


def test_max_features_larger_than_vocabulary_keeps_all():
    v = TextVectorizer(max_features=1000).fit(DOCS)
    assert len(v.tfidf_.df_) > 3

def test_fit_twice_resets_state():
    v = TextVectorizer()
    v.fit(DOCS)
    v.fit(DOCS)
    assert v.tfidf_.n_docs_ == 3


def test_none_and_nan_are_missing():
    v = TextVectorizer()
    v.learn_one(None)
    v.learn_one(float("nan"))
    v.learn_one(5)
    assert v.n_missing_ == 3
    assert v.tfidf_.n_docs_ == 0


def test_state_dict_keeps_stop_words():
    import json
    v = TextVectorizer(stop_words=["the"]).fit(DOCS)
    s = json.dumps(v.state_dict())
    w = TextVectorizer().load_state_dict(json.loads(s))
    assert w.stop_words == ("the",)
    assert w.transform_one(DOCS[0])["scores"] == v.transform_one(DOCS[0])["scores"]


def test_generator_stop_words_not_consumed():
    v = TextVectorizer(stop_words=(w for w in ["the"]))
    for d in DOCS:
        v.learn_one(d)
    assert "the" not in v.tfidf_.df_
