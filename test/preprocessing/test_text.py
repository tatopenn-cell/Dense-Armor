import math

import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.text import (
    TFIDF,
    BagOfWords,
    FeatureHasher,
    Tokenizer,
)


def test_tokenizer_ngrams():
    tk = Tokenizer(ngrams=2)
    out = tk.transform_one({"text": "Hello world"})
    assert out["tokens"] == ["hello", "world", "hello world"]


def test_bag_of_words_counts():
    bw = BagOfWords()
    out = bw.transform_one({"tokens": ["a", "b", "a"]})
    assert out["counts"] == {"a": 2, "b": 1}


def test_tfidf_paper_formula_three_docs():
    tf = TFIDF()
    tf.learn_one({"tokens": ["a", "b", "c"]})
    tf.learn_one({"tokens": ["a", "b"]})
    tf.learn_one({"tokens": ["a"]})
    out = tf.transform_one({"tokens": ["a", "b", "c"]})
    n = 3
    df_a, df_b, df_c = 3, 2, 1
    doc_len = 3.0
    assert out["scores"]["a"] == pytest.approx(
        (1 / doc_len) * math.log(n / df_a), rel=1e-12
    )
    assert out["scores"]["b"] == pytest.approx(
        (1 / doc_len) * math.log(n / df_b), rel=1e-12
    )
    assert out["scores"]["c"] == pytest.approx(
        (1 / doc_len) * math.log(n / df_c), rel=1e-12
    )


def test_tfidf_term_in_every_doc_has_zero_idf():
    tf = TFIDF()
    tf.learn_one({"tokens": ["a", "b"]})
    tf.learn_one({"tokens": ["a"]})
    out = tf.transform_one({"tokens": ["a", "b"]})
    assert out["scores"]["a"] == 0.0
    assert out["scores"]["b"] > 0.0


def test_tfidf_smooth_idf_option():
    tf = TFIDF(smooth_idf=True)
    tf.learn_one({"tokens": ["a", "b"]})
    tf.learn_one({"tokens": ["a"]})
    out = tf.transform_one({"tokens": ["a"]})
    want = 1.0 * math.log(1.0 + 2.0 / 2.0)
    assert out["scores"]["a"] == pytest.approx(want, rel=1e-12)


def test_feature_hasher_deterministic():
    fh = FeatureHasher(n_features=16, seed=7)
    o1 = fh.transform_one({"counts": {"hello": 2.0}})
    o2 = fh.transform_one({"counts": {"hello": 2.0}})
    assert o1["hashed"] == o2["hashed"]


def test_check_estimator_tokenizer():
    check_estimator(Tokenizer())


def test_check_estimator_bow():
    check_estimator(BagOfWords())


def test_check_estimator_tfidf():
    check_estimator(TFIDF())


def test_check_estimator_hasher():
    check_estimator(FeatureHasher())
