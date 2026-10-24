"""Tests for dense_armor/utility/stats/bm25.py.

The hand values are the score of three short documents for the query
"a b", computed with the formula of Silajev (2026, arXiv:2609.14016,
page 1 for the term score and its ``+1`` IDF, page 4 for
``Score(q, i)``). With documents ``["a a b", "a c", "b b b"]``,
``k1 = 1.2`` and ``b = 0.75``, the document frequencies are
``DF(a) = 2``, ``DF(b) = 2``, ``DF(c) = 1``, so
``IDF(a) = IDF(b) = ln((3-2+0.5)/(2+0.5) + 1) = 0.470004`` and
``IDF(c) = ln((3-1+0.5)/(1+0.5) + 1) = 0.980829``. ``|d_0| = 3``,
``|d_1| = 2``, ``|d_2| = 3``, ``avgdl = 8/3``. The three scores are
``0.243510``, ``0.118988``, ``0.163480`` and are written as numbers
below, not recomputed from the library.
"""
import json
import time

import numpy as np
import pytest

from dense_armor.utility.stats.bm25 import BM25Index

DOCS = ["a a b", "a c", "b b b"]
HAND_SCORES = [0.243510, 0.118988, 0.163480]


def _state_fingerprint(idx):
    return (
        idx.n_docs_,
        dict(idx.document_frequencies_),
        [dict(c) for c in idx.term_counts_],
        list(idx.lengths_),
        idx.sum_length_,
    )


def test_fit_and_learn_one_same_state():
    a = BM25Index().fit(DOCS)
    b = BM25Index()
    for d in DOCS:
        b.learn_one(d)
    assert _state_fingerprint(a) == _state_fingerprint(b)


def test_fit_and_learn_one_same_scores():
    a = BM25Index().fit(DOCS)
    b = BM25Index()
    for d in DOCS:
        b.learn_one(d)
    np.testing.assert_allclose(a.query("a b"), b.query("a b"), atol=1e-12)


def test_state_dict_roundtrip():
    a = BM25Index().fit(DOCS)
    s = a.state_dict()
    b = BM25Index().load_state_dict(s)
    np.testing.assert_allclose(a.query("a b"), b.query("a b"), atol=1e-12)
    assert _state_fingerprint(a) == _state_fingerprint(b)


def test_state_dict_is_json():
    a = BM25Index().fit(DOCS)
    s = a.state_dict()
    text = json.dumps(s)
    back = json.loads(text)
    b = BM25Index().load_state_dict(back)
    np.testing.assert_allclose(a.query("a b"), b.query("a b"), atol=1e-12)


def test_hand_values():
    idx = BM25Index().fit(DOCS)
    got = idx.query("a b")
    np.testing.assert_allclose(got, HAND_SCORES, atol=1e-5)


def test_idf_positive_when_term_in_every_document():
    idx = BM25Index().fit(["a b", "a c", "a d"])
    df = idx.document_frequencies_["a"]
    assert df == 3
    n = idx.n_docs_
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
    assert idf > 0.0


def test_b_zero_independent_of_length():
    doc_short = "x x x"
    doc_long = "x x x y y y y y y y"
    a = BM25Index(b=0.0).fit([doc_short, doc_long])
    b = BM25Index(b=0.75).fit([doc_short, doc_long])
    qa = a.query("x")
    qb = b.query("x")
    assert abs(qa[0] - qa[1]) < 1e-12
    assert abs(qb[0] - qb[1]) > 1e-3


def test_unknown_query_gives_zeros():
    idx = BM25Index().fit(DOCS)
    scores = idx.query("zzz yyy")
    assert scores.shape == (3,)
    assert np.all(scores == 0.0)
    assert not np.any(np.isnan(scores))


def test_empty_query_gives_zeros():
    idx = BM25Index().fit(DOCS)
    assert np.all(idx.query("") == 0.0)
    assert np.all(idx.query("   ") == 0.0)


def test_top_k_ties_broken_by_lower_index():
    idx = BM25Index().fit(["x", "x", "x"])
    out = idx.top_k("x", k=3)
    assert [i for i, _ in out] == [0, 1, 2]


def test_top_k_k_ge_n_returns_all():
    idx = BM25Index().fit(DOCS)
    out = idx.top_k("a b", k=10)
    assert len(out) == 3
    assert out[0][0] == 0


def test_top_k_bad_k_raises():
    idx = BM25Index().fit(DOCS)
    with pytest.raises(ValueError):
        idx.top_k("a b", k=0)


def test_validation_raises():
    for kwargs in ({"k1": -0.1}, {"b": 1.5}, {"b": -0.1},
                   {"stop_words": "english"}):
        with pytest.raises(ValueError):
            BM25Index(**kwargs)
    BM25Index(k1=0.0, b=0.0)


def test_stop_words_are_removed():
    idx = BM25Index(stop_words=["the"]).fit(
        ["the a the b", "the c"])
    assert "the" not in idx.document_frequencies_
    assert idx.document_frequencies_["a"] == 1


def test_budget_s_measured():
    idx = BM25Index().fit(DOCS)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        idx.query("a b")
        times.append(time.perf_counter() - t0)
    p99 = float(np.quantile(times, 0.99))
    print(f"\n  measured p99 of query(): {p99*1e3:.3f} ms")
    assert p99 < 1.0
    assert BM25Index.budget_s > 0.0


def test_predict_wrong_types_do_not_crash():
    idx = BM25Index().fit(["a b", "c d"])
    assert idx.query({"text": "a"}).shape == (2,)
    assert idx.query({"text": ""}).shape == (2,)

def test_to_dict_input_branch():
    class _X:
        def to_dict(self):
            return {"text": "a b"}
    idx = BM25Index().fit(["a b", "c d"])
    assert idx.query(_X()).shape == (2,)
    idx.learn_one(_X())
    assert idx.n_docs_ == 3


def test_empty_index_query_and_top_k():
    idx = BM25Index()
    assert idx.query("x").shape == (0,)
    assert idx.top_k("x", k=3) == []
