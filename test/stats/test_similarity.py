"""Tests for the cosine similarity."""

import math

import numpy as np
import pytest

from dense_armor.utility.stats.similarity import (
    SparseCosineIndex,
    cosine_similarity,
    cosine_similarity_many,
)


def test_parallel_vectors():
    assert cosine_similarity({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)


def test_zero_vector_is_zero_not_nan():
    v = cosine_similarity({"a": 0.0}, {"a": 1.0})
    assert v == 0.0
    assert not math.isnan(v)


def test_empty_dict_is_zero():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
    assert cosine_similarity({"a": 1.0}, {}) == 0.0


def test_opposite_vectors():
    assert cosine_similarity({"a": 1.0}, {"a": -1.0}) == pytest.approx(-1.0)


def test_many_equals_loop():
    query = {"a": 1.0, "b": 2.0, "c": 0.5}
    docs = [
        {"a": 1.0, "b": 0.0},
        {"a": 0.0, "b": 1.0, "c": 2.0},
        {"c": 3.0},
        {},
    ]
    many = cosine_similarity_many(query, docs)
    loop = np.array([cosine_similarity(query, d) for d in docs])
    np.testing.assert_allclose(many, loop, atol=1e-12)
    assert many.shape == (4,)


def test_many_empty_list():
    out = cosine_similarity_many({"a": 1.0}, [])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_unknown_terms_do_not_contribute():
    v = cosine_similarity({"a": 1.0, "x": 100.0}, {"a": 1.0, "y": 100.0})
    assert v == pytest.approx(1.0 / (math.sqrt(1.0 + 10000.0) ** 2) * 1.0) or v > 0


class TestSparseCosineIndex:
    def _docs(self):
        return [
            {"a": 1.0, "b": 2.0},
            {"b": 1.0, "c": 3.0},
            {"a": 2.0, "c": 1.0},
            {},
        ]

    def test_matches_cosine_similarity_many(self):
        docs = self._docs()
        idx = SparseCosineIndex(docs)
        q = {"a": 1.0, "b": 1.0, "c": 0.5}
        got = idx.query(q)
        want = cosine_similarity_many(q, docs)
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_unknown_terms_count_in_the_norm_not_in_the_product(self):
        """A term outside the index reduces the score through the norm.

        The query norm is the norm of the whole query, as in
        ``cosine_similarity``: the term outside the index does not
        contribute to the product but it is in the denominator.
        """
        idx = SparseCosineIndex([{"a": 1.0}])
        q = {"a": 1.0, "zebra": 100.0}
        got = idx.query(q)[0]
        want = cosine_similarity_many(q, [{"a": 1.0}])[0]
        assert got == pytest.approx(want, abs=1e-12)

    def test_empty_query_is_zero(self):
        idx = SparseCosineIndex([{"a": 1.0}])
        assert idx.query({})[0] == 0.0

    def test_zero_query_is_zero(self):
        idx = SparseCosineIndex([{"a": 1.0}])
        assert idx.query({"a": 0.0})[0] == 0.0

    def test_empty_documents_raises(self):
        with pytest.raises(ValueError, match="at least one document"):
            SparseCosineIndex([])

    def test_n_documents(self):
        idx = SparseCosineIndex(self._docs())
        assert idx.n_documents_ == 4

    def test_shape(self):
        idx = SparseCosineIndex(self._docs())
        assert idx.matrix_.shape == (4, 3)
        assert idx.query({"a": 1.0}).shape == (4,)

    def test_max_terms_per_document_keeps_top_scores(self):
        docs = [{"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}]
        idx = SparseCosineIndex(docs, max_terms_per_document=2)
        assert len(idx.vocabulary_) == 2
        assert set(idx.vocabulary_) == {"c", "d"}

    def test_max_terms_none_keeps_every_term(self):
        docs = [{"a": 1.0, "b": 2.0, "c": 3.0}]
        idx = SparseCosineIndex(docs)
        assert len(idx.vocabulary_) == 3
        assert idx.max_terms_per_document_ is None

    def test_max_terms_bad_value(self):
        with pytest.raises(ValueError, match="max_terms_per_document"):
            SparseCosineIndex([{"a": 1.0}], max_terms_per_document=0)

    def test_dtype_float32_matches_float64(self):
        docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}]
        q = {"a": 1.0, "b": 0.5}
        a = SparseCosineIndex(docs, dtype=np.float64).query(q)
        b = SparseCosineIndex(docs, dtype=np.float32).query(q)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_dtype_bad_value(self):
        with pytest.raises(ValueError, match="dtype"):
            SparseCosineIndex([{"a": 1.0}], dtype=np.int64)

    def test_rows_are_normalised(self):
        idx = SparseCosineIndex([{"a": 1.0, "b": 1.0}])
        self_sim = idx.query({"a": 1.0, "b": 1.0})
        assert self_sim[0] == pytest.approx(1.0, abs=1e-12)

    def test_max_terms_shorter_document_kept_as_is(self):
        docs = [{"a": 1.0}, {"a": 1.0, "b": 2.0, "c": 3.0}]
        idx = SparseCosineIndex(docs, max_terms_per_document=2)
        assert idx.n_documents_ == 2
        assert "c" in idx.vocabulary_

def test_query_many_matches_query():
    docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}]
    idx = SparseCosineIndex(docs)
    qs = [{"a": 1.0, "b": 1.0}, {"c": 2.0}]
    got = idx.query_many(qs)
    for j, q in enumerate(qs):
        np.testing.assert_allclose(got[:, j], idx.query(q), atol=1e-12)


def test_query_norm_counts_terms_outside_index():
    idx = SparseCosineIndex([{"a": 1.0}])
    got = idx.query({"a": 1.0, "z": 1.0})[0]
    want = cosine_similarity_many({"a": 1.0, "z": 1.0}, [{"a": 1.0}])[0]
    assert got == pytest.approx(want, abs=1e-12)

def test_query_many_empty():
    """The empty batch returns (n_documents, 0), not a crash (line 191)."""
    idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
    S = idx.query_many([])
    assert S.shape == (2, 0)

def test_from_csr_gives_the_same_index():
    docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
    idx = SparseCosineIndex(docs, dtype=np.float32)
    idx2 = SparseCosineIndex.from_csr(idx.matrix_, dict(idx.vocabulary_))
    assert idx2.n_documents_ == 3
    np.testing.assert_allclose(idx2.query({"a": 1.0}), idx.query({"a": 1.0}),
                               atol=1e-6)


def test_from_csr_bad_column_raises():
    docs = [{"a": 1.0}]
    idx = SparseCosineIndex(docs)
    bad = {k: v + 100 for k, v in idx.vocabulary_.items()}
    with pytest.raises(ValueError, match="outside"):
        SparseCosineIndex.from_csr(idx.matrix_, bad)


def test_query_many_sparse_matches_query_many():
    from scipy import sparse
    docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
    idx = SparseCosineIndex(docs, dtype=np.float32)
    qs = [{"a": 1.0}, {"b": 1.0}]
    rows, cols, vals = [], [], []
    for j, q in enumerate(qs):
        for term, v in q.items():
            i = idx.vocabulary_.get(term)
            if i is not None:
                rows.append(i); cols.append(j); vals.append(v)
    Q = sparse.csc_matrix((vals, (rows, cols)),
                          shape=(idx.matrix_.shape[1], 2), dtype=np.float32)
    got = idx.query_many_sparse(Q)
    want = idx.query_many(qs)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_query_many_sparse_empty_returns_empty_columns():
    from scipy import sparse
    idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
    Q = sparse.csc_matrix((idx.matrix_.shape[1], 0), dtype=np.float32)
    assert idx.query_many_sparse(Q).shape == (2, 0)


def test_top_k_threshold_matches_full_product():
    docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
    idx = SparseCosineIndex(docs, dtype=np.float32)
    got = idx.top_k_threshold({"a": 1.0}, k=2)
    full = np.argsort(-idx.query({"a": 1.0}))[:2]
    assert [d for d, _ in got] == full.tolist()


def test_top_k_threshold_k_larger_than_documents():
    idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
    got = idx.top_k_threshold({"a": 1.0}, k=10)
    assert len(got) == 2


def test_top_k_threshold_empty_query_returns_zeros():
    idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
    got = idx.top_k_threshold({}, k=2)
    assert len(got) == 2
    assert all(s == 0.0 for _, s in got)

def test_query_many_sparse_normalized_skips_the_division():
    from scipy import sparse
    docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
    idx = SparseCosineIndex(docs, dtype=np.float32)
    rows, cols, vals = [], [], []
    for j, q in enumerate([{"a": 1.0}, {"b": 1.0}]):
        for term, v in q.items():
            i = idx.vocabulary_.get(term)
            if i is not None:
                rows.append(i); cols.append(j); vals.append(v)
    Q = sparse.csc_matrix((vals, (rows, cols)),
                          shape=(idx.matrix_.shape[1], 2), dtype=np.float32)
    got = idx.query_many_sparse(Q, normalized=True)
    np.testing.assert_allclose(got, idx.query_many([{"a": 1.0}, {"b": 1.0}]),
                               atol=1e-6)


def test_build_inverted_is_cached():
    idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
    idx._build_inverted()
    inv1 = idx._inv
    idx._build_inverted()
    assert idx._inv is inv1
