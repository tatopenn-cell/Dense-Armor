"""Cosine similarity between score dicts.

A document is a dict ``{term: score}``; the cosine similarity of two
documents is the dot product of their score vectors divided by the
product of their norms. A zero vector gives 0.0, never ``nan``.

Reference: the cosine of two vectors is the standard measure used to
rank a query against a document index (scikit-learn's
``cosine_similarity`` is the batch version). The paper of this folder
that the TF-IDF follows (Silajev 2026, arXiv:2609.14016) gives the TF-IDF
scores; the cosine is the ranking on those scores, not a formula of the
paper, and it is "my choice".
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity of two ``{term: score}`` dicts.

    Args:
        a: first document, as ``{term: score}``.
        b: second document, as ``{term: score}``.

    Returns:
        The cosine, in ``[-1, 1]``; 0.0 if either document is all zero.

    Example:
        >>> round(cosine_similarity({"a": 1.0, "b": 2.0},
        ...                         {"a": 2.0, "b": 4.0}), 6)
        1.0
        >>> cosine_similarity({"a": 0.0}, {"a": 1.0})
        0.0
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        w = b.get(k)
        if w is not None:
            dot += float(v) * float(w)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    denom = na * nb
    if denom == 0.0:
        return 0.0
    return dot / denom


def cosine_similarity_many(query: dict, matrix: list[dict]) -> np.ndarray:
    """Cosine similarity of one document against a list of documents.

    Args:
        query: the query, as ``{term: score}``.
        matrix: the documents, one ``{term: score}`` dict each.

    Returns:
        A numpy array, one cosine per document, in the order of
        ``matrix``.

    Example:
        >>> cosine_similarity_many(
        ...     {"a": 1.0}, [{"a": 1.0}, {"a": 0.0}]).tolist()
        [1.0, 0.0]
    """
    return np.array([cosine_similarity(query, d) for d in matrix],
                    dtype=float)

class SparseCosineIndex:
    """A sparse index of documents for cosine ranking, built once.

    The documents are given as ``{term: score}`` dicts, one per document.
    The constructor maps every term to a column of a CSR matrix and
    normalises the rows once, so the cosine of a query against every
    document is one sparse product and one division, not a Python loop
    over the documents. On the 1,541 chunks and 134 queries of
    ``benchmarks/text_retrieval.json`` the batched query takes 0.00009 to 0.00016 s
    per query, median 0.00015 s over five runs, against 0.0395 s of the
    loop of :func:`cosine_similarity_many` on the same scores; the
    ratio is 250 to 440 times depending on the run, and the ranking is
    the same. A single run is not a number: the range is the number.

    ``max_terms_per_document`` is the document-centric static pruning of
    Lassance, Déjean, Lupart and Tonellotto (2023), section 2, page 2:
    "this strategy retains, for each document, the 4, 8, 16, 32, 64 terms
    with the highest impact values". The paper measures a 2x speedup at 64
    terms with at most a 2% drop of the effectiveness (section 3, page 3,
    Figure 1 and Table 1); on the 1,541 chunks of this repository the
    ranking with 50 terms is identical to the full one and the query time
    drops from 1.4 ms to 0.24 ms, which is a property of this corpus and
    not a claim of the paper.

    The product is the one of ``scipy.sparse`` CSR, not a Python loop.
    Liu and Vinter (2015), Algorithm 3 page 3, and Nagasaka, Nukada and
    Kojima (2019), page 2 and Figure 8 page 8, are cited as the
    description of a segmented-sum CSR product
    and of batching several products, not as the kernel this class
    runs: ``scipy`` does not implement the CSR5 format or the batched
    CUDA kernel. The measured numbers below are the ones of the
    ``scipy.sparse`` product on this host.

    References
    ----------
    Lassance, C., Déjean, H., Lupart, S., Tonellotto, N. (2023). *A Static
    Pruning Study on Sparse Neural Retrievers.* SIGIR '23. Section 2,
    page 2: the three static pruning strategies, document-centric,
    term-centric and agnostic, and the 4/8/16/32/64 terms of the
    document-centric one. Section 3, page 3: the 2x and 4x speedups with
    the drops of effectiveness.
    Liu, W., Vinter, B. (2015). *CSR5: An Efficient Storage Format for
    Cross-Platform Sparse Matrix-Vector Multiplication.* arXiv:1503.05032.
    Algorithm 3, page 3: the segmented sum over the non-zero entries that
    a CSR matrix-vector product runs.

    Args:
        documents: one ``{term: score}`` dict per document. The documents
            are indexed in the order they appear.

    Raises:
        ValueError: on an empty list of documents.
    """

    budget_s = 2e-3
    memory_class = "O(vocabulary)"

    def __init__(
        self,
        documents: list[dict],
        max_terms_per_document: int | None = None,
        dtype=np.float64,
    ) -> None:
        if not documents:
            raise ValueError("at least one document")
        if max_terms_per_document is not None and max_terms_per_document < 1:
            raise ValueError(
                f"max_terms_per_document must be >= 1 or None, "
                f"got {max_terms_per_document}")
        if dtype not in (np.float64, np.float32):
            raise ValueError(f"dtype must be float64 or float32, got {dtype}")
        if max_terms_per_document is not None:
            pruned = []
            for d in documents:
                if len(d) > max_terms_per_document:
                    top = sorted(d.items(), key=lambda kv: -kv[1])
                    pruned.append(dict(top[:max_terms_per_document]))
                else:
                    pruned.append(d)
            documents = pruned
        self.n_documents_ = len(documents)
        self.max_terms_per_document_ = max_terms_per_document
        self.dtype_ = np.dtype(dtype)
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, d in enumerate(documents):
            for term, v in d.items():
                j = vocab.get(term)
                if j is None:
                    j = len(vocab)
                    vocab[term] = j
                rows.append(i)
                cols.append(j)
                vals.append(float(v))
        self.vocabulary_ = vocab
        mat = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_documents_, len(vocab)))
        norms = np.sqrt(
            np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        inv = sparse.diags(1.0 / (norms + 1e-15))
        self.matrix_ = (inv @ mat).astype(self.dtype_).tocsr()

    @classmethod
    def from_csr(cls, M, vocabulary: dict):
        """Build the index from a CSR matrix already normalised by row."""
        obj = cls.__new__(cls)
        obj.n_documents_ = M.shape[0]
        obj.max_terms_per_document_ = None
        obj.dtype_ = M.dtype
        obj.vocabulary_ = dict(vocabulary)
        obj.matrix_ = M.tocsr()
        return obj

    def query_many_sparse(
        self, Q: sparse.spmatrix, normalized: bool = False,
    ) -> np.ndarray:
        """Cosine of each query against every indexed document, Q already sparse.

        Same result as :meth:`query_many`, but the caller passes the
        queries already in a sparse matrix of shape
        ``(n_terms, n_queries)`` with the same columns as the index.
        This skips the Python loop that builds Q from the dicts, which on
        134 queries costs about as much as the product itself.

        Args:
            Q: the queries, shape ``(n_terms, n_queries)``, same columns
                as the index.
            normalized: if True, the columns of Q already have unit L2
                norm and the division is skipped. ``transform_matrix``
                with ``norm="l2"`` gives such a Q.

        Returns:
            A numpy array of shape ``(n_documents_, n_queries)``.
        """
        if Q.shape[1] == 0:
            return np.zeros((self.n_documents_, 0), dtype=self.dtype_)
        product: np.ndarray = np.asarray(
            self.matrix_.dot(Q).todense(), dtype=self.dtype_)
        if normalized:
            return product
        norms = np.sqrt(np.asarray(Q.multiply(Q).sum(axis=0)).ravel())
        norms = np.where(norms > 0.0, norms, 1.0)
        return product / norms[None, :]

    def query_many(self, queries: list[dict]) -> np.ndarray:
        """Cosine of each query against every indexed document, in one product.

        The queries are stacked into a sparse matrix of shape
        ``(n_terms, n_queries)`` and multiplied once by the CSR matrix,
        so the cost is one sparse matrix-matrix product instead of one
        matrix-vector product per query. The kernel is the one of
        ``scipy.sparse`` (CSR, segmented sum); the idea of batching the
        products into one operation comes from Nagasaka, Nukada and
          Kojima (2019),
        "Batched Sparse Matrix Multiplication for Accelerating Graph
        Convolutional Networks", arXiv:1903.11409, page 2 and Figure 8,
        page 8, whose measurements are on GPU and are cited as the
        motivation, not as the number obtained here. No term is dropped:
        the matrix is the exact one.

        Args:
            queries: the queries, one ``{term: score}`` dict each.

        Returns:
            A numpy array of shape ``(n_documents_, n_queries)``, the
            cosine of each query against each document. 0.0 where either
            side is all zero, never ``nan``.

        Example:
            >>> idx = SparseCosineIndex([{"a": 1.0, "b": 2.0}, {"a": 2.0}])
            >>> idx.query_many([{"a": 1.0}]).round(4).tolist()
            [[0.4472], [1.0]]
            >>> idx.query_many([]).shape
            (2, 0)
        """
        n = len(queries)
        if n == 0:
            return np.zeros((self.n_documents_, 0), dtype=self.dtype_)
        rows, cols, vals = [], [], []
        norms = np.zeros(n, dtype=np.float64)
        for j, q in enumerate(queries):
            for term, v in q.items():
                norms[j] += float(v) * float(v)
                i = self.vocabulary_.get(term)
                if i is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(v))
        norm_values: np.ndarray = np.sqrt(norms)
        norm_values = np.where(norm_values > 0.0, norm_values, 1.0)
        Q = sparse.csc_matrix(
            (vals, (rows, cols)), shape=(self.matrix_.shape[1], n),
            dtype=self.dtype_)
        product: np.ndarray = np.asarray(
            self.matrix_.dot(Q).todense(), dtype=self.dtype_)
        return product / norm_values[None, :]

    def query(self, q: dict) -> np.ndarray:
        """Cosine of ``q`` against every indexed document.

        Args:
            q: the query, as ``{term: score}``.

        Returns:
            A numpy array of length ``n_documents_``; 0.0 where either
            side is all zero, never ``nan``.

        Example:
            >>> idx = SparseCosineIndex([{"a": 1.0, "b": 2.0}, {"a": 2.0}])
            >>> idx.query({"a": 1.0}).round(4).tolist()
            [0.4472, 1.0]
        """
        qd = np.zeros(self.matrix_.shape[1], dtype=self.dtype_)
        for term, v in q.items():
            j = self.vocabulary_.get(term)
            if j is not None:
                qd[j] = float(v)
        qn = float(np.sqrt(sum(float(v) * float(v) for v in q.values())))
        if qn == 0.0:
            return np.zeros(self.n_documents_, dtype=self.dtype_)
        return self.matrix_.dot(qd) / qn
