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
    over the documents. The cost per query of the batched product and of
    the loop of :func:`cosine_similarity_many` on the same scores, with
    the range over five runs, is in ``query_s``, ``query_s_min``,
    ``query_s_max`` and ``ours_loop`` of ``benchmarks/text_retrieval.json``.

    ``max_terms_per_document`` is the document-centric static pruning of
    Lassance, Déjean, Lupart and Tonellotto (2023), section 2, page 2:
    "this strategy retains, for each document, the 4, 8, 16, 32, 64 terms
    with the highest impact values". The paper measures a 2x speedup at 64
    terms with at most a 2% drop of the effectiveness (section 3, page 3,
    Figure 1 and Table 1). On the chunks of this repository the top-5 of
    the pruned index overlaps the exact one by the value of
    ``pruned_k50_top5_overlap_vs_exact_median`` in
    ``benchmarks/text_retrieval.json``, so the default stays the exact
    index and the pruning is an option.

    The product is the one of ``scipy.sparse`` CSR, not a Python loop.
    Liu and Vinter (2015), Algorithm 3 page 3, and Nagasaka et al.
    (2019), page 2 and Figure 8 page 8, are cited as the
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
        """Build the index from a CSR matrix already normalised by row.

        Args:
            M: the documents, shape (n_documents, n_terms), rows with
                unit L2 norm.
            vocabulary: term to column, each column in [0, M.shape[1]).

        Raises:
            ValueError: on a column outside [0, M.shape[1]), so the index
                does not fail later with an IndexError on a query.
        """
        n_terms = M.shape[1]
        for term, col in vocabulary.items():
            if not 0 <= col < n_terms:
                raise ValueError(
                    f"column of {term!r} is {col}, outside [0, {n_terms})")
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
        products into one operation comes from Nagasaka et al. (2019),
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

    def _build_inverted(self) -> None:
        """Inverted index: term -> (doc ids, weights), sorted by weight desc."""
        if getattr(self, "_inv", None) is not None:
            return
        M = self.matrix_.tocsc()
        self._inv = {}
        for col in range(M.shape[1]):
            start, end = M.indptr[col], M.indptr[col + 1]
            if start == end:
                continue
            docs = M.indices[start:end]
            weights = M.data[start:end]
            order = np.argsort(-weights)
            self._inv[col] = (docs[order].astype(np.int32),
                              weights[order].astype(np.float32))

    def top_k_threshold(self, q: dict, k: int = 5) -> list[tuple[int, float]]:
        """Exact top-k by the Threshold Algorithm of Fagin, Lotem and Naor.

        Fagin, R., Lotem, A., Naor, M. (2001). Optimal Aggregation
        Algorithms for Middleware. PODS 2001; JCSS 67(4), 2003. The
        Threshold Algorithm, section 3. The paper is not on arXiv and
        it is not in the folder: the section number is the one the
        citation reports, not verified against the PDF.

        The posting lists are sorted by weight descending, so the first
        not-yet-consumed weight of each list bounds every further
        contribution of that term. The threshold is the sum over the terms
        not yet processed of q_t times the next weight of that term. Every
        document not yet seen has sim(q, d) <= threshold. When the
        threshold is below the k-th partial score, the scan stops. The
        partial scores of the touched documents are then recomputed
        exactly, because a touched document can still grow. Exact: the
        top-k equals the one of the full product.

        When to use it: a corpus where the full product is prohibitive
        (millions of documents) and the posting lists are long enough that
        the threshold drops after few rows. On a small corpus the random
        access on the candidates costs as much as the full product, and
        the full product is C while this is Python: measure before use.
        On the chunks of this repository it gives the same top-5 as the
        full product on every query and is slower: the cost per query of
        both is in ``benchmarks/primitives.json``, written by
        ``benchmarks/primitives.py``.

        Example:
            >>> idx = SparseCosineIndex([{"a": 1.0}, {"b": 1.0}])
            >>> [(d, round(s, 3)) for d, s in idx.top_k_threshold({"a": 1.0}, k=2)]
            [(0, 1.0), (1, 0.0)]

        Args:
            q: the query, as a term to score dict, L2 norm 1.
            k: how many documents to return.

        Returns:
            A list of (doc_index, score), highest first, k of them.
        """
        self._build_inverted()
        qn = float(np.sqrt(sum(float(v) * float(v) for v in q.values())))
        if qn > 0.0:
            q = {term: float(v) / qn for term, v in q.items()}
        rows = []
        for term, v in q.items():
            col = self.vocabulary_.get(term)
            if col is None:
                continue
            docs, weights = self._inv.get(col, ((), ()))
            if len(docs):
                rows.append([float(v), docs, weights, 0])
        rows.sort(key=lambda r: len(r[1]))

        partial = np.zeros(self.n_documents_, dtype=np.float32)
        touched = np.zeros(self.n_documents_, dtype=bool)
        kth = 0.0
        have_k = False
        while rows:
            threshold = 0.0
            for v, docs, weights, pos in rows:
                if pos < len(weights):
                    threshold += v * float(weights[pos])
            if have_k and threshold < kth:
                break
            progressed = False
            for row in rows:
                v, docs, weights, pos = row
                if pos >= len(weights):
                    continue
                d = int(docs[pos])
                partial[d] += v * float(weights[pos])
                touched[d] = True
                row[3] = pos + 1
                progressed = True
            if not progressed:
                break
            idx = np.flatnonzero(touched)
            if idx.size >= k:
                scores = partial[idx]
                kth = float(np.partition(scores, -k)[-k])
                have_k = True
        idx = np.flatnonzero(touched)
        top: list[tuple[int, float]] = []
        if idx.size:
            q_vec = np.zeros(self.matrix_.shape[1], dtype=np.float32)
            for term, v in q.items():
                col = self.vocabulary_.get(term)
                if col is not None:
                    q_vec[col] = v
            exact = np.asarray(self.matrix_[idx] @ q_vec).ravel()
            kk = min(k, idx.size)
            pick = np.argpartition(-exact, kk - 1)[:kk]
            order = pick[np.argsort(-exact[pick])]
            top = [(int(idx[i]), float(exact[i])) for i in order]
        if len(top) < k:
            have = {d for d, _ in top}
            for d in range(self.n_documents_):
                if d not in have:
                    top.append((d, 0.0))
                    if len(top) == k:
                        break
        return top[:k]

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
