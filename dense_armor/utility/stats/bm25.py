"""BM25 index, batch and online, for passage retrieval.

The score of a document ``i`` for a query ``q`` is the ``Score(q, i)``
of Silajev (2026, arXiv:2609.14016, page 4): the sum over the query
terms of ``q(t)/|q|`` times the term score ``BM25(t, i)/(k1 + 1)``.
The term score is the one of page 1, with the ``+1`` correction in
the IDF:

    BM25(t, i) = d_i(t)(k1+1) / (d_i(t) + k1(1-b+b*|d_i|/avgdl)) * IDF(t)
    IDF(t)     = ln((N - DF(t) + 0.5) / (DF(t) + 0.5) + 1)

where ``d_i(t)`` is the count of ``t`` in document ``i``, ``|d_i|`` its
length, ``avgdl`` the average document length, ``N`` the number of
documents and ``DF(t)`` the number of documents that contain ``t``.

The paper gives ``k1`` in ``[0, inf)`` and ``b`` in ``[0, 1]`` and no
default for either (page 1). The defaults ``k1=1.2`` and ``b=0.75``
are the common choice in the BM25 literature and are "my choice",
not a value of the paper. The original IDF without the ``+1`` (page 4)
is discussed by the paper and is not used here.

The class keeps the state of the index: one ``{term: count}`` dict per
document, the document frequencies, the lengths, the sum of the
lengths and the number of documents. ``learn_one`` updates the state
one document at a time, ``fit`` builds the same state from a list, and
the two end in the same state on the same documents (the tests pin
this).

The class is an index, not an estimator: it ranks with ``query`` and
``top_k``, and it is not built on ``dense_armor.roles.Root``. The
``state_dict``/``load_state_dict`` pair writes JSON, never ``pickle``.

Reference: Silajev, I. (2026). *TF-IDF and BM25 Are Exact KL
Divergences.* arXiv:2609.14016. Page 1: the BM25 term score and the
``+1`` correction in the IDF; page 4: the ``Score(q, i)`` that sums
``q(t)/|q|`` times ``BM25(t, i)/(k1 + 1)``, and the original IDF
without the ``+1`` (not used here).

Example:
    >>> from dense_armor.utility.stats.bm25 import BM25Index
    >>> idx = BM25Index().fit(["a a b", "a c", "b b b"])
    >>> idx.n_docs_
    3
    >>> scores = idx.query("a b")
    >>> len(scores)
    3
    >>> bool(scores[0] > scores[2] > scores[1])
    True
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from dense_armor.utility.preprocessing.text import Tokenizer


class BM25Index:
    """Batch and online BM25 index, with the ``+1`` IDF of the paper.

    Args:
        k1: term frequency saturation, in ``[0, inf)``. The paper gives
            no default; ``1.2`` is "my choice".
        b: length normalisation, in ``[0, 1]``. The paper gives no
            default; ``0.75`` is "my choice".
        stop_words: collection of words removed before the terms are
            counted, or ``None`` (default). Forwarded to
            :class:`~dense_armor.utility.preprocessing.text.Tokenizer`.

    Raises:
        ValueError: if ``k1 < 0``, if ``b`` is outside ``[0, 1]``, or if
            ``stop_words`` is a string (the library has no built-in list
            to resolve it to).

    Example:
        >>> from dense_armor.utility.stats.bm25 import BM25Index
        >>> idx = BM25Index().fit(["a a b", "a c", "b b b"])
        >>> [round(float(s), 4) for s in idx.query("a b")]
        [0.2435, 0.119, 0.1635]
        >>> idx.top_k("a b", k=2)[0][0]
        0
    """

    budget_s: float = 5e-3
    memory_class: str = "O(n_docs * avg_terms_per_doc)"

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        stop_words: Iterable[str] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if isinstance(stop_words, str):
            raise ValueError(
                "stop_words is a collection of words; the library has no "
                "built-in list, so pass e.g. ['the', 'a'] or the list of "
                "another library. A string such as 'english' cannot be "
                "resolved here.")
        self.k1 = float(k1)
        self.b = float(b)
        self.stop_words = (
            tuple(stop_words) if stop_words is not None else None)
        self._tokenizer = Tokenizer(ngrams=1, stop_words=self.stop_words)
        self.n_docs_ = 0
        self.document_frequencies_: dict[str, int] = {}
        self.term_counts_: list[dict[str, int]] = []
        self.lengths_: list[int] = []
        self.sum_length_ = 0

    def _tokenize(self, x: Any) -> list[str]:
        """Token list of one input, through the internal Tokenizer."""
        if isinstance(x, str):
            d: dict = {"text": x}
        elif hasattr(x, "to_dict"):
            d = dict(x.to_dict())
        else:
            d = dict(x)
        return list(self._tokenizer.transform_one(d)["tokens"])

    def learn_one(self, x: Any) -> BM25Index:
        """Add one document to the index.

        Args:
            x: the document, as a string or as a dict with a ``text``
                key.

        Returns:
            ``self``.
        """
        tokens = self._tokenize(x)
        counts = Counter(tokens)
        length = len(tokens)
        for term in counts:
            self.document_frequencies_[term] = (
                self.document_frequencies_.get(term, 0) + 1)
        self.term_counts_.append(dict(counts))
        self.lengths_.append(length)
        self.sum_length_ += length
        self.n_docs_ += 1
        return self

    def fit(self, documents: Iterable[Any]) -> BM25Index:
        """Build the index from a list of documents.

        The state is reset first, so the batch ``fit`` and the online
        ``learn_one`` repeated on the same documents end in the same
        state.

        Args:
            documents: one document per item, as strings or as dicts
                with a ``text`` key.

        Returns:
            ``self``.
        """
        self.n_docs_ = 0
        self.document_frequencies_ = {}
        self.term_counts_ = []
        self.lengths_ = []
        self.sum_length_ = 0
        for doc in documents:
            self.learn_one(doc)
        return self

    def query(self, x: Any) -> np.ndarray:
        """Score of every document for the query ``x``.

        Args:
            x: the query, as a string or as a dict with a ``text`` key.

        Returns:
            A numpy array of ``n_docs_`` scores, one per document, in
            the order the documents were added. A query with no known
            term gives zeros, never ``nan``.
        """
        tokens = self._tokenize(x)
        if self.n_docs_ == 0:
            return np.zeros(0, dtype=float)
        if not tokens or self.sum_length_ == 0:
            return np.zeros(self.n_docs_, dtype=float)
        q_counts = Counter(tokens)
        q_len = len(tokens)
        avgdl = self.sum_length_ / self.n_docs_
        scores = np.zeros(self.n_docs_, dtype=float)
        for term, qf in q_counts.items():
            df = self.document_frequencies_.get(term, 0)
            if df == 0:
                continue
            idf = math.log(
                (self.n_docs_ - df + 0.5) / (df + 0.5) + 1.0)
            q_weight = qf / q_len
            for i, counts in enumerate(self.term_counts_):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                norm = 1.0 - self.b + self.b * self.lengths_[i] / avgdl
                scores[i] += q_weight * (tf / (tf + self.k1 * norm)) * idf
        return scores

    def top_k(self, x: Any, k: int = 5) -> list[tuple[int, float]]:
        """The ``k`` best documents for the query ``x``.

        Ties are broken by the lower document index.

        Args:
            x: the query, as a string or as a dict with a ``text`` key.
            k: number of documents to return. Default 5.

        Returns:
            A list of ``(index, score)`` pairs, best first, with
            ``len == min(k, n_docs_)``.

        Raises:
            ValueError: if ``k < 1``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scores = self.query(x)
        order = np.lexsort((np.arange(len(scores)), -scores))
        n = min(k, len(scores))
        return [(int(i), float(scores[i])) for i in order[:n]]

    def state_dict(self) -> dict[str, Any]:
        """JSON-serialisable state of the index.

        Returns:
            A dict with the hyper-parameters and the index state, with
            only built-in Python types (no numpy arrays, no ``pickle``).
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "stop_words": (
                list(self.stop_words) if self.stop_words is not None
                else None),
            "n_docs_": self.n_docs_,
            "document_frequencies_": dict(self.document_frequencies_),
            "term_counts_": [dict(c) for c in self.term_counts_],
            "lengths_": list(self.lengths_),
            "sum_length_": self.sum_length_,
        }

    def load_state_dict(self, state: dict[str, Any]) -> BM25Index:
        """Restore the state from :meth:`state_dict`.

        Args:
            state: the dict returned by :meth:`state_dict`.

        Returns:
            ``self``.
        """
        self.k1 = float(state["k1"])
        self.b = float(state["b"])
        sw = state["stop_words"]
        self.stop_words = tuple(sw) if sw is not None else None
        self._tokenizer = Tokenizer(ngrams=1, stop_words=self.stop_words)
        self.n_docs_ = int(state["n_docs_"])
        self.document_frequencies_ = dict(state["document_frequencies_"])
        self.term_counts_ = [dict(c) for c in state["term_counts_"]]
        self.lengths_ = list(state["lengths_"])
        self.sum_length_ = int(state["sum_length_"])
        return self
