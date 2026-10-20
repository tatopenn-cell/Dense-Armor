"""Text vectorizer: a fixed vocabulary, TF-IDF, and batch or online fit.

The vectorizer chains :class:`Tokenizer` and :class:`TFIDF` and fixes
the vocabulary: after ``fit`` or a sequence of ``learn_one`` calls, a
term that was not in the vocabulary gets no score, whatever the query.
The batch form ``fit`` and the online form ``learn_one`` end in the same
state on the same documents, which the tests pin.

Reference: Silajev, I. (2026). *TF-IDF and BM25 as Kullback-Leibler
divergences.* arXiv:2609.14016, section 1, for the TF-IDF score (the
one of :class:`TFIDF`); Rennie et al. (2003), pages 6-7, for the length
normalisation (the ``norm`` of :class:`TFIDF`). The vocabulary selection
with ``max_features`` is a choice of this implementation, and the tie
rule (alphabetical order) is stated in the docstring of :meth:`fit`.
"""

from __future__ import annotations

from collections.abc import Iterable

from dense_armor.roles import Transformer
from dense_armor.utility.preprocessing.text import TFIDF, Tokenizer


def _as_text(x) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    if isinstance(x, dict):
        return str(x.get("text", ""))
    return str(x)


class TextVectorizer(Transformer):
    """TF-IDF vectorizer with a fixed vocabulary, batch or online.

    Args:
        stop_words: collection of words removed before the n-grams, or
            ``None`` (default). Forwarded to :class:`Tokenizer`.
        ngram_range: ``(min_n, max_n)`` for the n-grams. Default
            ``(1, 2)``, unigrams and bigrams.
        max_features: keep at most this many terms, the ones with the
            highest document frequency; ties broken by alphabetical
            order. ``None`` (default) keeps every term.
        norm: ``None`` or ``"l2"``, forwarded to :class:`TFIDF`. Default
            ``"l2"``.
        smooth_idf: forwarded to :class:`TFIDF`. Default ``False``.

    Raises:
        ValueError: on a bad ``ngram_range``, a negative ``max_features``,
            or a bad ``norm``.
    """

    budget_s = 1.5e-3
    memory_class = "O(vocabulary)"

    def __init__(
        self,
        stop_words: Iterable[str] | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = None,
        norm: str | None = "l2",
        smooth_idf: bool = False,
    ) -> None:
        if (len(ngram_range) != 2 or ngram_range[0] < 1
                or ngram_range[1] < ngram_range[0]):
            raise ValueError(
                f"ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n, "
                f"got {ngram_range!r}")
        if max_features is not None and max_features < 1:
            raise ValueError(
                f"max_features must be >= 1 or None, got {max_features}")
        if norm not in (None, "l2"):
            raise ValueError(f"norm must be None or 'l2', got {norm!r}")
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.tokenizer_ = Tokenizer(
            ngrams=ngram_range[1], stop_words=stop_words)
        self.tfidf_ = TFIDF(smooth_idf=smooth_idf, norm=norm)
        self._fitted = False

    def _learn_doc(self, text: str) -> None:
        tokens = self.tokenizer_.transform_one({"text": text})["tokens"]
        self.tfidf_.learn_one({"tokens": tokens})

    def _scores(self, text: str) -> dict:
        tokens = self.tokenizer_.transform_one({"text": text})["tokens"]
        scores = self.tfidf_.transform_one({"tokens": tokens})["scores"]
        vocab = self.tfidf_.df_
        return {k: v for k, v in scores.items() if k in vocab}

    def learn_one(self, x, y=None, t: float | None = None) -> TextVectorizer:
        """Count one document towards the vocabulary and the document
        frequencies."""
        self._time_step(t)
        if not self._fitted:
            self._fitted = True
        self._learn_doc(_as_text(x))
        return self

    def transform_one(self, x, t: float | None = None) -> dict:
        """Return ``{"scores": {term: score}}`` for one document."""
        d = x if isinstance(x, dict) else {"text": _as_text(x)}
        out = dict(d)
        out["scores"] = self._scores(_as_text(x))
        return out

    def _apply_max_features(self) -> None:
        if self.max_features is None:
            return
        df = self.tfidf_.df_
        if len(df) <= self.max_features:
            return
        ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = {term for term, _ in ranked[: self.max_features]}
        self.tfidf_.df_ = {term: df[term] for term in keep}

    def fit(self, documents: Iterable) -> TextVectorizer:
        """Fit the state from a list of documents.

        The state after ``fit`` equals the state after the same documents
        given one at a time to ``learn_one``. When ``max_features`` is
        set, the terms kept are the ones with the highest document
        frequency; ties are broken by alphabetical order, so the result
        does not depend on the order of the documents.
        """
        for doc in documents:
            self._learn_doc(_as_text(doc))
            self._fitted = True
        self._apply_max_features()
        return self

    def transform(self, documents: Iterable) -> list[dict]:
        """Return one ``{"scores": {term: score}}`` dict per document."""
        return [self.transform_one(doc) for doc in documents]

    def state_dict(self) -> dict:
        """JSON-serialisable state: the document frequencies and ``N``."""
        return {
            "n_docs_": int(self.tfidf_.n_docs_),
            "df_": {str(k): int(v) for k, v in self.tfidf_.df_.items()},
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "norm": self.norm,
            "smooth_idf": self.smooth_idf,
        }

    def load_state_dict(self, state: dict) -> TextVectorizer:
        """Load a state written by :meth:`state_dict`."""
        self.tfidf_.n_docs_ = int(state["n_docs_"])
        self.tfidf_.df_ = {str(k): int(v) for k, v in state["df_"].items()}
        self.max_features = state["max_features"]
        self.ngram_range = tuple(state["ngram_range"])
        self.norm = state["norm"]
        self.smooth_idf = state["smooth_idf"]
        self.tokenizer_ = Tokenizer(
            ngrams=self.ngram_range[1], stop_words=self.stop_words)
        self.tfidf_ = TFIDF(
            smooth_idf=self.smooth_idf, norm=self.norm,
            tokens_key="tokens", counts_key="counts", scores_key="scores")
        self.tfidf_.n_docs_ = int(state["n_docs_"])
        self.tfidf_.df_ = {str(k): int(v) for k, v in state["df_"].items()}
        self._fitted = True
        return self
