"""Online text features: tokenization, bag of words, TF-IDF, hashing.

All four are :class:`~dense_armor.roles.Transformer` subclasses: they
read a string (or a token list) from one key and write their result
under another. Input is a plain ``str`` or a dict holding it.
"""
import hashlib
import math
import re
from collections import Counter
from typing import Any

from dense_armor.roles import Transformer

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    if isinstance(x, str):
        return {"text": x}
    return dict(x)


class Tokenizer(Transformer):
    """Lower-case word tokens, with optional n-grams.

    Memory is constant; no state is kept between calls.

    Args:
        lowercase: convert the text to lower case.
        ngrams: maximum n-gram order (1 = unigrams only).
        text_key: dict key holding the text string.
        tokens_key: dict key written with the token list.

    Examples:
        >>> from dense_armor.utility.preprocessing.text import Tokenizer
        >>> tk = Tokenizer(ngrams=2)
        >>> tk.transform_one({"text": "Hello world"})["tokens"]
        ['hello', 'world', 'hello world']
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self,
        lowercase: bool = True,
        ngrams: int = 1,
        text_key: str = "text",
        tokens_key: str = "tokens",
    ) -> None:
        self.lowercase = lowercase
        self.ngrams = ngrams
        self.text_key = text_key
        self.tokens_key = tokens_key

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "Tokenizer":
        """No state to update; kept for the interface."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Split the text into words (and n-grams) and lower-case them."""
        d = _as_dict(x)
        text = d.get(self.text_key, "")
        if not isinstance(text, str):
            text = str(text)
        if self.lowercase:
            text = text.lower()
        words = _TOKEN_RE.findall(text)
        tokens = list(words)
        for n in range(2, self.ngrams + 1):
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i : i + n]))
        out = dict(d)
        out[self.tokens_key] = tokens
        return out


class BagOfWords(Transformer):
    """Term counts of one document, from a token list.

    Memory grows with the number of distinct tokens in the document
    only; the counts dict is rebuilt on every call.

    Args:
        tokens_key: dict key holding the token list.
        counts_key: dict key written with the ``{term: count}`` dict.

    Examples:
        >>> from dense_armor.utility.preprocessing.text import BagOfWords
        >>> bw = BagOfWords()
        >>> bw.transform_one({"tokens": ["a", "b", "a"]})["counts"]
        {'a': 2, 'b': 1}
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, tokens_key: str = "tokens", counts_key: str = "counts"
    ) -> None:
        self.tokens_key = tokens_key
        self.counts_key = counts_key

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "BagOfWords":
        """No state to update; kept for the interface."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return the term counts of the document."""
        d = _as_dict(x)
        tokens = d.get(self.tokens_key, [])
        out = dict(d)
        out[self.counts_key] = dict(Counter(tokens))
        return out


class TFIDF(Transformer):
    """Online TF-IDF following Silajev 2026, section 1.

    The paper defines, for a term ``t`` in document ``i`` with raw
    counts ``d_i`` and length ``|d_i|``:

    .. math::

        \\mathrm{TF\\!\\cdot\\!IDF}(t, i)
        = \\frac{d_i(t)}{|d_i|} \\ln\\!\\left(\\frac{N}{\\mathrm{DF}(t)}\\right)

    where ``N`` is the number of documents seen so far and
    ``\\mathrm{DF}(t)`` is the number of documents containing ``t``.
    The term frequency is the relative frequency in the document
    (count divided by document length), and the idf is ``ln(N / DF)``
    with no ``+1`` smoothing. A term present in every document has
    ``DF(t) = N``, so its idf is ``ln(1) = 0`` and the term vanishes
    from the score, exactly as the paper states.

    ``smooth_idf=True`` replaces the idf with ``ln(1 + N / DF)``; the
    smoothed variant is not the paper's formula and is provided only
    to avoid zero scores for terms that appear in every document.

    Memory grows with the vocabulary: one counter per distinct term
    seen, so the total is ``O(V)`` for ``V`` distinct terms.

    Args:
        smooth_idf: if True, use ``ln(1 + N / DF)``. Default False,
            which is the paper's formula.
        tokens_key: dict key holding the token list.
        counts_key: optional dict key holding a precomputed count
            dict; when present it is used instead of recounting.
        scores_key: dict key written with ``{term: score}``.

    Examples:
        >>> from dense_armor.utility.preprocessing.text import TFIDF
        >>> tf = TFIDF()
        >>> _ = tf.learn_one({"tokens": ["a", "b"]})
        >>> _ = tf.learn_one({"tokens": ["a"]})
        >>> out = tf.transform_one({"tokens": ["a", "b"]})
        >>> out["scores"]["a"]
        0.0
        >>> round(out["scores"]["b"], 6)
        0.346574

    References:
        Silajev, I. (2026). TF-IDF and BM25 as Kullback-Leibler
            divergences. arXiv:2609.14016, section 1.
    """

    budget_s = 1e-4
    memory_class = "O(window)"

    def __init__(
        self,
        smooth_idf: bool = False,
        tokens_key: str = "tokens",
        counts_key: str = "counts",
        scores_key: str = "scores",
    ) -> None:
        self.smooth_idf = smooth_idf
        self.tokens_key = tokens_key
        self.counts_key = counts_key
        self.scores_key = scores_key
        self.n_docs_: int = 0
        self.df_: dict[str, int] = {}

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "TFIDF":
        """Count one document towards ``N`` and ``DF``."""
        self._time_step(t)
        d = _as_dict(x)
        if self.counts_key in d:
            terms = set(d[self.counts_key].keys())
        else:
            terms = set(d.get(self.tokens_key, []))
        self.n_docs_ += 1
        for term in terms:
            self.df_[term] = self.df_.get(term, 0) + 1
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return the TF-IDF scores of the current document."""
        d = _as_dict(x)
        if self.counts_key in d:
            counts = dict(d[self.counts_key])
        else:
            counts = dict(Counter(d.get(self.tokens_key, [])))
        doc_len = sum(counts.values())
        n = max(self.n_docs_, 1)
        scores: dict[str, float] = {}
        for term, c in counts.items():
            tf = (c / doc_len) if doc_len > 0 else 0.0
            df = self.df_.get(term, 1)
            if self.smooth_idf:
                idf = math.log(1.0 + n / df)
            else:
                idf = math.log(n / df)
            scores[term] = float(tf) * idf
        out = dict(d)
        out[self.scores_key] = scores
        return out


def _hash_token(
    token: str, seed: int, n_features: int
) -> tuple[int, int]:
    """Return ``(index, sign)`` for a token, both derived from the seed."""
    digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    idx = int.from_bytes(digest[:8], "big") % n_features
    sign = 1 if (digest[8] & 1) == 0 else -1
    return idx, sign


class FeatureHasher(Transformer):
    """Signed feature hashing (Weinberger et al. 2009, Theorem 3).

    For each term ``j`` the hasher draws an index ``h(j)`` in
    ``{1, ..., m}`` and a sign ``xi(j)`` in ``{-1, +1}`` from one
    SHA-256 digest. The hashed feature vector is

    .. math::

        \\phi_i(x) = \\sum_{j : h(j) = i} \\xi(j) x_j.

    The complete statement of Theorem 3 (equation 4 of the paper) is
    the following. Let ``epsilon < 1`` be a fixed constant and let
    ``x`` be an instance with ``||x||_2 = 1``. If

    .. math::

        m \\geq \\frac{72 \\log(1/\\delta)}{\\epsilon^2}
        \\quad\\text{and}\\quad
        \\|x\\|_\\infty \\leq
        \\frac{\\epsilon}{18\\sqrt{\\log(1/\\delta)\\log(m/\\delta)}},

    then

    .. math::

        \\Pr\\!\\left[\\,
        \\bigl|\\,\\|x\\|_\\phi^2 - 1\\,\\bigr| \\geq \\epsilon
        \\,\\right] \\leq 2\\delta.

    The bound is what sizes ``n_features``: for a target distortion
    ``epsilon`` and a failure probability ``2\\delta``, ``m`` must grow
    like ``log(1/\\delta) / \\epsilon^2`` and the largest coordinate of
    ``x`` must stay below the stated threshold.

    Memory is fixed by the hash space: one accumulator slot per bucket
    in the current document, so the total is ``O(m)``.

    Args:
        n_features: hash space size ``m``.
        seed: hash seed; different seeds give independent hashes.
        counts_key: dict key holding the term counts or scores.
        hashed_key: dict key written with the hashed vector as a dict.

    Examples:
        >>> from dense_armor.utility.preprocessing.text import FeatureHasher
        >>> fh = FeatureHasher(n_features=8, seed=0)
        >>> out = fh.transform_one({"counts": {"a": 1.0}})
        >>> len(out["hashed"])
        1

    References:
        Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J.,
        Smola, A. (2009). Feature hashing for large scale multitask
        learning. ICML, Theorem 3 and equation 4.
    """

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self,
        n_features: int = 1024,
        seed: int = 0,
        counts_key: str = "counts",
        hashed_key: str = "hashed",
    ) -> None:
        self.n_features = n_features
        self.seed = seed
        self.counts_key = counts_key
        self.hashed_key = hashed_key

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "FeatureHasher":
        """No state to update; kept for the interface."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Hash the term weights into a lower-dimensional sparse vector."""
        d = _as_dict(x)
        source = d.get(self.counts_key, {})
        acc: dict[int, float] = {}
        for term, weight in source.items():
            idx, sign = _hash_token(str(term), self.seed, self.n_features)
            acc[idx] = acc.get(idx, 0.0) + sign * float(weight)
        out = dict(d)
        out[self.hashed_key] = acc
        return out
