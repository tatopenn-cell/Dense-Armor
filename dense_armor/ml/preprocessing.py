"""Preprocessing: scalers, selectors, joint derivatives and text on the whole table.

``fit`` reads the whole table (rows are samples, columns are features) and
``transform`` applies the result to any table, so the scaler is the
textbook formula and the online scaler is the same formula with running
statistics.

References:
    Standardisation, min-max scaling, variance threshold, correlation
    ranking and finite differences are textbook definitions.
    Silajev (2026), TF-IDF and BM25 as exact KL divergences, section 1:
        ``tf = count / length`` and ``idf = ln(N / DF)``, or
        ``ln(1 + N / DF)`` when smoothed (not in the folder,
        bibliographic only).
    Weinberger, K. et al. (2009). Feature hashing for large scale multitask
        learning, Theorem 3: the signed hashing trick (not in the folder,
        bibliographic only).
    Malialis, K. et al. (2022). Queue-based resampling for online class
        imbalance learning: a sequential method over a stream, so
        ``QueueResampler`` has no class (not in the folder,
        bibliographic only).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np
from scipy.stats import pearsonr  # noqa: F401  (kept for the docstring of SelectKBest)

from dense_armor.ml.base import MLBase
from dense_armor.ml.stats import EWStats

__all__ = [
    "StandardScaler",
    "EWScaler",
    "RobustScaler",
    "MinMaxScaler",
    "VarianceThreshold",
    "SelectKBest",
    "JointDerivatives",
    "JointPower",
    "Tokenizer",
    "BagOfWords",
    "TFIDF",
    "FeatureHasher",
    "TextVectorizer",
]

STOPPED = {
    "QueueResampler": "the reference describes a queue over a stream: it holds no algorithm for the whole dataset",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _table(X) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    return A[:, None] if A.ndim == 1 else A


class _Scaler(MLBase):
    """Fit a centre and a scale per column; ``transform`` is ``(x - c) / s``."""

    def __init__(self, features: list[str] | None = None, eps: float = 1e-12) -> None:
        self.features = features
        self.eps = eps

    def _centre_scale(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit(self, X, y=None):  # type: ignore[override]
        A = _table(X)
        self.center_, self.scale_ = self._centre_scale(A)
        self.n_samples_ = int(A.shape[0])
        return self

    def transform(self, X) -> np.ndarray:
        """Apply the fitted centre and scale; columns with a scale below ``eps`` pass through."""
        A = _table(X)
        ok = np.isfinite(self.scale_) & (self.scale_ > self.eps)
        out: np.ndarray = A.astype(float).copy()
        out[:, ok] = (A[:, ok] - self.center_[ok]) / self.scale_[ok]
        return out

    def fit_transform(self, X) -> np.ndarray:
        """``fit`` then ``transform`` on the same table."""
        return self.fit(X).transform(X)


class StandardScaler(_Scaler):
    """Standardise with the mean and the standard deviation (``n - 1``) of the table.

    Example:
        >>> StandardScaler().fit_transform([[1.0], [3.0]]).round(4).tolist()
        [[-0.7071], [0.7071]]
    """

    def _centre_scale(self, A):
        if A.shape[0] < 2:
            return A.mean(axis=0), np.full(A.shape[1], np.nan)
        return A.mean(axis=0), A.std(axis=0, ddof=1)


class EWScaler(_Scaler):
    """Standardise with the exponentially weighted mean and standard deviation.

    Args:
        alpha: weight of the new sample, in ``(0, 1]``.
        features: names of the columns (kept for the online pair).
        eps: scale floor below which a column passes through.

    Example:
        >>> EWScaler(alpha=0.5).fit([[1.0], [3.0]]).center_.tolist()
        [2.0]
    """

    def __init__(
        self, alpha: float = 0.1, features: list[str] | None = None, eps: float = 1e-12
    ) -> None:
        super().__init__(features, eps)
        self.alpha = alpha

    def _centre_scale(self, A):
        cols = [EWStats(alpha=self.alpha).fit(A[:, j]) for j in range(A.shape[1])]
        return np.array([c.mean_ for c in cols]), np.array([c.std_ for c in cols])


class RobustScaler(_Scaler):
    """Centre on the median and scale by the MAD of the last ``window`` rows.

    The scale is ``1.4826`` times the median absolute deviation, the factor
    that makes it a consistent estimator of sigma for Gaussian data
    (Hampel 1974, not in the folder, bibliographic only).

    Args:
        window: number of rows used.
        features: names of the columns (kept for the online pair).
        eps: scale floor below which a column passes through.
        scale: factor applied to the raw MAD.

    Example:
        >>> RobustScaler(window=3, scale=1.0).fit([[1.0], [2.0], [9.0]]).scale_.tolist()
        [1.0]
    """

    def __init__(
        self,
        window: int = 20,
        features: list[str] | None = None,
        eps: float = 1e-12,
        scale: float = 1.4826,
    ) -> None:
        super().__init__(features, eps)
        self.window = window
        self.scale = scale

    def _centre_scale(self, A):
        W = A[-self.window :]
        med = np.median(W, axis=0)
        return med, self.scale * np.median(np.abs(W - med), axis=0)


class MinMaxScaler(_Scaler):
    """Rescale each column to ``[0, 1]`` with the minimum and the maximum of the table.

    Example:
        >>> MinMaxScaler().fit_transform([[1.0], [3.0], [2.0]]).T.tolist()
        [[0.0, 1.0, 0.5]]
    """

    def _centre_scale(self, A):
        return A.min(axis=0), A.max(axis=0) - A.min(axis=0)


class VarianceThreshold(MLBase):
    """Drop the columns whose variance (``n - 1``) is at or below ``threshold``.

    Args:
        threshold: variance floor.
        features: names of the columns (kept for the online pair).

    Example:
        >>> VarianceThreshold().fit([[1.0, 5.0], [2.0, 5.0]]).keep_.tolist()
        [True, False]
    """

    def __init__(
        self, threshold: float = 0.0, features: list[str] | None = None
    ) -> None:
        self.threshold = threshold
        self.features = features

    def fit(self, X, y=None):  # type: ignore[override]
        A = _table(X)
        var = A.var(axis=0, ddof=1) if A.shape[0] > 1 else np.full(A.shape[1], np.nan)
        self.variance_ = var
        self.keep_ = ~(var <= self.threshold)
        return self

    def transform(self, X) -> np.ndarray:
        """The kept columns of ``X``."""
        return _table(X)[:, self.keep_]


class SelectKBest(MLBase):
    """Keep the ``k`` columns with the highest absolute Pearson correlation with ``y``.

    Args:
        k: number of columns kept.
        features: names of the columns (kept for the online pair).

    Example:
        >>> X = [[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]
        >>> SelectKBest(k=1).fit(X, [1.0, 2.0, 3.0]).selected_.tolist()
        [0]
    """

    def __init__(self, k: int = 10, features: list[str] | None = None) -> None:
        self.k = k
        self.features = features

    def fit(self, X, y=None):  # type: ignore[override]
        A = _table(X)
        t = np.asarray(y, dtype=float)
        c = np.array(
            [
                abs(np.corrcoef(A[:, j], t)[0, 1])
                if A[:, j].std() > 0 and t.std() > 0
                else 0.0
                for j in range(A.shape[1])
            ]
        )
        self.scores_ = c
        self.selected_ = np.sort(np.argsort(-c, kind="stable")[: self.k])
        return self

    def transform(self, X) -> np.ndarray:
        """The selected columns of ``X``."""
        return _table(X)[:, self.selected_]


class JointDerivatives(MLBase):
    """Velocity, acceleration and jerk from the polynomial through the last ``order + 1`` samples.

    For each row ``i`` the interpolating polynomial of the last ``order + 1``
    positions (Newton divided differences) is differentiated at ``t_i``:
    ``qd`` is its first derivative, ``qdd`` the second and ``jerk`` the
    third. It is exact for a polynomial of degree ``order``. The first rows,
    which have fewer samples, use the polynomial they have and the missing
    derivatives are zero.

    Args:
        order: number of derivatives, 1 to 3.
        q_key: name of the joint positions (kept for the online pair).

    Example:
        >>> d = JointDerivatives(order=3).fit([[0.0], [1.0], [8.0], [27.0]], [0.0, 1.0, 2.0, 3.0])
        >>> d.qd_[-1].round(6).tolist(), d.qdd_[-1].round(6).tolist(), d.jerk_[-1].round(6).tolist()
        ([27.0], [18.0], [6.0])
    """

    def __init__(self, order: int = 3, q_key: str = "q") -> None:
        self.order = order
        self.q_key = q_key

    def fit(self, X, y=None):  # type: ignore[override]
        q = _table(X)
        t = np.asarray(y, dtype=float)
        n, d = q.shape
        out = np.zeros((3, n, d))
        for i in range(n):
            k = min(self.order, i)
            if k == 0:
                continue
            tau = t[i - k : i + 1] - t[i]
            coef = np.linalg.solve(
                np.vander(tau, k + 1, increasing=True), q[i - k : i + 1]
            )
            for j in range(1, k + 1):
                out[j - 1, i] = coef[j] * math.factorial(j)
        self.qd_, self.qdd_, self.jerk_ = out
        return self


class JointPower(MLBase):
    """Mechanical power ``tau * qd`` per joint and its total, for every row.

    Args:
        tau_key: name of the torques (kept for the online pair).
        qd_key: name of the velocities (kept for the online pair).

    Example:
        >>> JointPower().fit([[2.0, 1.0]], [[3.0, 4.0]]).total_.tolist()
        [10.0]
    """

    def __init__(self, tau_key: str = "tau", qd_key: str = "qd") -> None:
        self.tau_key = tau_key
        self.qd_key = qd_key

    def fit(self, X, y=None):  # type: ignore[override]
        self.power_ = _table(X) * _table(y)
        self.total_ = self.power_.sum(axis=1)
        return self


class Tokenizer(MLBase):
    """Lower-case word tokens with optional n-grams, for a list of documents.

    Args:
        lowercase: lower-case the text first.
        ngrams: largest n-gram size.
        text_key: name of the text field (kept for the online pair).
        tokens_key: name of the token field (kept for the online pair).
        stop_words: words removed before the n-grams are formed.

    Example:
        >>> Tokenizer(ngrams=2).tokenize("The cat sat")
        ['the', 'cat', 'sat', 'the cat', 'cat sat']
    """

    def __init__(
        self,
        lowercase: bool = True,
        ngrams: int = 1,
        text_key: str = "text",
        tokens_key: str = "tokens",
        stop_words=None,
    ) -> None:
        self.lowercase = lowercase
        self.ngrams = ngrams
        self.text_key = text_key
        self.tokens_key = tokens_key
        self.stop_words = stop_words

    def tokenize(self, text) -> list[str]:
        """Tokens of one document."""
        text = text if isinstance(text, str) else str(text)
        if self.lowercase:
            text = text.lower()
        words = _TOKEN_RE.findall(text)
        if self.stop_words is not None:
            sw = {w.lower() for w in self.stop_words}
            words = [w for w in words if w not in sw]
        tokens = list(words)
        for n in range(2, self.ngrams + 1):
            tokens += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        return tokens

    def fit(self, X, y=None):  # type: ignore[override]
        self.tokens_ = [self.tokenize(d) for d in X]
        return self


class BagOfWords(MLBase):
    """Term counts of every document of a list of token lists.

    Args:
        tokens_key: name of the token field (kept for the online pair).
        counts_key: name of the counts field (kept for the online pair).

    Example:
        >>> BagOfWords().fit([["a", "b", "a"]]).counts_
        [{'a': 2, 'b': 1}]
    """

    def __init__(self, tokens_key: str = "tokens", counts_key: str = "counts") -> None:
        self.tokens_key = tokens_key
        self.counts_key = counts_key

    def fit(self, X, y=None):  # type: ignore[override]
        self.counts_ = [dict(Counter(t)) for t in X]
        return self


class TFIDF(MLBase):
    """TF-IDF of a corpus: ``tf = count / length``, ``idf = ln(N / DF)``.

    Args:
        smooth_idf: use ``ln(1 + N / DF)``.
        tokens_key: name of the token field (kept for the online pair).
        counts_key: name of the counts field (kept for the online pair).
        scores_key: name of the scores field (kept for the online pair).
        norm: ``"l2"`` divides each document by the norm of its scores.

    Example:
        >>> t = TFIDF().fit([["a", "b"], ["a", "c"]])
        >>> t.df_["a"], round(t.transform_one(["b", "b"])["b"], 4)
        (2, 0.6931)
    """

    def __init__(
        self,
        smooth_idf: bool = False,
        tokens_key: str = "tokens",
        counts_key: str = "counts",
        scores_key: str = "scores",
        norm: str | None = None,
    ) -> None:
        self.smooth_idf = smooth_idf
        self.tokens_key = tokens_key
        self.counts_key = counts_key
        self.scores_key = scores_key
        self.norm = norm

    def fit(self, X, y=None):  # type: ignore[override]
        docs = [list(d) for d in X]
        self.n_docs_ = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.df_ = dict(df)
        return self

    def transform_one(self, tokens) -> dict[str, float]:
        """TF-IDF scores of one token list against the fitted corpus."""
        counts = Counter(tokens)
        length = sum(counts.values())
        n = max(self.n_docs_, 1)
        out = {}
        for term, c in counts.items():
            tf = c / length if length else 0.0
            df = self.df_.get(term, 1)
            out[term] = tf * (
                math.log(1.0 + n / df) if self.smooth_idf else math.log(n / df)
            )
        if self.norm == "l2":
            s = math.sqrt(sum(v * v for v in out.values()))
            if s > 0:
                out = {k: v / s for k, v in out.items()}
        return out


class FeatureHasher(MLBase):
    """Signed feature hashing of a list of count dictionaries into a dense matrix.

    Args:
        n_features: number of hashed columns.
        seed: seed of the hash.
        counts_key: name of the counts field (kept for the online pair).
        hashed_key: name of the hashed field (kept for the online pair).

    Example:
        >>> FeatureHasher(n_features=8).fit_transform([{"a": 2.0}]).shape
        (1, 8)
    """

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

    def fit(self, X, y=None):  # type: ignore[override]
        return self

    def fit_transform(self, X) -> np.ndarray:
        """The hashed matrix of the documents, one row each."""
        out = np.zeros((len(X), self.n_features))
        for i, doc in enumerate(X):
            for term, w in doc.items():
                dg = hashlib.sha256(f"{self.seed}:{term}".encode()).digest()
                out[i, int.from_bytes(dg[:8], "big") % self.n_features] += (
                    1 if (dg[8] & 1) == 0 else -1
                ) * float(w)
        return out


class TextVectorizer(MLBase):
    """TF-IDF matrix of a corpus with a fixed vocabulary.

    The vocabulary keeps the ``max_features`` terms of highest document
    frequency (ties by alphabetical order), sorted alphabetically; rows are
    normalised by the L2 norm when ``norm="l2"``.

    Args:
        stop_words: words removed before the n-grams.
        ngram_range: smallest and largest n-gram size.
        max_features: size of the vocabulary, or ``None``.
        norm: ``"l2"`` or ``None``.
        smooth_idf: use ``ln(1 + N / DF)``.

    Example:
        >>> v = TextVectorizer(ngram_range=(1, 1)).fit(["a b", "a c"])
        >>> v.vocabulary_
        ['a', 'b', 'c']
    """

    def __init__(
        self,
        stop_words=None,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = None,
        norm: str | None = "l2",
        smooth_idf: bool = False,
    ) -> None:
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.norm = norm
        self.smooth_idf = smooth_idf

    def _tokens(self, doc) -> list[str]:
        lo, hi = self.ngram_range
        tok = Tokenizer(ngrams=hi, stop_words=self.stop_words).tokenize(doc)
        return [t for t in tok if lo <= t.count(" ") + 1 <= hi]

    def fit(self, X, y=None):  # type: ignore[override]
        docs = [self._tokens(d) for d in X if isinstance(d, str)]
        tf = TFIDF(smooth_idf=self.smooth_idf).fit(docs)
        terms = sorted(tf.df_, key=lambda t: (-tf.df_[t], t))
        if self.max_features is not None:
            terms = terms[: self.max_features]
        self.vocabulary_ = sorted(terms)
        self.tfidf_ = tf
        self.n_docs_ = tf.n_docs_
        return self

    def transform(self, documents) -> np.ndarray:
        """Dense TF-IDF matrix of the documents on the fitted vocabulary."""
        idx = {t: i for i, t in enumerate(self.vocabulary_)}
        M = np.zeros((len(documents), len(idx)))
        for r, doc in enumerate(documents):
            toks = self._tokens(doc)
            counts = Counter(toks)
            length = sum(counts.values())
            n = max(self.n_docs_, 1)
            for term, c in counts.items():
                if term in idx:
                    df = self.tfidf_.df_.get(term, 1)
                    idf = (
                        math.log(1.0 + n / df) if self.smooth_idf else math.log(n / df)
                    )
                    M[r, idx[term]] = (c / length) * idf
            if self.norm == "l2":
                s = np.linalg.norm(M[r])
                if s > 0:
                    M[r] /= s
        return M
