"""Naive Bayes classifiers fitted on the whole table by counting.

References:
    McCallum, A., Nigam, K. (1998). A comparison of event models for naive
        Bayes text classification, page 4 (multinomial and Bernoulli models)
        (not in the folder, bibliographic only).
    Rennie, J. D. M. et al. (2003). Tackling the poor assumptions of naive
        Bayes text classifiers, pages 6-7 (complement naive Bayes) (not in
        the folder, bibliographic only).
    Laplace smoothing with a pseudo-count ``alpha`` is the textbook form.
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["MultinomialNB", "BernoulliNB", "CategoricalNB", "ComplementNB"]


def _softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


class _NB(MLBase):
    """Shared ``predict`` and ``predict_proba`` from class log scores."""

    def _scores(self, X) -> np.ndarray:
        raise NotImplementedError

    def _prior(self, y) -> None:
        self.classes_, counts = np.unique(y, return_counts=True)
        if self.class_prior is None:
            self.log_prior_ = np.log(counts / counts.sum())
        else:
            p = np.array([self.class_prior[c] for c in self.classes_], dtype=float)
            self.log_prior_ = np.log(p / p.sum())

    def predict_proba(self, X) -> np.ndarray:
        """Posterior probability of every class, one row per sample."""
        return _softmax(self._scores(np.asarray(X, dtype=float)))

    def predict(self, X) -> np.ndarray:
        """The most probable class of every row."""
        return self.classes_[
            np.argmax(self._scores(np.asarray(X, dtype=float)), axis=1)
        ]


class MultinomialNB(_NB):
    """Multinomial naive Bayes on count features.

    ``theta_cj = (N_cj + alpha) / (N_c + alpha d)`` and the score of a class is
    ``log P(c) + sum_j x_j log theta_cj``.

    Args:
        alpha: pseudo-count of the smoothing.
        class_prior: dictionary of class priors, or ``None`` for the class frequencies.

    Example:
        >>> X = [[3, 0], [2, 1], [0, 3], [1, 2]]
        >>> MultinomialNB().fit(X, [0, 0, 1, 1]).predict([[4, 0], [0, 4]]).tolist()
        [0, 1]
    """

    def __init__(self, alpha: float = 1.0, class_prior: dict | None = None) -> None:
        self.alpha = alpha
        self.class_prior = class_prior

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self._prior(t)
        N = np.array([A[t == c].sum(axis=0) for c in self.classes_])
        theta = (N + self.alpha) / (
            N.sum(axis=1, keepdims=True) + self.alpha * A.shape[1]
        )
        self.feature_log_prob_ = np.log(theta)
        return self

    def _scores(self, X):
        return self.log_prior_ + X @ self.feature_log_prob_.T


class BernoulliNB(_NB):
    """Bernoulli naive Bayes on presence features (a value above 0 is a presence).

    ``theta_cj = (n_cj + alpha) / (n_c + 2 alpha)`` with ``n_cj`` the number of
    samples of class ``c`` where feature ``j`` is present.

    Args:
        alpha: pseudo-count of the smoothing.
        class_prior: dictionary of class priors, or ``None`` for the class frequencies.

    Example:
        >>> X = [[1, 0], [1, 1], [0, 1], [0, 1]]
        >>> BernoulliNB().fit(X, [0, 0, 1, 1]).predict([[1, 0], [0, 1]]).tolist()
        [0, 1]
    """

    def __init__(self, alpha: float = 1.0, class_prior: dict | None = None) -> None:
        self.alpha = alpha
        self.class_prior = class_prior

    def fit(self, X, y=None):  # type: ignore[override]
        B = (np.asarray(X, dtype=float) > 0).astype(float)
        t = np.asarray(y)
        self._prior(t)
        n = np.array([(t == c).sum() for c in self.classes_], dtype=float)[:, None]
        k = np.array([B[t == c].sum(axis=0) for c in self.classes_])
        theta = (k + self.alpha) / (n + 2.0 * self.alpha)
        self.log_theta_ = np.log(theta)
        self.log_1m_theta_ = np.log1p(-theta)
        return self

    def _scores(self, X):
        B = (X > 0).astype(float)
        return (
            self.log_prior_ + B @ self.log_theta_.T + (1.0 - B) @ self.log_1m_theta_.T
        )


class CategoricalNB(_NB):
    """Categorical naive Bayes: every feature takes one of a few category codes.

    ``P(x_j = k | c) = (n_cjk + alpha) / (n_c + alpha K_j)`` with ``K_j`` the
    number of categories of feature ``j`` seen in the training table; a
    category never seen gets ``alpha / (n_c + alpha K_j)``.

    Args:
        alpha: pseudo-count of the smoothing.
        class_prior: dictionary of class priors, or ``None`` for the class frequencies.

    Example:
        >>> X = [[0, 1], [0, 0], [1, 1], [1, 0]]
        >>> CategoricalNB().fit(X, [0, 0, 1, 1]).predict([[0, 1], [1, 0]]).tolist()
        [0, 1]
    """

    def __init__(self, alpha: float = 1.0, class_prior: dict | None = None) -> None:
        self.alpha = alpha
        self.class_prior = class_prior

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X)
        t = np.asarray(y)
        self._prior(t)
        self.categories_ = [np.unique(A[:, j]) for j in range(A.shape[1])]
        n = np.array([(t == c).sum() for c in self.classes_], dtype=float)
        self.log_prob_ = []
        self.log_unseen_ = []
        for j, cats in enumerate(self.categories_):
            K = len(cats)
            cnt = np.array(
                [
                    [np.sum((t == c) & (A[:, j] == k)) for k in cats]
                    for c in self.classes_
                ],
                dtype=float,
            )
            den = (n + self.alpha * K)[:, None]
            self.log_prob_.append(np.log((cnt + self.alpha) / den))
            self.log_unseen_.append(
                np.log(self.alpha / den[:, 0])
                if self.alpha > 0
                else np.full(len(n), -np.inf)
            )
        return self

    def _scores(self, X):
        s = np.tile(self.log_prior_, (X.shape[0], 1))
        for j, cats in enumerate(self.categories_):
            for r in range(X.shape[0]):
                hit = np.flatnonzero(cats == X[r, j])
                s[r] += (
                    self.log_prob_[j][:, hit[0]] if hit.size else self.log_unseen_[j]
                )
        return s


class ComplementNB(_NB):
    """Complement naive Bayes (Rennie et al. 2003): weights from the other classes.

    ``theta~_cj = (sum_{c' != c} N_c'j + alpha) / (sum_{c' != c} N_c' + alpha d)``,
    ``w_cj = log theta~_cj``, divided by ``sum_j |w_cj|`` when ``norm`` is true;
    the score of a class is ``- sum_j x_j w_cj``.

    Args:
        alpha: pseudo-count of the smoothing.
        norm: divide the weights of each class by their absolute sum.

    Example:
        >>> X = [[3, 0], [2, 1], [0, 3], [1, 2]]
        >>> ComplementNB().fit(X, [0, 0, 1, 1]).predict([[4, 0], [0, 4]]).tolist()
        [0, 1]
    """

    class_prior = None

    def __init__(self, alpha: float = 1.0, norm: bool = True) -> None:
        self.alpha = alpha
        self.norm = norm

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self._prior(t)
        N = np.array([A[t == c].sum(axis=0) for c in self.classes_])
        comp = N.sum(axis=0) - N
        theta = (comp + self.alpha) / (
            comp.sum(axis=1, keepdims=True) + self.alpha * A.shape[1]
        )
        w = np.log(theta)
        if self.norm:
            w = w / np.abs(w).sum(axis=1, keepdims=True)
        self.weights_ = w
        return self

    def _scores(self, X):
        return -(X @ self.weights_.T)
