"""Linear models fitted on the whole table.

References:
    Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Eq. (3.50)
        to (3.53), pages 153-154: the posterior of the weights of a Bayesian
        linear regression with a Gaussian prior (not in the folder,
        bibliographic only).
    Ridge regression and logistic regression by Newton steps (iteratively
        reweighted least squares) are textbook definitions.
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

STOPPED = {
    "PayloadEstimator": "the reference estimates the payload with the dynamics model of the robot; the interface on a table of regressors and residuals is not in it",
}

__all__ = ["LinearRegression", "LogisticRegression", "BayesianLinearRegression"]


def _design(X, intercept: bool) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    A = A[:, None] if A.ndim == 1 else A
    return np.hstack([np.ones((A.shape[0], 1)), A]) if intercept else A


class LinearRegression(MLBase):
    """Ridge regression in closed form.

    Minimises ``mean((X w - y)^2) + (l2 / 2) ||w||^2`` over the augmented weight
    vector (the intercept is penalised, as in the loss of the online model), so
    ``(X^T X + n l2 / 2 I) w = X^T y``.

    Args:
        l2: strength of the penalty, 0 for none.
        intercept: prepend a constant 1 to the features.

    Example:
        >>> m = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
        >>> m.coef_.round(6).tolist(), round(m.intercept_, 6)
        ([2.0], 1.0)
    """

    def __init__(self, l2: float = 0.0, intercept: bool = True) -> None:
        self.l2 = l2
        self.intercept = intercept

    def fit(self, X, y=None):  # type: ignore[override]
        A = _design(X, self.intercept)
        t = np.asarray(y, dtype=float)
        n, d = A.shape
        w = np.linalg.solve(A.T @ A + (n * self.l2 / 2.0) * np.eye(d), A.T @ t)
        self.intercept_ = float(w[0]) if self.intercept else 0.0
        self.coef_ = w[1:] if self.intercept else w
        return self

    def predict(self, X) -> np.ndarray:
        """``X coef + intercept`` for every row."""
        A = np.asarray(X, dtype=float)
        A = A[:, None] if A.ndim == 1 else A
        return A @ self.coef_ + self.intercept_


class LogisticRegression(MLBase):
    """Binary logistic regression by Newton steps on the mean log loss.

    Minimises ``mean(log loss) + (l2 / 2) ||w||^2`` over the augmented weight vector.

    Args:
        l2: strength of the penalty, strictly positive keeps the problem well posed.
        intercept: prepend a constant 1 to the features.
        max_iter: largest number of Newton steps.
        tol: stop when the largest change of a weight is below ``tol``.

    Example:
        >>> m = LogisticRegression(l2=0.01).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        >>> m.predict([[0.0], [3.0]]).tolist()
        [0, 1]
    """

    def __init__(
        self,
        l2: float = 1e-6,
        intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        self.l2 = l2
        self.intercept = intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):  # type: ignore[override]
        A = _design(X, self.intercept)
        t = np.asarray(y, dtype=float)
        n, d = A.shape
        w = np.zeros(d)
        for i in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-(A @ w)))
            g = A.T @ (p - t) / n + self.l2 * w
            H = (A.T * (p * (1 - p))) @ A / n + self.l2 * np.eye(d)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < self.tol:
                break
        self.n_iter_ = i + 1
        self.intercept_ = float(w[0]) if self.intercept else 0.0
        self.coef_ = w[1:] if self.intercept else w
        return self

    def predict_proba(self, X) -> np.ndarray:
        """``P(y = 1 | x)`` for every row."""
        A = np.asarray(X, dtype=float)
        A = A[:, None] if A.ndim == 1 else A
        return 1.0 / (1.0 + np.exp(-(A @ self.coef_ + self.intercept_)))

    def predict(self, X) -> np.ndarray:
        """The class with probability at least 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)


class BayesianLinearRegression(MLBase):
    """Bayesian linear regression with a Gaussian prior, in closed form.

    The posterior of the weights is ``N(m, S)`` with
    ``S^-1 = alpha I + beta Phi^T Phi`` and ``m = beta S Phi^T t`` (Eq. 3.53, 3.54
    of Bishop 2006); the predictive variance is ``1 / beta + phi^T S phi``.

    Args:
        alpha: prior precision of the weights.
        beta: noise precision of the observations.
        intercept: prepend a constant 1 to the features.

    Example:
        >>> m = BayesianLinearRegression(alpha=1e-6, beta=100.0).fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
        >>> round(float(m.predict([[3.0]])[0]), 3)
        7.0
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 1.0, intercept: bool = True
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.intercept = intercept

    def fit(self, X, y=None):  # type: ignore[override]
        A = _design(X, self.intercept)
        t = np.asarray(y, dtype=float)
        d = A.shape[1]
        self.cov_ = np.linalg.inv(self.alpha * np.eye(d) + self.beta * A.T @ A)
        self.mean_ = self.beta * self.cov_ @ A.T @ t
        return self

    def predict(self, X, return_std: bool = False):
        """Posterior mean of every row, and the predictive standard deviation if asked."""
        A = _design(X, self.intercept)
        mu = A @ self.mean_
        if not return_std:
            return mu
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", A, self.cov_, A)
        return mu, np.sqrt(var)
