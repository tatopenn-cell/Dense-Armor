"""Online Bayesian linear regression with a conjugate Gaussian prior.

The model keeps a Gaussian belief over the weight vector and updates it
with the closed-form posterior of the linear-Gaussian model, one sample
at a time. The mean of the belief is the prediction; the covariance
gives a predictive variance that grows away from the data.

Model, with the convention of Bishop, Pattern Recognition and Machine
Learning, Section 3.3:

    p(w)         = N(w | 0, alpha^{-1} I)             (eq. 3.48)
    p(y | x, w)  = N(y | w^T x, beta^{-1})            (eq. 3.8)
    p(w | X, y)  = N(w | m_N, Sigma_N)                (eq. 3.49)
    Sigma_N^{-1} = alpha I + beta X^T X               (eq. 3.54)
    m_N          = beta Sigma_N X^T y                 (eq. 3.53)

The online update of the covariance is the rank-one Sherman-Morrison
step, and the update of the mean has the Kalman-filter form. The two
updates coincide with eq. 3.50 and eq. 3.53 on the same data; the
equivalence is derived in the Notes of the class.

    Sigma_N      = Sigma_{N-1}
                   - beta Sigma_{N-1} x x^T Sigma_{N-1}
                     / (1 + beta x^T Sigma_{N-1} x)
    m_N          = m_{N-1} + K (y - x^T m_{N-1})
    K            = beta Sigma_{N-1} x / (1 + beta x^T Sigma_{N-1} x)

References:
    Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
    Springer. Section 3.3 (Bayesian linear regression).
    Sherman, J., Morrison, W. J. (1950). Adjustment of an inverse
    matrix corresponding to a change in one element of a given matrix.
    The Annals of Mathematical Statistics 21(1), 124-127.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Estimate, Regressor


class BayesianLinearRegression(Regressor):
    """Online Bayesian linear regression with a conjugate Gaussian prior.

    The model keeps the posterior covariance Sigma and mean m of the
    weight vector w. Prediction returns the posterior mean w^T x; the
    predictive variance 1 / beta + x^T Sigma x grows away from the
    samples seen so far, and shrinks back toward 1 / beta where the
    data is dense.

    Args:
        alpha: prior precision of the weights, strictly positive.
            Large alpha means a tight prior around zero.
        beta: noise precision of the observations, strictly positive.
            Large beta means a confident measurement model.
        intercept: if True, prepend a constant 1 to the feature vector.
        feature_keys: explicit list of feature keys. None sorts the keys
            of the first dict seen.

    Examples:
        >>> from dense_armor.utility.linear.bayes import (
        ...     BayesianLinearRegression,
        ... )
        >>> model = BayesianLinearRegression(alpha=1.0, beta=1.0)
        >>> for i in range(20):
        ...     _ = model.learn_one({"x": float(i) / 10.0}, 2.0 * i / 10.0)
        >>> est = model.predict_one({"x": 0.5}, return_estimate=True)
        >>> round(est.mean, 2)
        1.1
        >>> bool(est.var > 0.0)
        True

    Notes:
        The online update of the mean is written in the Kalman-filter
        form ``m_N = m_{N-1} + K (y - x^T m_{N-1})``. The derivation
        from Bishop eq. 3.53, with ``b_N = beta X^T y`` the accumulator
        of the precision-weighted targets and ``A_N = Sigma_N^{-1}``,

            m_N = Sigma_N b_N
                = Sigma_N (A_N m_{N-1} + beta x y)
                = m_{N-1} + Sigma_N beta x (y - x^T m_{N-1}),

        where the last step uses ``A_N = A_{N-1} + beta x x^T`` and the
        definition of ``K``. The two forms agree step by step, and the
        tests compare the streaming posterior with the closed-form batch
        posterior on the same samples.
    """

    budget_s: float = 5e-3
    memory_class: str = "O(1)"

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        intercept: bool = True,
        feature_keys: Sequence[str] | None = None,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.intercept = bool(intercept)
        self.feature_keys = feature_keys
        self._keys: list | None = None
        self._m: np.ndarray | None = None
        self._S: np.ndarray | None = None

    @staticmethod
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        to_dict = getattr(x, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        raise TypeError(
            f"expected a dict or an object with to_dict(), got {type(x).__name__}"
        )

    def _resolve_keys(self, x) -> list:
        x = self._as_dict(x)
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        return self._keys

    def _row(self, x) -> np.ndarray:
        x = self._as_dict(x)
        keys = self._resolve_keys(x)
        xv = np.asarray([float(x.get(k, 0.0)) for k in keys], dtype=float)
        if self.intercept:
            return np.concatenate([[1.0], xv])
        return xv

    def _ensure_state(self, n: int) -> None:
        if self._m is None:
            self._m = np.zeros(n, dtype=float)
            self._S = np.eye(n, dtype=float) / self.alpha

    def learn_one(
        self, x: dict, y: Any, t: float | None = None
    ) -> "BayesianLinearRegression":
        """Update the posterior with a single (x, y) sample.

        The covariance is updated with the Sherman-Morrison rank-one
        formula, the mean with the Kalman-filter form. The covariance
        is symmetrised after each step to keep numerical errors from
        breaking the symmetry the derivation assumes.

        Args:
            x: feature dict.
            y: target scalar.
            t: optional timestamp, currently unused.

        Returns:
            self.
        """
        xv = self._row(x)
        self._ensure_state(xv.size)
        Sx = self._S @ xv
        denom = 1.0 + self.beta * float(xv @ Sx)
        K = self.beta * Sx / denom
        err = float(y) - float(self._m @ xv)
        self._m = self._m + K * err
        self._S = self._S - np.outer(K, Sx)
        self._S = 0.5 * (self._S + self._S.T)
        return self

    def predict_one(
        self,
        x: dict,
        t: float | None = None,
        return_std: bool = False,
        return_estimate: bool = False,
    ) -> Any:
        """Predict the target of a single sample.

        Args:
            x: feature dict.
            t: optional timestamp, currently unused.
            return_std: if True, also return the standard deviation of
                the predictive Gaussian.
            return_estimate: if True, return an :class:`Estimate` with
                the predictive mean and variance.

        Returns:
            The predictive mean, or ``(mean, std)``, or an
            :class:`Estimate`.
        """
        xv = self._row(x)
        if self._m is None:
            n = xv.size
            m = np.zeros(n)
            S = np.eye(n) / self.alpha
        else:
            assert self._S is not None
            m = self._m
            S = self._S
        mu = float(m @ xv)
        var = 1.0 / self.beta + float(xv @ S @ xv)
        if return_estimate:
            return Estimate(mean=mu, var=var)
        if return_std:
            return mu, float(np.sqrt(var))
        return mu

    def _unit_test_skips(self) -> set:
        return set()
