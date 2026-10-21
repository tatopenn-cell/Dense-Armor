"""Optimisers that run on the whole gradient of a loss.

Each class has ``minimize(grad, w0, n_iter)``: ``grad(w)`` returns the gradient of
the loss over the whole dataset, and the update rule is the one of the paper.

References:
    Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods for
        online learning and stochastic optimization, Eq. (4) (not in the
        folder, bibliographic only).
    Kingma, D. P., Ba, J. (2015). Adam: a method for stochastic optimization,
        Algorithm 1, arXiv:1412.6980 (not in the folder, bibliographic only).
    Orabona, F. (2019). A modern introduction to online learning, Chapter 7,
        Algorithm 7.2 (FTRL with linearised losses) (not in the folder,
        bibliographic only).
    Hazan, E., Agarwal, A., Kale, S. (2007). Logarithmic regret algorithms for
        online convex optimization, Algorithm 2 (Online Newton Step) (not in
        the folder, bibliographic only).
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["GradientDescent", "AdaGrad", "Adam", "FTRLProximal", "NewtonStep"]


class _Optimizer(MLBase):
    """``minimize`` runs ``n_iter`` steps of the update rule from ``w0``."""

    def _init(self, d: int) -> None:
        raise NotImplementedError

    def _step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def minimize(self, grad, w0, n_iter: int = 100) -> np.ndarray:
        """Return the weights after ``n_iter`` steps; ``path_`` holds every iterate."""
        w = np.asarray(w0, dtype=float).copy()
        self._init(w.size)
        self.path_ = [w.copy()]
        for _ in range(n_iter):
            w = self._step(w, np.asarray(grad(w), dtype=float))
            self.path_.append(w.copy())
        self.weights_ = w
        return w

    def fit(self, X, y=None):  # type: ignore[override]
        raise TypeError("an optimiser has no fit: call minimize(grad, w0)")


class GradientDescent(_Optimizer):
    """``w <- w - lr g`` on the whole gradient.

    Args:
        lr: step size.

    Example:
        >>> GradientDescent(lr=0.5).minimize(lambda w: 2 * w, [1.0], 3).round(4).tolist()
        [0.0]
    """

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def _init(self, d):
        pass

    def _step(self, w, g):
        return w - self.lr * g


class AdaGrad(_Optimizer):
    """``w <- w - lr g / (sqrt(sum g^2) + eps)`` per coordinate.

    Args:
        lr: step size.
        eps: constant added to the denominator.

    Example:
        >>> AdaGrad(lr=1.0).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist()
        [-1.0]
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8) -> None:
        self.lr = lr
        self.eps = eps

    def _init(self, d):
        self.acc_ = np.zeros(d)

    def _step(self, w, g):
        self.acc_ = self.acc_ + g * g
        return w - self.lr * g / (np.sqrt(self.acc_) + self.eps)


class Adam(_Optimizer):
    """Adam with bias-corrected moments (Algorithm 1 of Kingma and Ba).

    Args:
        lr: step size.
        beta1: decay of the first moment.
        beta2: decay of the second moment.
        eps: constant added to the denominator.

    Example:
        >>> Adam(lr=0.1).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist()
        [-0.1]
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def _init(self, d):
        self.m_ = np.zeros(d)
        self.v_ = np.zeros(d)
        self.t_ = 0

    def _step(self, w, g):
        self.t_ += 1
        self.m_ = self.beta1 * self.m_ + (1 - self.beta1) * g
        self.v_ = self.beta2 * self.v_ + (1 - self.beta2) * g * g
        mh = self.m_ / (1 - self.beta1**self.t_)
        vh = self.v_ / (1 - self.beta2**self.t_)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)


class FTRLProximal(_Optimizer):
    """Follow-the-Regularized-Leader with linearised losses and an L1 and L2 term.

    ``z_t = sum_{s <= t} g_s`` and ``eta_t = alpha / t^beta``; the weights are
    the soft threshold ``w = -sign(z) max(|z| - l1, 0) / (1 / eta_t + l2)``
    (Orabona 2019, Chapter 7, Algorithm 7.2; the soft threshold is the
    proximal operator of the L1 norm, Section 7.8; not in the folder,
    bibliographic only).

    Args:
        alpha: numerator of the learning-rate schedule.
        beta: exponent of the schedule, in ``[0, 1]``.
        l1: L1 strength, which makes small weights exactly zero.
        l2: L2 strength.

    Example:
        >>> FTRLProximal(alpha=1.0, beta=0.5).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist()
        [-1.0]
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 0.5, l1: float = 0.0, l2: float = 0.0
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

    def _init(self, d):
        self.z_ = np.zeros(d)
        self.t_ = 0

    def _step(self, w, g):
        self.t_ += 1
        self.z_ = self.z_ + g
        eta = self.alpha / self.t_**self.beta
        shrunk = np.maximum(np.abs(self.z_) - self.l1, 0.0)
        return -np.sign(self.z_) * shrunk / (1.0 / eta + self.l2)


class NewtonStep(_Optimizer):
    """``w <- w - (1 / gamma) A^-1 g`` with ``A = eps I + sum g g^T``.

    Args:
        gamma: step parameter.
        eps: initial diagonal of ``A``.

    Example:
        >>> NewtonStep(gamma=1.0, eps=1.0).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist()
        [-0.5]
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-6) -> None:
        self.gamma = gamma
        self.eps = eps

    def _init(self, d):
        self.A_ = self.eps * np.eye(d)

    def _step(self, w, g):
        self.A_ = self.A_ + np.outer(g, g)
        return w - np.linalg.solve(self.A_, g) / self.gamma
