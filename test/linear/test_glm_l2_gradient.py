"""Finite-difference check of the L2 gradient in glm.py."""

import numpy as np

from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.optim import SGD


def _linear_loss(w, xv, y, l2):
    r = float(w @ xv) - y
    return r * r + 0.5 * l2 * float(w @ w)


def _logistic_loss(w, xv, y, l2):
    z = float(w @ xv)
    p = 1.0 / (1.0 + np.exp(-z))
    return -y * np.log(p) - (1.0 - y) * np.log(1.0 - p) + 0.5 * l2 * float(w @ w)


def test_linear_l2_gradient_matches_finite_difference():
    l2 = 0.3
    reg = LinearRegression(optimizer=SGD(lr=0.01), l2=l2, intercept=True)
    reg._ensure_weights(2)
    reg._w = np.array([0.5, -0.3, 0.8])
    x = {"a": 0.4, "b": -0.7}
    y = 1.3
    xv = reg._row(x)
    w = reg._w.copy()

    err = float(w @ xv) - y
    grad_analytic = 2.0 * err * xv + l2 * w

    eps = 1e-6
    grad_fd = np.zeros_like(w)
    for j in range(len(w)):
        w_p, w_m = w.copy(), w.copy()
        w_p[j] += eps
        w_m[j] -= eps
        grad_fd[j] = (_linear_loss(w_p, xv, y, l2) - _linear_loss(w_m, xv, y, l2)) / (
            2.0 * eps
        )

    assert np.allclose(grad_analytic, grad_fd, atol=1e-6)


def test_logistic_l2_gradient_matches_finite_difference():
    l2 = 0.25
    clf = LogisticRegression(optimizer=SGD(lr=0.01), l2=l2, intercept=True)
    clf._ensure_weights(2)
    clf._w = np.array([-0.2, 0.6, 1.1])
    x = {"a": -0.3, "b": 0.5}
    y = 1.0
    xv = clf._row(x)
    w = clf._w.copy()

    z = float(w @ xv)
    p = 1.0 / (1.0 + np.exp(-z))
    grad_analytic = (p - y) * xv + l2 * w

    eps = 1e-6
    grad_fd = np.zeros_like(w)
    for j in range(len(w)):
        w_p, w_m = w.copy(), w.copy()
        w_p[j] += eps
        w_m[j] -= eps
        grad_fd[j] = (
            _logistic_loss(w_p, xv, y, l2) - _logistic_loss(w_m, xv, y, l2)
        ) / (2.0 * eps)

    assert np.allclose(grad_analytic, grad_fd, atol=1e-6)
