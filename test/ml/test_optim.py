"""Hand cases of the optimisers of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.optim as O
from dense_armor.ml.optim import (
    AdaGrad,
    Adam,
    FTRLProximal,
    GradientDescent,
    NewtonStep,
)


def test_doctests():
    r = doctest.testmod(O)
    assert r.failed == 0 and r.attempted >= 5


def test_gradient_descent_hand_case():
    o = GradientDescent(lr=0.25)
    w = o.minimize(lambda w: 2 * w, [1.0], 2)
    assert w.tolist() == [0.25] and len(o.path_) == 3 and o.weights_.tolist() == [0.25]


def test_adagrad_hand_case():
    w = AdaGrad(lr=1.0, eps=0.0).minimize(lambda w: np.array([2.0]), [0.0], 2)
    assert np.allclose(w, [-1.0 - 2.0 / np.sqrt(8.0)])


def test_adam_constant_gradient():
    w = Adam(lr=0.1, eps=0.0).minimize(lambda w: np.array([1.0, -2.0]), [0.0, 0.0], 4)
    assert np.allclose(w, [-0.4, 0.4])


def test_ftrl_hand_cases():
    w = FTRLProximal(alpha=1.0, beta=0.5).minimize(lambda w: np.array([1.0]), [0.0], 2)
    assert np.allclose(w, [-np.sqrt(2.0)])
    sparse = FTRLProximal(l1=2.0).minimize(lambda w: np.array([1.0]), [0.0], 1)
    assert sparse.tolist() == [0.0]
    shrunk = FTRLProximal(alpha=1.0, beta=0.5, l1=0.5, l2=1.0).minimize(
        lambda w: np.array([1.0]), [0.0], 1
    )
    assert np.allclose(shrunk, [-0.25])


def test_newton_step_hand_case():
    w = NewtonStep(gamma=2.0, eps=1.0).minimize(lambda w: np.array([1.0]), [0.0], 2)
    assert np.allclose(w, [-0.25 - 1.0 / (3.0 * 2.0)])


def test_an_optimiser_has_no_fit():
    with pytest.raises(TypeError):
        GradientDescent().fit([1.0])
    with pytest.raises(NotImplementedError):
        O._Optimizer()._init(1)
    with pytest.raises(NotImplementedError):
        O._Optimizer()._step(np.zeros(1), np.zeros(1))
