"""The online metrics against the scoring functions of model-diagnostics.

The reference is computed from whole arrays by an independent package
(model-diagnostics 1.5.0); the metrics here accumulate one pair at a time.
The tests are skipped when the package is missing, which is the case on
Python 3.10 (the package needs 3.11).
"""

import math

import numpy as np
import pytest
from scipy import special

scoring = pytest.importorskip("model_diagnostics.scoring")

from dense_armor.utility.metrics import (  # noqa: E402
    BrierScore,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
)

REL = 1e-12


def _feed(metric, y, z, w=None):
    for i in range(len(y)):
        weight = 1.0 if w is None else float(w[i])
        metric.update(float(y[i]), float(z[i]), w=weight)
    return metric.get()


@pytest.fixture(scope="module")
def probs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 500)
    y = (rng.random(500) < p).astype(float)
    w = rng.uniform(0.5, 2.0, 500)
    return y, p, w


@pytest.fixture(scope="module")
def reals():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 500)
    z = y + rng.normal(0.0, 0.3, 500)
    w = rng.uniform(0.5, 2.0, 500)
    return y, z, w


@pytest.mark.parametrize("weighted", [False, True])
def test_brier_score_equals_the_squared_error(probs, weighted):
    y, p, w = probs
    weights = w if weighted else None
    ref = scoring.SquaredError()(y, p, weights=weights)
    assert _feed(BrierScore(), y, p, weights) == pytest.approx(ref, rel=REL)


def test_brier_score_with_soft_labels_equals_the_squared_error(probs):
    _, p, w = probs
    y = np.random.default_rng(2).uniform(0.0, 1.0, 500)
    ref = scoring.SquaredError()(y, p, weights=w)
    assert _feed(BrierScore(), y, p, w) == pytest.approx(ref, rel=REL)


@pytest.mark.parametrize("weighted", [False, True])
def test_log_loss_equals_the_reference_for_hard_labels(probs, weighted):
    y, p, w = probs
    weights = w if weighted else None
    ref = scoring.LogLoss()(y, p, weights=weights)
    assert _feed(LogLoss(), y, p, weights) == pytest.approx(ref, rel=REL)


def test_log_loss_with_soft_labels_exceeds_the_reference_by_the_entropy(probs):
    """The reference subtracts the entropy of a soft label (a deviance);
    LogLoss is the cross-entropy, so the two differ by that entropy."""
    _, p, w = probs
    y = np.random.default_rng(3).uniform(0.05, 0.95, 500)
    ref = scoring.LogLoss()(y, p, weights=w)
    entropy = -special.xlogy(y, y) - special.xlogy(1.0 - y, 1.0 - y)
    expected = ref + np.average(entropy, weights=w)
    assert _feed(LogLoss(), y, p, w) == pytest.approx(expected, rel=REL)


def test_log_loss_is_finite_where_the_reference_is_infinite():
    ref = scoring.LogLoss()([1.0], [0.0])
    assert math.isinf(ref)
    assert _feed(LogLoss(), [1.0], [0.0]) == pytest.approx(-math.log(1e-15), rel=REL)


@pytest.mark.parametrize("weighted", [False, True])
def test_mean_squared_error_equals_the_squared_error(reals, weighted):
    y, z, w = reals
    weights = w if weighted else None
    ref = scoring.SquaredError()(y, z, weights=weights)
    assert _feed(MeanSquaredError(), y, z, weights) == pytest.approx(ref, rel=REL)


def test_root_mean_squared_error_is_the_root_of_the_squared_error(reals):
    y, z, w = reals
    ref = math.sqrt(scoring.SquaredError()(y, z, weights=w))
    assert _feed(RootMeanSquaredError(), y, z, w) == pytest.approx(ref, rel=REL)


@pytest.mark.parametrize("weighted", [False, True])
def test_mean_absolute_error_is_twice_the_median_pinball_loss(reals, weighted):
    y, z, w = reals
    weights = w if weighted else None
    ref = 2.0 * scoring.PinballLoss(level=0.5)(y, z, weights=weights)
    assert _feed(MeanAbsoluteError(), y, z, weights) == pytest.approx(ref, rel=REL)


def test_hand_case_of_the_squared_error_and_the_absolute_error():
    y = [0.0, 0.0, 1.0, 1.0]
    z = [-1.0, 1.0, 1.0, 2.0]
    assert _feed(MeanSquaredError(), y, z) == 0.75
    assert _feed(MeanAbsoluteError(), y, z) == 0.75
    assert scoring.SquaredError()(y, z) == 0.75
    assert 2.0 * scoring.PinballLoss(level=0.5)(y, z) == 0.75
