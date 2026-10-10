"""Tests for OnlineRobustMahalanobis.

Reproduces ISSUE_mahalanobis_punteggio_esplode.md: on healthy
residual samples of the SyntheticArm payload run, the score must stay
below the threshold; before the fix it reached ~10^4.

The residual of joint 1 on the SyntheticArm is cached here as a
constant: it is what ``SyntheticArm(period_s=2.0, rate_hz=20.0,
n_cycles=1, fault_at_s=None, noise_std=(1e-3, 5e-3, 5e-2), seed=0)``
produces for ``tau_1 - nominal_torque(...)[1]``, computed once when
this test was written. Recomputing it at every run costs about 50
seconds because the URDF inverse dynamics dominates the time; the
values below are the real ones and the detector under test is the
library's own, so the test still checks the fix on the real signal.
"""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.mahalanobis import (
    OnlineRobustMahalanobis,
)

HEALTHY_RESIDUAL = np.array(
    [
        0.018080,
        -0.007961,
        -0.033085,
        0.029058,
        0.010106,
        0.016981,
        -0.074354,
        -0.014156,
        -0.100833,
        -0.016224,
        -0.001085,
        0.027917,
        -0.014564,
        -0.018370,
        0.006127,
        0.002667,
        0.067879,
        0.053404,
        0.007094,
        -0.085672,
        -0.061294,
        0.053721,
        0.002552,
        0.001544,
        0.027315,
        -0.057797,
        -0.031787,
        -0.062090,
        0.056718,
        -0.034513,
        -0.020437,
        -0.005161,
        -0.023477,
        0.066777,
        0.014348,
        -0.013549,
        0.004183,
        -0.068308,
        -0.015414,
        0.022489,
    ]
)


def test_mahalanobis_healthy_scores_stay_below_threshold():
    d = OnlineRobustMahalanobis(feature_keys=["r"], n_init=5)
    scores = []
    for r in HEALTHY_RESIDUAL:
        scores.append(d.score_one({"r": float(r)}))
        d.learn_one({"r": float(r)})
    scores_arr = np.asarray(scores)
    assert np.all(np.isfinite(scores_arr))
    assert scores_arr.max() < 7.0


def test_mahalanobis_outlier_scores_higher():
    d = OnlineRobustMahalanobis(feature_keys=["r"], n_init=5)
    for r in HEALTHY_RESIDUAL:
        d.learn_one({"r": float(r)})
    s_in = d.score_one({"r": 0.0})
    s_out = d.score_one({"r": 1.0})
    assert s_out > s_in


def test_mahalanobis_nan_counted_missing():
    d = OnlineRobustMahalanobis(feature_keys=["r"], n_init=5)
    d.learn_one({"r": float("nan")})
    assert d.n_missing == 1


def test_mahalanobis_rejects_bad_params():
    with pytest.raises(ValueError):
        OnlineRobustMahalanobis(c_gamma=0.0)
    with pytest.raises(ValueError):
        OnlineRobustMahalanobis(gamma_exp=0.4)
    with pytest.raises(ValueError):
        OnlineRobustMahalanobis(gamma_exp=1.0)
    with pytest.raises(ValueError):
        OnlineRobustMahalanobis(n0=-1)
    with pytest.raises(ValueError):
        OnlineRobustMahalanobis(n_init=1)


def test_mahalanobis_before_ready_returns_zero():
    d = OnlineRobustMahalanobis(feature_keys=["r"], n_init=50)
    for _ in range(10):
        d.learn_one({"r": 0.1})
    assert d.score_one({"r": 0.1}) == 0.0


def test_mahalanobis_two_features():
    d = OnlineRobustMahalanobis(feature_keys=["a", "b"], n_init=50)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.standard_normal(2)
        d.learn_one({"a": float(x[0]), "b": float(x[1])})
    s = d.score_one({"a": 0.0, "b": 0.0})
    assert s < 7.0
    s_out = d.score_one({"a": 10.0, "b": 10.0})
    assert s_out > 7.0


def test_mahalanobis_check_estimator():
    check_estimator(OnlineRobustMahalanobis(feature_keys=["r"], n_init=20))
