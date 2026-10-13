"""Tests for OnlineRobustMahalanobis.

Reproduces ISSUE_mahalanobis_punteggio_esplode.md: on the 100 healthy
residual samples of the SyntheticArm payload run, the score must stay
below the threshold; before the fix it reached ~10^4.

The residuals below are cached as constants with 6 decimals. They are
what ``SyntheticArm("test/fixtures/urdf/panda.urdf", period_s=2.0,
rate_hz=50.0, n_cycles=2, fault_at_s=2.0, fault="payload",
payload_mass=5.0, noise_std=(1e-3, 5e-3, 5e-2), seed=0)`` produces
for ``tau_1 - nominal_torque(...)[1]`` on its 100 healthy samples,
regenerated with the library's own generator. Recomputing them at
every run costs about 45 seconds because the URDF inverse dynamics
dominates the time, so they are stored here and the detector under
test is the library's own.
"""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.mahalanobis import (
    OnlineRobustMahalanobis,
)

ISSUE_HEALTHY_RESIDUAL = np.array(
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
        0.018634,
        0.010391,
        -0.014279,
        0.035678,
        -0.078920,
        -0.084754,
        0.137095,
        0.098833,
        -0.081755,
        -0.084425,
        0.021798,
        0.004123,
        -0.053229,
        0.045264,
        -0.043806,
        0.025770,
        -0.011059,
        0.037505,
        0.029552,
        0.010155,
        -0.094861,
        -0.140965,
        -0.013149,
        0.084007,
        -0.054209,
        -0.046969,
        0.065705,
        0.030694,
        -0.034929,
        0.010068,
        -0.004354,
        -0.070336,
        -0.038810,
        0.073520,
        -0.080015,
        -0.040111,
        0.004967,
        0.034610,
        -0.020271,
        0.111063,
        0.094900,
        -0.007044,
        -0.056706,
        -0.034416,
        -0.027124,
        0.060405,
        -0.027393,
        -0.026144,
        0.016883,
        0.047139,
        0.098683,
        0.002861,
        -0.059022,
        -0.032567,
        -0.010796,
        -0.041247,
        0.030076,
        -0.005769,
        0.001823,
        0.058005,
    ]
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
    """Issue scenario: 100 healthy samples, n_init=20, scores stay < 7.

    Asserts that at least 10 scores are non-zero (the model is out of
    its warm-up), that every score is finite and that the maximum is
    below the default threshold.
    """
    d = OnlineRobustMahalanobis(feature_keys=["r"], n_init=20)
    scores = []
    for r in ISSUE_HEALTHY_RESIDUAL:
        scores.append(d.score_one({"r": float(r)}))
        d.learn_one({"r": float(r)})
    scores_arr = np.asarray(scores)
    assert np.count_nonzero(scores_arr) >= 10, (
        f"only {int(np.count_nonzero(scores_arr))} non-zero scores, "
        "the model never left its warm-up"
    )
    assert np.all(np.isfinite(scores_arr))
    assert scores_arr.max() < 7.0, f"max score = {scores_arr.max():.6f}"


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


def test_mahalanobis_cap_bounds_the_step():
    """The step cap ``max_step_frac`` keeps healthy scores below 7.

    On the same 40 samples of N(0, 0.05^2), the default cap (0.1) keeps
    the maximum score below 7, and with the cap disabled (1.0) the
    maximum exceeds 7. A detector that ignores the cap fails the first
    assertion.
    """
    rng = np.random.default_rng(0)
    series = rng.normal(0.0, 0.05, size=40).tolist()

    def _run(max_step_frac):
        d = OnlineRobustMahalanobis(
            feature_keys=["r"], n_init=5, max_step_frac=max_step_frac
        )
        scores = []
        for r in series:
            scores.append(d.score_one({"r": float(r)}))
            d.learn_one({"r": float(r)})
        return max(scores)

    m_capped = _run(0.1)
    m_uncapped = _run(1.0)
    assert m_capped < 7.0, f"capped max = {m_capped:.3f}"
    assert m_uncapped > 7.0, f"uncapped max = {m_uncapped:.3f}"
