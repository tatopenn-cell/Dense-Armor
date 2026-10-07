# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/utility/streaming_mahalanobis.py.

The online robust Mahalanobis detector (minimal version, no Robbins-Monro
reconstruction) is built on the geometric median and the median
covariation matrix, both updated by averaged stochastic gradient (Cardot
et al. 2013, Cardot and Godichon-Baggioni 2017). Tests cover warm-up,
feature selection, separation on a seeded Gaussian ellipsoid with
injected outliers, basic river integration (clone, pickle, repr), the
missing-river error, and the doctest.

`check_estimator` is not used: it feeds dicts whose keys vary between
rows (e.g. river's credit-card benchmark), incompatible with a
multichannel estimator that fixes its feature keys from the first dict
seen. `_unit_test_skips` in the module declares this, as the other
Dense-Armor scorers already do.
"""
import doctest
import importlib
import pickle
import sys
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("river")

import dense_armor.utility.anomaly.mahalanobis as streaming_mahalanobis  # noqa: E402
from dense_armor.utility.anomaly.mahalanobis import (  # noqa: E402
    OnlineRobustMahalanobis,
)


def _seeded_ellipsoid(n=5000, d=3, seed=42):
    rng = np.random.default_rng(seed)
    sigmas_sq = np.array([2.0 * i / (d + 1) for i in range(1, d + 1)])
    D = np.diag(np.sqrt(sigmas_sq))
    rho = 0.3
    T = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    Sigma_true = D @ T @ D
    L = np.linalg.cholesky(Sigma_true)
    x = rng.standard_normal((n, d)) @ L.T
    return x, sigmas_sq


def _seeded_with_outliers(n=5000, d=3, seed=42, n_outliers=50):
    x, sigmas_sq = _seeded_ellipsoid(n=n, d=d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(n - 100, size=n_outliers, replace=False) + 50
    truth = np.zeros(n, dtype=bool)
    truth[idx] = True
    shifts = rng.uniform(5.0, 10.0, (n_outliers, 1)) * rng.choice(
        [-1.0, 1.0], (n_outliers, 1))
    x[idx] += shifts
    return x, truth, sigmas_sq


def _feed(model, x):
    scores = np.empty(len(x))
    for i, row in enumerate(x):
        d = {f"f{j}": float(row[j]) for j in range(row.size)}
        scores[i] = model.score_one(d)
        model.learn_one(d)
    return scores


def test_warmup_scores_zero():
    model = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    for v in [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]:
        assert model.score_one({"a": v[0], "b": v[1]}) == 0.0
        model.learn_one({"a": v[0], "b": v[1]})


def test_score_high_for_outliers_low_for_inliers():
    x, truth, _ = _seeded_with_outliers()
    model = OnlineRobustMahalanobis(feature_keys=["f0", "f1", "f2"],
                                    threshold=7.0)
    scores = _feed(model, x)
    assert np.median(scores[~truth]) < 3.0
    assert np.median(scores[truth]) > 12.0
    outlier_flags = [
        model.is_outlier({"f0": float(x[i, 0]), "f1": float(x[i, 1]),
                          "f2": float(x[i, 2])})
        for i in np.where(truth)[0]
    ]
    assert np.mean(outlier_flags) > 0.9


def test_doctest():
    assert doctest.testmod(streaming_mahalanobis).failed == 0


def test_missing_river_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules,
                       "dense_armor.utility.anomaly.mahalanobis",
                       raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.utility.anomaly.mahalanobis")


def test_mcm_eigenvalues_underestimate_true_variance():
    # The MCM eigenvalues under-estimate the true-covariance eigenvalues
    # (Guillot et al.). On a well-conditioned Gaussian this is what the
    # minimal version leaves uncorrected; the test asserts the property
    # explicitly so that changing to the full version (which reconstructs
    # the true eigenvalues) would be caught here.
    x, sigmas_sq = _seeded_ellipsoid(n=3000, d=2, seed=3)
    model = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    for row in x:
        model.learn_one({"a": float(row[0]), "b": float(row[1])})
    assert np.all(model._delta < sigmas_sq)


def test_feature_keys_default_smallest_lexicographic():
    x, _ = _seeded_ellipsoid(n=300, d=2, seed=4)
    default_model = OnlineRobustMahalanobis()
    explicit_model = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    for row in x:
        d = {"a": float(row[0]), "b": float(row[1])}
        default_model.learn_one(d)
        explicit_model.learn_one(d)
    np.testing.assert_allclose(default_model._m_bar, explicit_model._m_bar,
                               atol=1e-12)
    np.testing.assert_allclose(default_model._V_bar, explicit_model._V_bar,
                               atol=1e-12)


def test_score_one_does_not_change_state():
    x, _ = _seeded_ellipsoid(n=200, d=2, seed=5)
    model = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    for row in x[:100]:
        model.learn_one({"a": float(row[0]), "b": float(row[1])})
    m_before = model._m_bar.copy()
    V_before = model._V_bar.copy()
    d_before = model._delta.copy()
    for row in x[100:150]:
        _ = model.score_one({"a": float(row[0]), "b": float(row[1])})
    np.testing.assert_allclose(model._m_bar, m_before, atol=1e-15)
    np.testing.assert_allclose(model._V_bar, V_before, atol=1e-15)
    np.testing.assert_allclose(model._delta, d_before, atol=1e-15)


def test_single_channel_degenerate_does_not_crash():
    model = OnlineRobustMahalanobis(feature_keys=["a"])
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 1, 100):
        model.learn_one({"a": float(v)})
    score = model.score_one({"a": 0.0})
    assert np.isfinite(score)


def test_river_clone_pickle_repr():
    # River-compatible surface: clone (via deepcopy), pickle round-trip
    # and repr. check_estimator itself is skipped for the reason in the
    # module docstring.
    m = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    m2 = deepcopy(m)
    assert m2.feature_keys == m.feature_keys
    assert m2.threshold == m.threshold
    blob = pickle.dumps(m)
    m3 = pickle.loads(blob)
    assert m3.feature_keys == m.feature_keys
    assert m3.threshold == m.threshold
    assert isinstance(repr(m), str)


def test_is_outlier_uses_threshold():
    from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis

    rng = np.random.default_rng(1)
    m = OnlineRobustMahalanobis(feature_keys=["a", "b"])
    for v in rng.normal(0, 1, (300, 2)):
        m.learn_one({"a": float(v[0]), "b": float(v[1])})
    assert m.is_outlier({"a": 40.0, "b": -40.0})
    assert not m.is_outlier({"a": 0.0, "b": 0.0})
    assert "check_roc_auc" in m._unit_test_skips()
