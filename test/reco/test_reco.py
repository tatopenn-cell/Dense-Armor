"""Tests for the recommenders (Paterek 2007, Koren 2009)."""

import doctest

import numpy as np
import pytest

from dense_armor.utility.reco import Baseline, BiasedMF, FunkMF
from dense_armor.utility.reco import reco as reco_module


def _planted(n_users=20, n_items=15, k=3, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_users, k))
    Q = rng.normal(size=(n_items, k))
    mu = 3.5
    ratings = []
    for u in range(n_users):
        for i in range(n_items):
            r = mu + float(P[u] @ Q[i]) + float(rng.normal(0, 0.01))
            ratings.append((u, i, r))
    return ratings, P, Q, mu


def _planted_bias_only(n_users=10, n_items=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = 3.5
    bu = rng.normal(0, 0.5, size=n_users)
    bi = rng.normal(0, 0.5, size=n_items)
    ratings = []
    for u in range(n_users):
        for i in range(n_items):
            r = mu + bu[u] + bi[i] + float(rng.normal(0, 0.01))
            ratings.append((u, i, r))
    return ratings


def test_baseline_recovers_bias_only_data():
    ratings = _planted_bias_only(seed=0)
    b = Baseline(lr=0.05, l2=0.0, seed=0)
    for _ in range(200):
        for u, i, r in ratings:
            _ = b.learn_one(u, i, r)
    errs = [abs(b.predict_one(u, i) - r) for u, i, r in ratings]
    assert float(np.median(errs)) < 0.3


def test_baseline_missing_counted():
    b = Baseline(lr=0.01)
    b.learn_one("u", "i", None)
    b.learn_one(None, "i", 3.0)
    b.learn_one("u", "i", float("nan"))
    b.learn_one("u", "i", "bad")
    assert b.n_missing == 4
    assert b.n_seen_ == 0


def test_baseline_unknown_user_item_returns_mean():
    b = Baseline(lr=0.05)
    b.learn_one("u", "i", 4.0)
    assert b.predict_one("new_user", "new_item") == b.mu_


def test_biased_mf_recovers_planted_low_rank():
    ratings, _, _, _ = _planted(seed=1)
    m = BiasedMF(k=3, lr=0.02, l2=0.001, seed=0)
    for _ in range(60):
        for u, i, r in ratings:
            _ = m.learn_one(u, i, r)
    errs = [abs(m.predict_one(u, i) - r) for u, i, r in ratings]
    assert float(np.median(errs)) < 0.5
    assert float(np.percentile(errs, 90)) < 1.5


def test_funk_mf_recovers_planted_low_rank():
    ratings, _, _, _ = _planted(seed=2)
    m = FunkMF(k=3, lr=0.02, l2=0.001, seed=0)
    for _ in range(30):
        for u, i, r in ratings:
            _ = m.learn_one(u, i, r)
    errs = [abs(m.predict_one(u, i) - r) for u, i, r in ratings]
    assert np.mean(errs) < 0.5


def test_rank_returns_items_by_predicted_rating():
    ratings, _, _, _ = _planted(seed=3)
    m = BiasedMF(k=3, lr=0.02, l2=0.001, seed=0)
    for _ in range(30):
        for u, i, r in ratings:
            _ = m.learn_one(u, i, r)
    items = list(range(15))
    ranked = m.rank(0, items, top_k=5)
    assert len(ranked) == 5
    scores = [m.predict_one(0, i) for i in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_without_top_k_returns_all():
    ratings, _, _, _ = _planted(seed=4)
    m = FunkMF(k=3, lr=0.02, l2=0.001, seed=0)
    for _ in range(10):
        for u, i, r in ratings:
            _ = m.learn_one(u, i, r)
    ranked = m.rank(0, list(range(15)))
    assert len(ranked) == 15


def test_biased_mf_missing_counted():
    m = BiasedMF(k=3)
    m.learn_one("u", "i", None)
    m.learn_one("u", "i", float("nan"))
    m.learn_one(None, "i", 3.0)
    m.learn_one("u", None, 3.0)
    assert m.n_missing == 4


def test_funk_mf_missing_counted():
    m = FunkMF(k=3)
    m.learn_one("u", "i", None)
    m.learn_one("u", "i", float("nan"))
    assert m.n_missing == 2


def test_predict_before_learning_returns_mu_zero():
    assert Baseline().predict_one("u", "i") == 0.0
    assert BiasedMF().predict_one("u", "i") == 0.0
    assert FunkMF().predict_one("u", "i") == 0.0


def test_constructor_arguments_not_overwritten():
    m = BiasedMF(k=4, lr=0.05, l2=0.01, seed=7)
    for _ in range(5):
        _ = m.learn_one(0, 0, 3.0)
    assert m.k == 4
    assert m.lr == 0.05
    assert m.l2 == 0.01
    assert m.seed == 7


def test_clone_is_independent():
    m = BiasedMF(k=3, seed=0)
    for _ in range(5):
        _ = m.learn_one(0, 0, 3.0)
    c = m.clone()
    assert len(c.p_) == 0
    assert c.mu_ == 0.0


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        Baseline(lr=0.0)
    with pytest.raises(ValueError):
        Baseline(l2=-1.0)
    with pytest.raises(ValueError):
        BiasedMF(k=0)
    with pytest.raises(ValueError):
        FunkMF(k=0)


def test_doctest():
    res = doctest.testmod(reco_module, verbose=False)
    assert res.failed == 0
