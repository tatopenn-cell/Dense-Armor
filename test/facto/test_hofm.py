"""Tests for the higher-order FM (Blondel et al. 2016, eq. (3)-(6))."""

import doctest
from itertools import combinations

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.facto import HOFMClassifier, HOFMRegressor
from dense_armor.utility.facto import hofm as hofm_module
from dense_armor.utility.optim import SGD


def _factory_clf():
    return HOFMClassifier(optimizer=SGD(lr=0.05), m=3, k=2, seed=0)


def _factory_reg():
    return HOFMRegressor(optimizer=SGD(lr=0.05), m=3, k=2, seed=0)


def _anova_bruteforce(p, idx, vals, m):
    """Direct sum of eq. (5), page 2: sum over all m-combinations."""
    total = 0.0
    for S in combinations(range(len(idx)), m):
        prod = 1.0
        for j in S:
            prod *= p[idx[j]] * vals[j]
        total += prod
    return total


def test_anova_dp_matches_bruteforce():
    """Algorithm 1 page 4 equals the direct sum of eq. (5) page 2."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = 6
        p = rng.normal(size=d)
        nz = rng.integers(2, d + 1)
        idx = np.sort(rng.choice(d, size=nz, replace=False))
        vals = rng.normal(size=nz)
        for m in range(2, nz + 1):
            a = hofm_module._anova_forward(p, idx, vals, m)
            dp = float(a[len(idx), m])
            brute = _anova_bruteforce(p, idx, vals, m)
            assert abs(dp - brute) < 1e-10, f"dp={dp} brute={brute} m={m}"


def test_anova_grad_matches_finite_differences():
    """Algorithm 2 page 4 equals the finite-difference gradient."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=6)
    idx = np.array([0, 2, 4])
    vals = np.array([0.5, -1.0, 2.0])
    m = 2
    g = hofm_module._anova_grad(p, idx, vals, m)
    eps = 1e-6
    for i in idx:
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        a_plus = hofm_module._anova_forward(p_plus, idx, vals, m)[len(idx), m]
        a_minus = hofm_module._anova_forward(p_minus, idx, vals, m)[len(idx), m]
        fd = (a_plus - a_minus) / (2 * eps)
        assert abs(g[i] - fd) < 1e-5, f"i={i}  an={g[i]}  fd={fd}"
    for i in range(6):
        if i not in idx:
            assert g[i] == 0.0


def test_model_equals_direct_sum():
    """Eq. (3) page 2: y_hat = <w,x> + sum_t sum_s A_t(p_s, x)."""
    rng = np.random.default_rng(2)
    m = HOFMClassifier(optimizer=SGD(lr=0.05), m=3, k=2, seed=0)
    m._resolve_keys({f"x{i}": 0.0 for i in range(6)})
    m._ensure_state()
    for _ in range(20):
        x = {f"x{i}": float(rng.normal()) for i in range(6) if rng.random() < 0.5}
        if len(x) < 3:
            continue
        fast = m._y_hat(x)
        idx, vals = m._active(x)
        w = m._w()
        slow = float(w[idx] @ vals)
        for t in range(2, 4):
            P = m._block(t)
            for s in range(m.k):
                p_s = P[:, s]
                for S in combinations(range(len(idx)), t):
                    prod = 1.0
                    for j in S:
                        prod *= p_s[idx[j]] * vals[j]
                    slow += prod
        assert abs(fast - slow) < 1e-10, f"fast={fast} slow={slow}"


def test_m2_equals_plain_fm():
    """At m = 2 the ANOVA kernel reduces to the plain pairwise FM."""
    rng = np.random.default_rng(3)
    ho = HOFMClassifier(optimizer=SGD(lr=0.0 + 1e-12), m=2, k=3, seed=0)
    ho._resolve_keys({f"x{i}": 0.0 for i in range(5)})
    ho._ensure_state()
    x = {f"x{i}": float(rng.normal()) for i in range(5) if rng.random() < 0.6}
    if len(x) < 2:
        return
    idx, vals = ho._active(x)
    P = ho._block(2)
    direct = 0.0
    for a, i in enumerate(idx):
        for b in range(a + 1, len(idx)):
            j = idx[b]
            direct += float(P[i] @ P[j]) * vals[a] * vals[b]
    a = ho._y_hat(x)
    w = ho._w()
    expected = float(w[idx] @ vals) + direct
    assert abs(a - expected) < 1e-10


def test_learns_a_constant_target():
    m = _factory_reg()
    for _ in range(300):
        _ = m.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1.5)
    assert abs(m.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) - 1.5) < 0.5


def test_predict_before_learning():
    r = _factory_reg()
    assert r.predict_one({"a": 1.0}) == 0.0
    c = _factory_clf()
    assert c.predict_proba_one({"a": 1.0}) == {}
    assert c.predict_one({"a": 1.0}) is None


def test_nan_skipped_and_counted():
    m = _factory_reg()
    _ = m.learn_one({"a": 1.0}, 1.0)
    _ = m.learn_one({"a": float("nan")}, 1.0)
    _ = m.learn_one({"a": 1.0}, float("nan"))
    assert m.n_missing == 2
    assert m.n_seen_ == 1


def test_missing_key_skipped_and_counted():
    m = _factory_reg()
    _ = m.learn_one({"a": 1.0}, 1.0)
    _ = m.learn_one({}, 1.0)
    assert m.n_missing == 1


def test_constructor_arguments_not_overwritten():
    m = HOFMClassifier(optimizer=SGD(lr=0.5), m=3, k=4, l2=0.01, seed=7)
    for _ in range(10):
        _ = m.learn_one({"a": 1.0}, 0)
    assert m.m == 3
    assert m.k == 4
    assert m.l2 == 0.01
    assert m.seed == 7


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        HOFMClassifier(optimizer=SGD(lr=0.1), m=1)
    with pytest.raises(ValueError):
        HOFMClassifier(optimizer=SGD(lr=0.1), k=0)
    with pytest.raises(ValueError):
        HOFMClassifier(optimizer=SGD(lr=0.1), l2=-1.0)


def test_check_estimator_classifier():
    r = check_estimator(_factory_clf())
    assert r["not_applicable"] == []


def test_check_estimator_regressor():
    r = check_estimator(_factory_reg())
    assert r["not_applicable"] == []


def test_doctest():
    res = doctest.testmod(hofm_module, verbose=False)
    assert res.failed == 0
