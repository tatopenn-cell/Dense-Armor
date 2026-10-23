"""Tests for the field-aware FM (Juan et al. 2016, eq. (4)-(6))."""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.facto import FFMClassifier, FFMRegressor
from dense_armor.utility.facto import ffm as ffm_module

FIELDS = {"a": "F1", "b": "F1", "c": "F2", "d": "F2"}


def _factory_clf():
    return FFMClassifier(fields=FIELDS, k=2, lr=0.1, seed=0)


def _factory_reg():
    return FFMRegressor(fields=FIELDS, k=2, lr=0.1, seed=0)


def test_model_equals_double_sum():
    """Eq. (4) page 3: the score is the sum over all pairs."""
    rng = np.random.default_rng(0)
    ffm = FFMClassifier(fields=FIELDS, k=3, lr=0.1, seed=0)
    ffm._resolve_keys({"a": 0.0, "c": 0.0})
    ffm._ensure_state()
    for _ in range(200):
        x = {k: float(rng.normal()) for k in FIELDS if rng.random() < 0.5}
        if len(x) < 2:
            continue
        fast = ffm._y_hat(x)
        active = ffm._active(x)
        slow = 0.0
        for a, (i, fi, xi) in enumerate(active):
            for j, fj, xj in active[a + 1 :]:
                slow += float(np.dot(ffm._w[i, fj], ffm._w[j, fi])) * xi * xj
        assert abs(fast - slow) < 1e-12


def test_initialiser_is_uniform_0_to_inv_sqrt_k():
    """Page 3, below Algorithm 1: w ~ U[0, 1/sqrt(k)]."""
    ffm = FFMClassifier(fields=FIELDS, k=4, lr=0.1, seed=0)
    ffm._resolve_keys({"a": 0.0, "c": 0.0})
    ffm._ensure_state()
    assert ffm._w.min() >= 0.0
    assert ffm._w.max() <= 1.0 / np.sqrt(4) + 1e-12


def test_G_starts_at_one():
    """Page 3: G initialised to one."""
    ffm = FFMClassifier(fields=FIELDS, k=2, lr=0.1, seed=0)
    ffm._resolve_keys({"a": 0.0, "c": 0.0})
    ffm._ensure_state()
    assert np.all(ffm._G == 1.0)


def test_gradient_matches_eq_5_and_6():
    """Eq. (5)-(6) page 3: g_{j1,f2} = lambda w + err w_{j2,f1} x1 x2."""
    ffm = FFMClassifier(fields=FIELDS, k=2, lr=0.1, l2=0.0, seed=0)
    ffm._resolve_keys({"a": 0.0, "c": 0.0})
    ffm._ensure_state()
    x = {"a": 1.0, "c": 0.5}
    y = 1
    score = ffm._y_hat(x)
    err = ffm._grad_error(score, y)
    i_a = ffm._key_index["a"]
    i_c = ffm._key_index["c"]
    f_a = ffm._field_index["F1"]
    f_c = ffm._field_index["F2"]
    w_i_before = ffm._w[i_a, f_c].copy()
    w_j_before = ffm._w[i_c, f_a].copy()
    expected_g_i = err * w_j_before * 1.0 * 0.5
    expected_g_j = err * w_i_before * 1.0 * 0.5
    G_i_before = ffm._G[i_a, f_c].copy()
    G_j_before = ffm._G[i_c, f_a].copy()
    ffm.learn_one(x, y)
    expected_w_i = w_i_before - 0.1 * expected_g_i / np.sqrt(
        G_i_before + expected_g_i**2
    )
    expected_w_j = w_j_before - 0.1 * expected_g_j / np.sqrt(
        G_j_before + expected_g_j**2
    )
    assert np.allclose(expected_w_i, ffm._w[i_a, f_c], atol=1e-10)
    assert np.allclose(expected_w_j, ffm._w[i_c, f_a], atol=1e-10)


def test_learns_a_constant_target():
    ffm = _factory_reg()
    for _ in range(200):
        _ = ffm.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1.0)
    assert ffm.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) > 0.5


def test_predict_before_learning_returns_zero():
    ffm = _factory_reg()
    assert ffm.predict_one({"a": 1.0}) == 0.0
    ffm2 = _factory_clf()
    assert ffm2.predict_proba_one({"a": 1.0}) == {}
    assert ffm2.predict_one({"a": 1.0}) is None


def test_nan_skipped_and_counted():
    ffm = _factory_reg()
    _ = ffm.learn_one({"a": 1.0, "b": 0.5}, 1.0)
    _ = ffm.learn_one({"a": float("nan")}, 1.0)
    _ = ffm.learn_one({"a": 1.0}, float("nan"))
    assert ffm.n_missing == 2
    assert ffm.n_seen_ == 1


def test_missing_key_skipped_and_counted():
    ffm = _factory_reg()
    _ = ffm.learn_one({"a": 1.0}, 1.0)
    _ = ffm.learn_one({}, 1.0)
    assert ffm.n_missing == 1


def test_constructor_arguments_not_overwritten():
    ffm = FFMClassifier(fields=FIELDS, k=4, lr=0.5, l2=0.01, seed=7)
    for _ in range(10):
        _ = ffm.learn_one({"a": 1.0}, 0)
    assert ffm.k == 4
    assert ffm.lr == 0.5
    assert ffm.l2 == 0.01
    assert ffm.seed == 7


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        FFMClassifier(fields=FIELDS, k=0)
    with pytest.raises(ValueError):
        FFMClassifier(fields=FIELDS, lr=0.0)
    with pytest.raises(ValueError):
        FFMClassifier(fields=FIELDS, l2=-1.0)


def test_gradient_for_labels_zero_and_minus_one():
    ffm = _factory_clf()
    sig = 1.0 / (1.0 + np.exp(-0.3))
    assert ffm._grad_error(0.3, 0) == pytest.approx(sig)
    assert ffm._grad_error(0.3, -1) == pytest.approx(sig)
    assert ffm._grad_error(0.3, 1) == pytest.approx(sig - 1.0)
    assert ffm._grad_error(0.3, 1.0) == pytest.approx(sig - 1.0)


def test_label_zero_is_learned():
    x = {"a": 1.0, "c": 1.0}
    ffm = _factory_clf()
    _ = ffm.learn_one(x, 0)
    before = ffm.predict_proba_one(x)[0]
    for _ in range(300):
        _ = ffm.learn_one(x, 0)
    assert ffm.predict_proba_one(x)[0] > before


def test_label_zero_equals_label_minus_one():
    x = {"a": 1.0, "b": 0.5, "c": 0.3}
    zero, minus = _factory_clf(), _factory_clf()
    for _ in range(20):
        _ = zero.learn_one(x, 0)
        _ = minus.learn_one(x, -1)
    assert np.array_equal(zero._w, minus._w)


def test_both_classes_are_separated():
    pos, neg = {"a": 1.0, "c": 1.0}, {"b": 1.0, "d": 1.0}
    ffm = FFMClassifier(
        fields=FIELDS, k=2, lr=0.1, seed=0, feature_keys=["a", "b", "c", "d"]
    )
    for _ in range(300):
        _ = ffm.learn_one(pos, 1)
        _ = ffm.learn_one(neg, 0)
    assert ffm.predict_one(pos) == 1
    assert ffm.predict_one(neg) == 0


def test_check_estimator_classifier():
    r = check_estimator(_factory_clf())
    assert r["not_applicable"] == []


def test_check_estimator_regressor():
    r = check_estimator(_factory_reg())
    assert r["not_applicable"] == []


def test_doctest():
    res = doctest.testmod(ffm_module, verbose=False)
    assert res.failed == 0
