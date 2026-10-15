"""Tests for AMRulesRegressor."""

import doctest
import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.rules import AMRulesRegressor
from dense_armor.utility.rules import amrules as mod
from dense_armor.utility.rules.amrules import _Moments, _Rule, _side, _Splits


def _step_stream(n: int, seed: int = 0, flip: bool = False):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = float(rng.uniform(-1.0, 1.0))
        b = float(rng.uniform(-1.0, 1.0))
        y = 5.0 * (a <= 0.0) if flip else 5.0 * (a > 0.0)
        yield {"a": a, "b": b}, y + 0.1 * float(rng.normal())


def _handmade(ordered: bool) -> AMRulesRegressor:
    m = AMRulesRegressor(ordered_set=ordered, seed=0, prediction="mean")
    m._keys = ["a"]
    m.default_ = _Rule(1, m._rng)
    m.default_.target = _Moments(5, 9.0, 0.0)
    for bound, mean in ((1.0, 1.0), (2.0, 3.0)):
        r = _Rule(1, m._rng)
        r.literals = [(0, True, bound)]
        r.target = _Moments(5, mean, 0.0)
        m.rules_.append(r)
    return m


def test_doctests():
    res = doctest.testmod(mod)
    assert res.failed == 0 and res.attempted > 0


def test_check_estimator_passes_with_no_check_skipped():
    est = AMRulesRegressor(seed=0)
    assert est._unit_test_skips() == set()
    check_estimator(est)


def test_variance_ratio_of_a_hand_case():
    sp = _Splits()
    for x, y in ((1.0, 0.0), (2.0, 2.0), (3.0, 4.0), (4.0, 6.0)):
        sp.add(x, y, 50)
    vr, x, left, right = sp.best(4.0, 12.0, 56.0)
    assert vr == pytest.approx(0.8)
    assert x == 2.0 and left == (2.0, 2.0, 4.0) and right == (2.0, 10.0, 52.0)
    perfect = _Splits()
    for x, y in ((1.0, 0.0), (2.0, 0.0), (3.0, 10.0), (4.0, 10.0)):
        perfect.add(x, y, 50)
    assert perfect.best(4.0, 20.0, 200.0)[0] == pytest.approx(1.0)
    assert _Splits().best(0.0, 0.0, 0.0) is None
    one = _Splits()
    one.add(1.0, 1.0, 50)
    assert one.best(1.0, 1.0, 1.0) is None


def test_split_points_are_capped_and_merged_into_the_nearest():
    sp = _Splits()
    sp.add(0.0, 1.0, 2)
    sp.add(10.0, 1.0, 2)
    for x in (4.0, -5.0, 6.0, 20.0, 10.0):
        sp.add(x, 1.0, 2)
    assert sp.xs == [0.0, 10.0]
    assert sp.n == [3, 4]


def test_side_and_moments():
    assert _side(0, 0.0, 0.0).n == 0
    m = _side(4.0, 12.0, 56.0)
    assert m.n == 4 and m.mean == pytest.approx(3.0) and m.m2 == pytest.approx(20.0)
    assert _Moments().std == 0.0


def _tie_rule(n: int) -> _Rule:
    rng = np.random.default_rng(0)
    r = _Rule(2, rng)
    for _ in range(n):
        a = float(rng.uniform(-1.0, 1.0))
        r.learn(np.array([a, a]), 5.0 * (a > 0.0), 0.01, 0.99, 50)
    return r


def test_a_tie_between_two_identical_attributes_waits_for_the_tie_threshold():
    delta, tau = 1e-7, 0.05
    assert math.sqrt(math.log(1.0 / delta) / (2.0 * 200)) > tau
    assert _tie_rule(200).split_candidate(delta, tau) is None
    n_tie = math.ceil(math.log(1.0 / delta) / (2.0 * tau * tau)) + 50
    assert _tie_rule(n_tie).split_candidate(delta, tau) is not None


def test_a_clear_winner_expands_at_the_hoeffding_bound():
    rng = np.random.default_rng(0)
    r = _Rule(2, rng)
    for _ in range(200):
        a, b = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        r.learn(np.array([a, b]), 5.0 * (a > 0.0), 0.01, 0.99, 50)
    vr, j, c, _, _ = r.split_candidate(1e-7, 0.05)
    assert j == 0 and abs(c) < 0.15 and vr > 0.9


def test_constant_attributes_give_no_split_candidate():
    rng = np.random.default_rng(0)
    r = _Rule(2, rng)
    for _ in range(300):
        r.learn(np.array([1.0, 2.0]), float(rng.normal()), 0.01, 0.99, 50)
    assert r.split_candidate(1e-7, 0.05) is None
    flat = _Rule(1, rng)
    for i in range(300):
        flat.learn(np.array([float(i)]), 3.0, 0.01, 0.99, 50)
    assert flat.split_candidate(1e-7, 0.05) is None


def test_learns_a_step_and_names_the_condition():
    m = AMRulesRegressor(n_min=100, seed=0)
    for x, y in _step_stream(3000):
        m.learn_one(x, y)
    assert m.predict_one({"a": 0.5, "b": 0.0}) == pytest.approx(5.0, abs=0.3)
    assert m.predict_one({"a": -0.5, "b": 0.0}) == pytest.approx(0.0, abs=0.3)
    conds = [c for rule in m.explain_one({"a": 0.5, "b": 0.0}) for c in rule]
    assert conds and all(c["feature"] == "a" for c in conds)
    assert (
        m.explain_one({"a": -0.5, "b": 0.0}) == [[]]
        or m.explain_one({"a": -0.5, "b": 0.0}) == []
    )


def test_the_page_hinkley_test_removes_a_rule_after_a_change_of_regime():
    m = AMRulesRegressor(n_min=100, seed=0)
    for x, y in _step_stream(3000):
        m.learn_one(x, y)
    removed = m.n_removed_
    err = []
    for x, y in _step_stream(2000, seed=1, flip=True):
        err.append(abs(m.predict_one(x) - y))
        m.learn_one(x, y)
    assert m.n_removed_ > removed
    assert np.mean(err[:100]) > 1.5
    assert np.mean(err[-300:]) < 1.0


def test_page_hinkley_signals_on_a_jump_and_not_on_a_constant():
    r = _Rule(1, np.random.default_rng(0))
    assert not any(r.change_detected(0.0, 0.005, 35.0) for _ in range(50))
    hits = [r.change_detected(10.0, 0.005, 35.0) for _ in range(6)]
    assert any(hits) and not hits[0]
    assert hits.index(True) == 3


def test_the_delta_rule_descends_the_error():
    m = AMRulesRegressor(n_min=10**6, prediction="perceptron", seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        a, b = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        m.learn_one({"a": a, "b": b}, 3.0 * a - 2.0 * b)
    pts = rng.uniform(-1, 1, (200, 2))
    err = np.mean(
        [abs(m.predict_one({"a": a, "b": b}) - (3 * a - 2 * b)) for a, b in pts]
    )
    assert err < 0.2


def test_strategies_mean_perceptron_and_adaptive():
    base = {"n_min": 10**6, "seed": 0}
    out = {}
    for name in ("mean", "perceptron", "adaptive"):
        m = AMRulesRegressor(prediction=name, **base)
        for x, y in _step_stream(500):
            m.learn_one(x, y)
        out[name] = m.predict_one({"a": 0.9, "b": 0.0})
    assert out["mean"] == pytest.approx(2.5, abs=0.5)
    assert out["perceptron"] > 3.5
    assert out["adaptive"] in (
        pytest.approx(out["mean"]),
        pytest.approx(out["perceptron"]),
    )


def test_unordered_averages_covering_rules_and_ordered_takes_the_first():
    un = _handmade(False)
    od = _handmade(True)
    assert un.predict_one({"a": 0.5}) == pytest.approx(2.0)
    assert od.predict_one({"a": 0.5}) == pytest.approx(1.0)
    assert un.predict_one({"a": 1.5}) == pytest.approx(3.0)
    assert un.predict_one({"a": 5.0}) == pytest.approx(9.0)
    assert len(un.explain_one({"a": 0.5})) == 2 and len(od.explain_one({"a": 0.5})) == 1
    assert un.predict_one({"a": 0.5}, return_std=True) == (pytest.approx(2.0), 0.0)


def test_ordered_learning_updates_only_the_first_covering_rule():
    m = _handmade(True)
    before = [r.n_total for r in m.rules_]
    m.learn_one({"a": 0.5}, 1.0)
    assert [r.n_total for r in m.rules_] == [before[0] + 1, before[1]]
    u = _handmade(False)
    u.learn_one({"a": 0.5}, 1.0)
    assert [r.n_total for r in u.rules_] == [1, 1]


def test_a_sample_nobody_covers_updates_the_default_rule():
    m = _handmade(False)
    n0 = m.default_.n_total
    m.learn_one({"a": 5.0}, 7.0)
    assert m.default_.n_total == n0 + 1
    assert [r.n_total for r in m.rules_] == [0, 0]


def test_a_default_rule_that_expands_becomes_a_rule_and_a_new_default_starts():
    m = AMRulesRegressor(n_min=100, seed=0)
    for x, y in _step_stream(150):
        m.learn_one(x, y)
    assert len(m.rules_) == 1 and m.rules_[0].literals
    assert m.default_.literals == []
    assert m.default_.target.n > 0


def test_a_sample_that_is_anomalous_everywhere_is_ignored_by_the_rule():
    m = AMRulesRegressor(n_min=100, seed=0, anomaly_threshold=-0.75, m_min=30)
    for x, y in _step_stream(1500):
        m.learn_one(x, y)
    seen = [r.n_total for r in m.rules_]
    m.learn_one({"a": 0.5, "b": 40.0}, 0.0)
    assert m.n_anomalies_ == 0
    m.learn_one({"a": 40.0, "b": 40.0}, 0.0)
    assert m.n_anomalies_ >= 1
    assert sum(r.n_total for r in m.rules_) < sum(seen) + 2
    off = AMRulesRegressor(n_min=100, seed=0)
    for x, y in _step_stream(1500):
        off.learn_one(x, y)
    off.learn_one({"a": 40.0, "b": 40.0}, 0.0)
    assert off.n_anomalies_ == 0


def test_a_young_rule_does_not_check_anomalies():
    r = _Rule(2, np.random.default_rng(0))
    for i in range(10):
        r.learn(np.array([float(i), 0.0]), 1.0, 0.01, 0.99, 50)
    assert not r.is_anomaly(np.array([1e6, 1e6]), -0.75, 30)
    assert r.is_anomaly(np.array([1e6, 1e6]), -0.75, 5)


def test_the_size_of_the_model_is_bounded():
    m = AMRulesRegressor(
        n_min=20, max_rules=1, max_literals=1, max_split_points=3, seed=0
    )
    for x, y in _step_stream(3000):
        m.learn_one(x, y)
    assert len(m.rules_) <= 1
    assert all(len(r.literals) <= 1 for r in m.rules_ + [m.default_])
    assert all(len(s.xs) <= 3 for r in m.rules_ + [m.default_] for s in r.splits)


def test_nan_missing_and_none_are_skipped_and_counted():
    m = AMRulesRegressor(seed=0)
    for x, y in _step_stream(50):
        m.learn_one(x, y)
    n0, seen = m.n_missing, m.n_seen_
    for x, y in (
        ({"a": float("nan"), "b": 0.0}, 1.0),
        ({"a": float("inf"), "b": 0.0}, 1.0),
        ({"a": 1.0}, 1.0),
        ({"a": "word", "b": 0.0}, 1.0),
        ({}, 1.0),
        (None, 1.0),
        ({"a": 0.1, "b": 0.0}, None),
        ({"a": 0.1, "b": 0.0}, float("nan")),
        ({"a": 0.1, "b": 0.0}, "word"),
    ):
        m.learn_one(x, y)
    assert m.n_missing == n0 + 9 and m.n_seen_ == seen
    assert m.predict_one({"a": float("nan"), "b": 0.0}) == 0.0
    assert m.predict_one(None) == 0.0
    assert m.explain_one({"a": float("nan"), "b": 0.0}) == []
    for r in [m.default_, *m.rules_]:
        assert np.isfinite(r.w).all()


def test_predictions_before_learning():
    m = AMRulesRegressor(seed=0)
    assert m.predict_one({"a": 1.0}) == 0.0
    assert m.predict_one({"a": 1.0}, return_std=True) == (0.0, 0.0)
    assert m.explain_one({"a": 1.0}) == []


def test_non_finite_weights_fall_back_to_the_mean_and_are_reset():
    r = _Rule(1, np.random.default_rng(0))
    for i in range(5):
        r.learn(np.array([float(i)]), 2.0 * i, 0.01, 0.99, 50)
    r.w = np.array([np.inf, 0.0])
    assert r.predict_perceptron(np.array([1.0])) == pytest.approx(r.predict_mean())
    r.w = np.array([np.nan, 0.0])
    r.learn(np.array([2.0]), 4.0, 0.01, 0.99, 50)
    assert np.isfinite(r.w).all()


def test_constructor_arguments_not_overwritten_and_clone_is_fresh():
    m = AMRulesRegressor(
        n_min=80, ph_lambda=20.0, anomaly_threshold=-1.0, ordered_set=True, seed=3
    )
    before = m.get_params()
    for x, y in _step_stream(600):
        m.learn_one(x, y)
    assert m.get_params() == before
    c = m.clone()
    assert c.get_params() == before and c.rules_ == [] and c.default_ is None


def test_same_seed_same_model():
    outs = []
    for _ in range(2):
        m = AMRulesRegressor(n_min=100, seed=7)
        for x, y in _step_stream(800):
            m.learn_one(x, y)
        outs.append(m.predict_one({"a": 0.3, "b": -0.2}))
    assert outs[0] == outs[1]


def test_bad_parameters_raise():
    for kw in (
        {"n_min": 0},
        {"delta": 0.0},
        {"delta": 1.0},
        {"tau": -0.1},
        {"ph_lambda": 0.0},
        {"ph_phi": -1.0},
        {"alpha": 1.0},
        {"learning_rate": 0.0},
        {"prediction": "other"},
        {"m_min": -1},
        {"max_rules": 0},
        {"max_literals": 0},
        {"max_split_points": 0},
    ):
        with pytest.raises(ValueError):
            AMRulesRegressor(**kw)
