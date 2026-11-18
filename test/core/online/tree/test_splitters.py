"""The criteria and the observers of the Hoeffding trees, on small cases worked by hand."""

import math

import pytest

from dense_armor.core.online.tree.splitters import (
    NominalObserver,
    NumericObserver,
    TargetNominalObserver,
    TargetNumericObserver,
    entropy,
    gini,
    impurity,
    merit_range,
    split_gain,
)


def test_entropy_and_gini_of_known_counts():
    assert entropy({"a": 5, "b": 5}) == pytest.approx(1.0)
    assert entropy({"a": 10}) == 0.0
    assert entropy({}) == 0.0
    assert gini({"a": 5, "b": 5}) == pytest.approx(0.5)
    assert gini({"a": 10}) == 0.0
    assert gini({}) == 0.0
    assert impurity("entropy", {"a": 1, "b": 1}) == pytest.approx(1.0)
    assert impurity("gini", {"a": 1, "b": 1}) == pytest.approx(0.5)


def test_merit_range():
    assert merit_range("entropy", 4) == pytest.approx(2.0)
    assert merit_range("gini", 4) == pytest.approx(0.75)


def test_split_gain_of_a_perfect_split_is_the_parent_impurity():
    parent = {"a": 5, "b": 5}
    assert split_gain("entropy", parent, 10, {"a": 5}, {"b": 5}) == pytest.approx(1.0)
    assert split_gain("gini", parent, 10, {"a": 5}, {"b": 5}) == pytest.approx(0.5)
    assert split_gain("gini", parent, 10, {}, {"b": 5}) is None


def _observer(splitter, points, **kw):
    obs = NumericObserver(splitter, **kw)
    for v, y in points:
        obs.update(v, y)
    return obs


POINTS = [(1.0, "a"), (2.0, "a"), (3.0, "a"), (7.0, "b"), (8.0, "b"), (9.0, "b")]


def test_ebst_threshold_is_the_boundary_between_the_classes():
    obs = _observer("ebst", POINTS)
    gain, thr = obs.best_candidate("gini", {"a": 3, "b": 3}, 6)
    assert thr == pytest.approx(5.0)
    assert gain == pytest.approx(0.5)
    assert obs.test_gain("gini", {"a": 3, "b": 3}, 6, 5.0) == pytest.approx(0.5)
    assert obs.test_gain("gini", {"a": 3, "b": 3}, 6, 2.5) < 0.5


def test_ebst_with_one_distinct_value_has_no_candidate():
    obs = _observer("ebst", [(1.0, "a"), (1.0, "b")])
    assert obs.best_candidate("entropy", {"a": 1, "b": 1}, 2) == (-1.0, None)


def test_gaussian_candidates_are_evenly_spaced_between_min_and_max():
    obs = _observer("gaussian", POINTS, n_splits=3)
    thresholds = obs._thresholds()
    assert thresholds == pytest.approx([3.0, 5.0, 7.0])
    gain, thr = obs.best_candidate("gini", {"a": 3, "b": 3}, 6)
    assert thr == pytest.approx(5.0)
    assert gain > 0.3


def test_class_means_candidate_is_the_midpoint_of_the_means():
    obs = _observer("class_means", POINTS)
    assert obs._thresholds() == pytest.approx([5.0])
    assert _observer("class_means", [(1.0, "a")])._thresholds() == []


def test_quantile_observer_warms_up_then_uses_slots():
    obs = NumericObserver("quantile")
    for i in range(10):
        obs.update(float(i), "a" if i < 5 else "b")
    assert obs._radius is None
    gain, thr = obs.best_candidate("gini", {"a": 5, "b": 5}, 10)
    assert thr == pytest.approx(4.5)
    for i in range(10, 60):
        obs.update(float(i % 10), "a" if i % 10 < 5 else "b")
    assert obs._radius is not None
    assert len(obs._slots) <= 12
    gain, thr = obs.best_candidate("gini", {"a": 30, "b": 30}, 60)
    assert 3.0 < thr < 6.0


def test_quantile_observer_with_a_given_radius_skips_the_warm_up():
    obs = NumericObserver("quantile", radius=0.5)
    for v, y in POINTS:
        obs.update(v, y)
    assert obs._radius == 0.5 and obs._warm == []
    assert len(obs._slots) == 6


def test_nominal_observer_picks_the_category_that_isolates_a_class():
    obs = NominalObserver()
    for c, y in [("r", 1), ("r", 1), ("g", 0), ("g", 0), ("b", 0), ("b", 0)]:
        obs.update(c, y)
    gain, cat = obs.best_candidate("gini", {1: 2, 0: 4}, 6)
    assert cat == "r"
    assert gain == pytest.approx(gini({1: 2, 0: 4}))
    assert obs.test_gain("gini", {1: 2, 0: 4}, 6, "g") < gain
    assert obs.test_gain("gini", {1: 2, 0: 4}, 6, "zzz") is None
    assert NominalObserver().best_candidate("gini", {}, 0) == (-1.0, None)


def test_nominal_conditional_probability_is_laplace_smoothed():
    obs = NominalObserver()
    obs.update("r", 1)
    obs.update("r", 1)
    obs.update("g", 0)
    assert obs.cond_proba("r", 1, 2) == pytest.approx((2 + 1) / (2 + 2 + 1))
    assert obs.cond_proba("new", 1, 2) == pytest.approx(1 / (2 + 2 + 1))
    assert math.isfinite(obs.cond_proba("new", 0, 1))


def _target(splitter):
    obs = TargetNumericObserver(splitter)
    for v, y in [
        (1.0, 1.0),
        (2.0, 1.1),
        (3.0, 0.9),
        (7.0, 5.0),
        (8.0, 5.1),
        (9.0, 4.9),
    ]:
        obs.update(v, y)
    return obs


def test_target_observer_ebst_variance_reduction():
    obs = _target("ebst")
    gain, thr = obs.best_candidate()
    assert thr == pytest.approx(5.0)
    assert gain > 3.0
    assert obs.test_gain(5.0) == pytest.approx(gain)
    assert obs.test_gain(100.0) is None


def test_target_observer_quantile_warm_up_and_slots():
    obs = _target("quantile")
    assert obs.best_candidate()[1] == pytest.approx(5.0)
    for i in range(40):
        obs.update(float(i % 10), 1.0 if i % 10 < 5 else 5.0)
    assert obs._radius is not None
    gain, thr = obs.best_candidate()
    assert 3.0 < thr < 7.0 and gain > 1.0


def test_target_nominal_observer():
    obs = TargetNominalObserver()
    for c, y in [("a", 1.0), ("a", 1.2), ("b", 5.0), ("b", 5.2), ("c", 1.1)]:
        obs.update(c, y)
    gain, cat = obs.best_candidate()
    assert cat == "b"
    assert obs.test_gain("b") == pytest.approx(gain)
    assert obs.test_gain("zzz") is None
    assert TargetNominalObserver().best_candidate() == (-1.0, None)
