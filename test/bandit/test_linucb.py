"""Tests for LinUCBDisjoint."""

import numpy as np
import pytest

from dense_armor.utility.bandit import LinUCBDisjoint


def test_linucb_picks_best_arm_from_context():
    rng = np.random.default_rng(0)
    p = LinUCBDisjoint(arms=["left", "right"], alpha=0.3)
    for _ in range(600):
        c = float(rng.normal(0.0, 1.0))
        x = {"c": c}
        a = p.pull(context=x)
        r = 1.0 if (a == "left" and c < 0.0) or (a == "right" and c > 0.0) else 0.0
        _ = p.update(a, r, context=x)
    assert p.pull(context={"c": -1.0}) == "left"
    assert p.pull(context={"c": 1.0}) == "right"


def test_linucb_requires_context_on_pull():
    p = LinUCBDisjoint(arms=[0, 1], alpha=0.1)
    with pytest.raises(ValueError):
        p.pull(context=None)
    with pytest.raises(ValueError):
        p.pull(context={})


def test_linucb_nan_reward_counted():
    p = LinUCBDisjoint(arms=[0, 1], alpha=0.1)
    _ = p.update(0, float("nan"), context={"c": 1.0})
    assert p.n_missing == 1


def test_linucb_missing_context_counted():
    p = LinUCBDisjoint(arms=[0, 1], alpha=0.1)
    _ = p.update(0, 1.0, context=None)
    assert p.n_missing == 1


def test_linucb_unknown_arm_counted():
    p = LinUCBDisjoint(arms=[0, 1], alpha=0.1)
    _ = p.update(99, 1.0, context={"c": 1.0})
    assert p.n_missing == 1


def test_linucb_rejects_bad_alpha():
    for a in (0.0, -0.1):
        with pytest.raises(ValueError):
            LinUCBDisjoint(arms=[0, 1], alpha=a)


def test_linucb_rejects_empty_arms():
    with pytest.raises(ValueError):
        LinUCBDisjoint(arms=[], alpha=0.1)


def test_linucb_clone():
    p = LinUCBDisjoint(arms=["a", "b"], alpha=0.5, seed=3)
    c = p.clone()
    assert type(c) is LinUCBDisjoint
    assert c.alpha == 0.5
    assert c.arms == ("a", "b")


def test_linucb_counts_a_non_numeric_reward_as_missing():
    p = LinUCBDisjoint(arms=["left", "right"], alpha=0.3)
    p.update("left", "x", context={"a": 1.0})
    assert p.n_missing == 1


def test_linucb_unit_test_skips_is_a_set():
    assert LinUCBDisjoint(arms=["a", "b"])._unit_test_skips() == set()
