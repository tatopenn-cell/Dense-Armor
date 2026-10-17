"""Tests for EpsilonGreedy, UCB1, UCB1Tuned."""

import numpy as np
import pytest

from dense_armor.utility.bandit import UCB1, EpsilonGreedy, UCB1Tuned


def test_ucb1_plays_every_arm_once():
    p = UCB1(arms=["a", "b", "c"], seed=0)
    seen = []
    for _ in range(3):
        a = p.pull()
        seen.append(a)
        _ = p.update(a, 0.0)
    assert set(seen) == {"a", "b", "c"}


def test_ucb1_finds_best_arm():
    rng = np.random.default_rng(0)
    p = UCB1(arms=[0, 1, 2], seed=0)
    for _ in range(600):
        a = p.pull()
        r = float(rng.binomial(1, [0.2, 0.8, 0.5][a]))
        _ = p.update(a, r)
    best = p.pull()
    assert p.counts_[1] > p.counts_[0]
    assert p.counts_[1] > p.counts_[2]
    assert best == 1


def test_ucb1_rejects_empty_arms():
    with pytest.raises(ValueError):
        UCB1(arms=[])


def test_ucb1_nan_reward_counted():
    p = UCB1(arms=[0, 1], seed=0)
    _ = p.pull()
    _ = p.update(0, float("nan"))
    assert p.n_missing == 1
    assert p.n_seen_ == 0


def test_ucb1_unknown_arm_counted():
    p = UCB1(arms=[0, 1], seed=0)
    _ = p.update(99, 1.0)
    assert p.n_missing == 1


def test_ucb1_clone_keeps_arms():
    p = UCB1(arms=["a", "b"], seed=7)
    c = p.clone()
    assert type(c) is UCB1
    assert c.arms == ("a", "b")
    assert c.seed == 7


def test_epsilon_greedy_rejects_bad_epsilon():
    for e in (-0.1, 1.1):
        with pytest.raises(ValueError):
            EpsilonGreedy(arms=[0, 1], epsilon=e)


def test_epsilon_greedy_zero_explores():
    p = EpsilonGreedy(arms=[0, 1], epsilon=0.0, seed=0)
    for _ in range(40):
        a = p.pull()
        _ = p.update(a, 1.0 if a == 1 else 0.0)
    assert p.pull() == 1


def test_epsilon_greedy_warmup_pulls_each_arm():
    p = EpsilonGreedy(arms=["a", "b", "c"], epsilon=0.0, seed=0)
    seen = []
    for _ in range(3):
        a = p.pull()
        seen.append(a)
        _ = p.update(a, 0.0)
    assert set(seen) == {"a", "b", "c"}


def test_epsilon_greedy_full_explores():
    p = EpsilonGreedy(arms=[0, 1, 2], epsilon=1.0, seed=0)
    seen = set()
    for _ in range(60):
        a = p.pull()
        seen.add(a)
        _ = p.update(a, 0.0)
    assert seen == {0, 1, 2}


def test_ucb1_tuned_finds_best_arm():
    rng = np.random.default_rng(0)
    p = UCB1Tuned(arms=[0, 1], seed=0)
    for _ in range(400):
        a = p.pull()
        r = float(rng.binomial(1, [0.2, 0.8][a]))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]


def test_ucb1_tuned_nan_counted():
    p = UCB1Tuned(arms=[0, 1], seed=0)
    _ = p.update(0, float("nan"))
    assert p.n_missing == 1


def test_ucb1_tuned_clone():
    p = UCB1Tuned(arms=[0, 1], seed=3)
    c = p.clone()
    assert type(c) is UCB1Tuned
    assert c.seed == 3


def test_epsilon_greedy_falls_back_to_the_first_arm_when_means_are_nan():
    p = EpsilonGreedy(arms=["a", "b"], epsilon=0.0, seed=0)
    p.counts_ = {"a": 1, "b": 1}
    p.means_ = {"a": float("nan"), "b": float("nan")}
    assert p.pull() == "a"


def test_epsilon_greedy_counts_invalid_updates_as_missing():
    p = EpsilonGreedy(arms=["a", "b"], seed=0)
    p.update("a", float("nan"))
    p.update("zzz", 1.0)
    assert p.n_missing == 2


def test_ucb1_index_is_infinite_for_an_unplayed_arm():
    p = UCB1(arms=["a", "b"], seed=0)
    assert p._index("a", 1) == float("inf")


def test_ucb1_tuned_index_is_infinite_for_an_unplayed_arm():
    p = UCB1Tuned(arms=["a", "b"], seed=0)
    assert p._index("a", 1) == float("inf")
