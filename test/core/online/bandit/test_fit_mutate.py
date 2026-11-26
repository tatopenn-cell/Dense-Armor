"""fit and mutate of the bandit policies: fit says how to learn, mutate changes the settings read at every pull."""

import pytest

from dense_armor.checks.check_estimator import check_mutate_idempotent
from dense_armor.core.online.bandit import (
    UCB1,
    BayesUCB,
    EpsilonGreedy,
    Exp3,
    LinUCBDisjoint,
    RobotKpSelector,
    ThompsonSampling,
    UCB1Tuned,
)

ARMS = ["twap", "vwap", "pov"]


def _policies():
    return [
        EpsilonGreedy(ARMS, epsilon=0.1, seed=0),
        UCB1(ARMS, seed=0),
        UCB1Tuned(ARMS, seed=0),
        ThompsonSampling(ARMS, family="bernoulli", seed=0),
        BayesUCB(ARMS, family="bernoulli", horizon=100, seed=0),
        Exp3(ARMS, gamma=0.1, seed=0),
        LinUCBDisjoint(ARMS, alpha=0.3, feature_keys=("a",)),
    ]


@pytest.mark.parametrize("policy", _policies(), ids=lambda p: type(p).__name__)
def test_fit_refuses_a_table_and_says_how_to_learn(policy):
    with pytest.raises(TypeError, match=r"is not fitted on a table.*pull\(\) and update\(arm, reward\).*offline_replay"):
        policy.fit([{"arm": "pov"}], [1.0])


def test_fit_of_the_robot_selector_names_its_round():
    with pytest.raises(TypeError, match=r"learn with run_round\(\)"):
        RobotKpSelector(arms=[1.0, 2.0]).fit([{"x": 1.0}])


def test_mutate_changes_epsilon_and_returns_the_policy():
    p = EpsilonGreedy(ARMS, epsilon=0.1, seed=0)
    assert p.mutate({"epsilon": 0.2}) is p
    assert p.epsilon == 0.2
    assert p.get_params()["epsilon"] == 0.2


def test_mutate_epsilon_changes_the_behaviour():
    p = EpsilonGreedy(ARMS, epsilon=1.0, seed=0)
    for arm, r in (("twap", 0.0), ("vwap", 0.0), ("pov", 1.0)):
        p.update(arm, r)
    assert p.mutate({"epsilon": 0.0}).pull() == "pov"


def test_mutate_refuses_what_the_constructor_refuses_and_changes_nothing():
    p = EpsilonGreedy(ARMS, epsilon=0.1, seed=0)
    with pytest.raises(ValueError, match="epsilon"):
        p.mutate({"epsilon": 5})
    assert p.epsilon == 0.1
    a = LinUCBDisjoint(ARMS, alpha=0.3)
    with pytest.raises(ValueError, match="alpha"):
        a.mutate({"alpha": 0.0})
    assert a.alpha == 0.3
    g = ThompsonSampling(ARMS, family="gaussian", sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        g.mutate({"sigma": -1})
    assert g.sigma == 1.0


def test_mutate_changes_the_other_declared_settings():
    assert LinUCBDisjoint(ARMS, alpha=0.3).mutate({"alpha": 0.5}).alpha == 0.5
    b = BayesUCB(ARMS, family="bernoulli", horizon=100, seed=0).mutate({"c": 1, "horizon": 200})
    assert (b.c, b.horizon) == (1.0, 200)
    assert ThompsonSampling(ARMS, family="gaussian").mutate({"sigma": 2}).sigma == 2.0


def test_mutate_leaves_the_rest_alone():
    p = EpsilonGreedy(ARMS, epsilon=0.1, seed=3)
    p.update("pov", 1.0)
    p.mutate({"seed": 9, "arms": ("x",), "n_seen_": 50, "unknown": 1})
    assert p.seed == 3 and p.arms == tuple(ARMS) and p.n_seen_ == 1
    u = UCB1(ARMS, seed=0).mutate({"seed": 5})
    assert u.seed == 0


@pytest.mark.parametrize("policy", _policies(), ids=lambda p: type(p).__name__)
def test_mutate_is_idempotent_for_every_policy(policy):
    check_mutate_idempotent(policy)
