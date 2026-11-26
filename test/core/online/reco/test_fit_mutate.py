"""fit, mutate and set_params of the recommenders: fit says how to learn, lr and l2 are checked as in the constructor."""

import pytest

from dense_armor.core.online.reco import Baseline, BiasedMF, FunkMF


def _all():
    return [Baseline(lr=0.05, l2=0.0), BiasedMF(k=2, lr=0.05), FunkMF(k=2, lr=0.05)]


@pytest.mark.parametrize("m", _all(), ids=lambda m: type(m).__name__)
def test_fit_refuses_a_table_and_says_how_to_learn(m):
    with pytest.raises(TypeError, match=r"is not fitted on a table: learn with learn_one\(user, item, rating\)"):
        m.fit([("a", "x")], [4.0])


@pytest.mark.parametrize("m", _all(), ids=lambda m: type(m).__name__)
def test_mutate_changes_lr_and_l2_and_returns_the_model(m):
    assert m.mutate({"lr": 0.2, "l2": 0.01}) is m
    assert (m.lr, m.l2) == (0.2, 0.01)


@pytest.mark.parametrize("m", _all(), ids=lambda m: type(m).__name__)
def test_mutate_and_set_params_refuse_what_the_constructor_refuses(m):
    with pytest.raises(ValueError, match="lr"):
        m.mutate({"lr": 0.0})
    with pytest.raises(ValueError, match="l2"):
        m.set_params(l2=-1.0)
    with pytest.raises(ValueError, match="lr"):
        m.set_params(lr=-1.0, l2=0.5)
    assert m.lr == 0.05 and m.l2 != 0.5


@pytest.mark.parametrize("m", _all(), ids=lambda m: type(m).__name__)
def test_mutate_leaves_seed_and_what_was_learned_alone(m):
    m.learn_one("a", "x", 4.0)
    m.mutate({"seed": 9, "mu_": 0.0, "n_seen_": 50, "unknown": 1})
    assert m.seed == 0 and m.n_seen_ == 1 and m.mu_ == 4.0


def test_changing_lr_changes_the_next_step():
    slow, fast = Baseline(lr=0.01), Baseline(lr=0.01)
    fast.mutate({"lr": 0.5})
    for m in (slow, fast):
        m.learn_one("a", "x", 5.0)
        m.learn_one("b", "x", 1.0)
    assert abs(fast.b_u_["b"]) > abs(slow.b_u_["b"])
