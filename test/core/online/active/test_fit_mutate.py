"""fit and mutate of the active strategies, and the checks of theta and s."""

import pytest

from dense_armor.checks.check_estimator import check_mutate_idempotent
from dense_armor.core.online.active import (
    FixedUncertainty,
    RandomStrategy,
    VariableRandomizedUncertainty,
    VariableUncertainty,
)


def _all():
    return [
        RandomStrategy(budget=0.1, seed=0),
        FixedUncertainty(theta=0.7),
        VariableUncertainty(budget=0.1, s=0.01, seed=0),
        VariableRandomizedUncertainty(budget=0.1, s=0.01, delta=0.05, seed=0),
    ]


@pytest.mark.parametrize("s", _all(), ids=lambda s: type(s).__name__)
def test_fit_refuses_a_table_and_says_what_to_use(s):
    with pytest.raises(TypeError, match=r"is not fitted on a table.*should_label\(x, model\)"):
        s.fit([{"score": 1.0}], [1])


@pytest.mark.parametrize("s", _all(), ids=lambda s: type(s).__name__)
def test_mutate_changes_the_budget_and_returns_the_strategy(s):
    assert s.mutate({"budget": 0.3}) is s
    assert s.budget == 0.3 and s.get_params()["budget"] == 0.3


@pytest.mark.parametrize("s", _all(), ids=lambda s: type(s).__name__)
def test_mutate_refuses_a_bad_budget_and_changes_nothing(s):
    with pytest.raises(ValueError, match="budget"):
        s.mutate({"budget": 0.0})
    assert s.budget == 0.1


def test_mutate_changes_theta_s_and_delta_with_the_constructor_checks():
    f = FixedUncertainty(theta=0.7)
    assert f.mutate({"theta": 0.6}).theta == 0.6
    with pytest.raises(ValueError, match="theta"):
        f.mutate({"theta": 1.0})
    assert f.theta == 0.6
    v = VariableUncertainty(budget=0.1, s=0.01)
    assert v.mutate({"s": 0.02}).s == 0.02
    with pytest.raises(ValueError, match="s must"):
        v.mutate({"s": 0})
    r = VariableRandomizedUncertainty(budget=0.1, s=0.01, delta=0.05)
    assert r.mutate({"delta": 0.2}).delta == 0.2
    with pytest.raises(ValueError, match="delta"):
        r.mutate({"delta": -1})


def test_mutate_leaves_seed_and_counts_alone():
    v = VariableUncertainty(budget=0.1, seed=3)
    v.mutate({"seed": 9, "n_seen_": 50, "unknown": 1})
    assert v.seed == 3 and v.n_seen_ == 0


def test_constructors_check_theta_and_s():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="theta"):
            FixedUncertainty(theta=bad)
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError, match="s must"):
            VariableUncertainty(s=bad)


@pytest.mark.parametrize("s", _all(), ids=lambda s: type(s).__name__)
def test_mutate_is_idempotent_for_every_strategy(s):
    check_mutate_idempotent(s)
