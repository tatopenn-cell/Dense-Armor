"""Tests for apply_to_test."""

from dense_armor.core.graph.apply_to_test import apply_to_test
from dense_armor.core.graph.rule_synthesis import (
    Effector,
    EffectorType,
    Predicate,
    PredicateType,
    Rule,
)


def test_apply_to_test_recolor():
    rule = Rule(
        selector=(Predicate(PredicateType.HAS_COLOR, 1),),
        effector=Effector(EffectorType.RECOLOR, to_color=5),
    )
    task = {
        "train": [],
        "test": [{"input": [[1, 0]]}],
    }
    out = apply_to_test([rule], task)
    assert out["test"][0]["output"] == [[5, 0]]
