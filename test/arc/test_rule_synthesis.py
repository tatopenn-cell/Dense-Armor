"""Tests for rule_synthesis v2."""

from dense_armor.core.graph.graph_diff import level2_nodes
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects
from dense_armor.core.graph.rule_synthesis import (
    Effector,
    EffectorType,
    Predicate,
    PredicateType,
    Rule,
    render,
    synthesize,
)


def test_recolor_rule():
    g = Grid.from_list([[1, 0], [0, 0]])
    rule = Rule(
        selector=(Predicate(PredicateType.HAS_COLOR, 1),),
        effector=Effector(EffectorType.RECOLOR, to_color=2),
    )
    out = render(g, [rule], level2_nodes(g, extract_objects(g)))
    assert out.to_list() == [[2, 0], [0, 0]]


def test_delete_rule():
    g = Grid.from_list([[1, 0], [0, 2]])
    rule = Rule(
        selector=(Predicate(PredicateType.HAS_COLOR, 1),),
        effector=Effector(EffectorType.DELETE),
    )
    out = render(g, [rule], level2_nodes(g, extract_objects(g)))
    assert out.to_list() == [[0, 0], [0, 2]]


def test_synthesize_recolor_task():
    task = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]},
            {"input": [[1, 1], [0, 0]], "output": [[2, 2], [0, 0]]},
        ],
        "test": [{"input": [[1, 0], [0, 1]]}],
    }
    rules = synthesize(task)
    assert rules is not None


def test_synthesize_returns_none_for_unsolvable():
    # Merge of two diagonal 1s into a solid 1 block, with the
    # background filled by the object's color. No single effector
    # in v5 (which includes global rot/mirror/scale) does this:
    # fill_bg takes a fixed color, recolor_by_map is a per-cell
    # map, and there is no merge effector. Genuinely out of scope.
    task = {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[1, 1], [1, 1]]},
            {"input": [[2, 0], [0, 2]], "output": [[2, 2], [2, 2]]},
        ],
        "test": [],
    }
    rules = synthesize(task)
    assert rules is None


def test_shape_signature_invariant_to_translation():
    from dense_armor.core.graph.rule_synthesis import shape_signature

    # same horizontal domino, translated: (0,0)+(0,1) vs (1,0)+(1,1)
    g1 = Grid.from_list([[1, 1], [0, 0]])
    g2 = Grid.from_list([[0, 0], [1, 1]])
    sig1 = {shape_signature(o) for o in extract_objects(g1) if o.area == 2}
    sig2 = {shape_signature(o) for o in extract_objects(g2) if o.area == 2}
    assert sig1 == sig2
