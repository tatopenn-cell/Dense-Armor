"""Tests for what the rules of rule_synthesis do to a grid, and for the synthesis of small tasks."""

import pytest

from dense_armor.core.graph import dsl
from dense_armor.core.graph.graph_diff import level2_nodes
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects
from dense_armor.core.graph.rule_synthesis import (
    GLOBAL_EFFECTORS,
    Effector,
    EffectorType,
    Predicate,
    PredicateType,
    Rule,
    anchor,
    apply_global,
    render,
    shape_signature,
    synthesize,
)

# A (colour 1): one cell on the border. B (colour 2): a 2x2 square.
# C (colour 3): a ring with a hole, 8 cells.
GRID = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 3, 0, 3, 0],
    [0, 0, 0, 0, 0, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]
A, B, C = (0, 0), (2, 2), (4, 5)


def _render(rules, rows=GRID):
    g = Grid.from_list(rows)
    return render(g, rules, level2_nodes(g, extract_objects(g))).to_list()


def _rule(kind, *selector, **params):
    preds = tuple(Predicate(k, a) for k, a in selector)
    return Rule(selector=preds, effector=Effector(kind, **params))


def _at(out, cell):
    return out[cell[0]][cell[1]]


# --------------------------------------------------------------
# The predicates: which of A, B, C does each one select?
# --------------------------------------------------------------
P = PredicateType


@pytest.mark.parametrize(
    "selector, selected",
    [
        ([(P.HAS_COLOR, 2)], {2}),
        ([(P.AREA_GT, 1)], {2, 3}),
        ([(P.AREA_LT, 5)], {1, 2}),
        ([(P.IS_LARGEST, None)], {3}),
        ([(P.IS_SMALLEST, None)], {1}),
        ([(P.IS_RECTANGULAR, None)], {1, 2}),
        ([(P.IS_SQUARE, None)], {1, 2, 3}),
        ([(P.RANK, 1)], {3}),
        ([(P.RANK, 2)], {2}),
        ([(P.RANK, 3)], {1}),
        ([(P.TOUCHES_BORDER, None)], {1}),
        ([(P.HAS_HOLE, None)], {3}),
        ([(P.HAS_COLOR, 3), (P.IS_LARGEST, None)], {3}),
        ([(P.AREA_GT, 1), (P.TOUCHES_BORDER, None)], set()),
    ],
)
def test_each_predicate_selects_the_objects_it_describes(selector, selected):
    out = _render([_rule(EffectorType.RECOLOR, *selector, to_color=9)])
    recoloured = {k for k, cell in ((1, A), (2, B), (3, C)) if _at(out, cell) == 9}
    assert recoloured == selected


# --------------------------------------------------------------
# The effectors on objects
# --------------------------------------------------------------
def test_delete_clears_the_selected_objects():
    out = _render([_rule(EffectorType.DELETE, (P.HAS_COLOR, 2))])
    assert _at(out, B) == 0 and _at(out, C) == 3


def test_recolor_by_map_follows_the_mapping():
    rule = _rule(EffectorType.RECOLOR_BY_MAP, mapping=((1, 6), (2, 7)))
    out = _render([rule])
    assert (_at(out, A), _at(out, B), _at(out, C)) == (6, 7, 3)
    no_map = _render([_rule(EffectorType.RECOLOR_BY_MAP)])
    assert no_map == GRID


def test_translate_writes_the_object_at_the_new_place():
    out = _render([_rule(EffectorType.TRANSLATE, (P.HAS_COLOR, 1), dr=1, dc=1)])
    assert out[1][1] == 1


def test_fill_holes_colours_the_cavities_of_the_selected_objects():
    out = _render([_rule(EffectorType.FILL_HOLES, (P.HAS_COLOR, 3), to_color=4)])
    assert out[5][6] == 4 and out[1][1] == 0


def test_fill_bg_colours_the_background():
    out = _render([_rule(EffectorType.FILL_BG, to_color=5)])
    assert out[1][1] == 5 and _at(out, B) == 2


def test_clone_copies_an_object_at_an_offset():
    objs = extract_objects(Grid.from_list(GRID))
    source = next(i for i, o in enumerate(objs) if o.color == 1)
    out = _render([_rule(EffectorType.CLONE, source_idx=source, dr=1, dc=1)])
    assert out[1][1] == 1
    far = _render([_rule(EffectorType.CLONE, source_idx=999, dr=1, dc=1)])
    assert far == GRID


def test_clone_around_draws_the_cells_around_each_selected_cell():
    rule = _rule(
        EffectorType.CLONE_AROUND,
        (P.HAS_COLOR, 1),
        to_color=7,
        deltas=((0, 1), (1, 0)),
    )
    out = _render([rule])
    assert out[0][1] == 7 and out[1][0] == 7
    assert _render([_rule(EffectorType.CLONE_AROUND, (P.HAS_COLOR, 1))]) == GRID


def test_fill_l2_fills_a_node_of_the_second_level():
    g = Grid.from_list(GRID)
    level2 = level2_nodes(g, extract_objects(g))
    cavity = next(
        name
        for name, node in level2.items()
        if name.startswith("cavity") and (5, 6) in node["state"]["cells"]
    )
    rule = Rule(
        selector=(), effector=Effector(EffectorType.FILL_L2, to_color=6, l2_name=cavity)
    )
    assert render(g, [rule], level2).to_list()[5][6] == 6
    missing = Rule(
        selector=(), effector=Effector(EffectorType.FILL_L2, to_color=6, l2_name="none")
    )
    assert render(g, [missing], level2).to_list() == GRID


def test_recolor_relational_takes_the_colour_of_the_anchor():
    rule = Rule(
        selector=(),
        effector=Effector(
            EffectorType.RECOLOR_RELATIONAL,
            anchor_predicate=(Predicate(P.HAS_COLOR, 3),),
            target_predicate=(Predicate(P.HAS_COLOR, 1),),
        ),
    )
    out = _render([rule])
    assert _at(out, A) == 3 and _at(out, B) == 2
    alone = Rule(selector=(), effector=Effector(EffectorType.RECOLOR_RELATIONAL))
    assert _render([alone]) == GRID


def test_recolor_nearest_takes_the_colour_of_the_closest_anchor():
    def nearest(anchor_colour):
        eff = Effector(
            EffectorType.RECOLOR_NEAREST,
            anchor_predicate=(Predicate(P.HAS_COLOR, anchor_colour),),
            target_predicate=(Predicate(P.HAS_COLOR, 1),),
        )
        return _render([Rule(selector=(), effector=eff)])

    assert _at(nearest(2), A) == 2
    assert _at(nearest(9), A) == 1
    assert (
        _at(
            _render(
                [Rule(selector=(), effector=Effector(EffectorType.RECOLOR_NEAREST))]
            ),
            A,
        )
        == 1
    )


# --------------------------------------------------------------
# The global effectors
# --------------------------------------------------------------
GLOBALS = {
    EffectorType.ROT90: dsl.rot90,
    EffectorType.ROT180: dsl.rot180,
    EffectorType.ROT270: dsl.rot270,
    EffectorType.HMIRROR: dsl.hmirror,
    EffectorType.VMIRROR: dsl.vmirror,
    EffectorType.DMIRROR: dsl.dmirror,
    EffectorType.UPSCALE2: lambda g: dsl.upscale(g, 2),
    EffectorType.UPSCALE3: lambda g: dsl.upscale(g, 3),
    EffectorType.DOWNSCALE2: lambda g: dsl.downscale(g, 2),
    EffectorType.DOWNSCALE3: lambda g: dsl.downscale(g, 3),
}


def test_the_global_effectors_are_the_dsl_functions():
    assert set(GLOBALS) == set(GLOBAL_EFFECTORS)
    g = Grid.from_list(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    for kind, fn in GLOBALS.items():
        assert apply_global(g, Effector(kind)) == fn(g)


def test_a_global_effector_is_applied_before_the_object_rules():
    rules = [
        _rule(EffectorType.RECOLOR, (P.HAS_COLOR, 1), to_color=9),
        _rule(EffectorType.HMIRROR),
    ]
    out = _render(rules)
    assert out[0][8] == 9


def test_apply_global_refuses_an_effector_that_is_not_global():
    with pytest.raises(ValueError):
        apply_global(Grid.from_list(GRID), Effector(EffectorType.RECOLOR))


# --------------------------------------------------------------
# Shapes, anchors, cost of a rule
# --------------------------------------------------------------
def test_shape_signature_does_not_depend_on_the_position():
    objs = {o.color: o for o in extract_objects(Grid.from_list(GRID))}
    assert shape_signature(objs[2]) == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    moved = extract_objects(Grid.from_list([[0, 0, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]))
    assert shape_signature([o for o in moved if o.color == 2][0]) == shape_signature(
        objs[2]
    )
    ar, ac = anchor(objs[2])
    assert (ar, ac) == (3.0, 3.0)


def test_the_cost_of_a_rule_grows_with_its_parameters():
    simple = _rule(EffectorType.DELETE)
    precise = _rule(
        EffectorType.RECOLOR,
        (P.HAS_COLOR, 1),
        (P.AREA_GT, 2),
        to_color=5,
    )
    mapped = _rule(EffectorType.RECOLOR_BY_MAP, mapping=((1, 2), (3, 4)))
    assert simple.mdl_bits() > 0
    assert precise.mdl_bits() > simple.mdl_bits()
    assert mapped.mdl_bits() > simple.mdl_bits()


# --------------------------------------------------------------
# Synthesis of small tasks
# --------------------------------------------------------------
def _task(pairs):
    return {"train": [{"input": a, "output": b} for a, b in pairs]}


def _solved(rules, rows, expected):
    g = Grid.from_list(rows)
    return render(g, rules, level2_nodes(g, extract_objects(g))).to_list() == expected


def test_synthesize_finds_a_recolour():
    task = _task(
        [
            ([[1, 0, 0], [0, 2, 0], [0, 0, 0]], [[5, 0, 0], [0, 2, 0], [0, 0, 0]]),
            ([[0, 1, 0], [0, 0, 0], [2, 0, 1]], [[0, 5, 0], [0, 0, 0], [2, 0, 5]]),
        ]
    )
    rules = synthesize(task)
    assert rules is not None
    assert _solved(
        rules, [[1, 1, 0], [0, 0, 2], [0, 0, 0]], [[5, 5, 0], [0, 0, 2], [0, 0, 0]]
    )


def test_synthesize_finds_global_transformations():
    cases = {
        EffectorType.HMIRROR: (
            [[1, 2, 0], [3, 0, 0]],
            [[0, 2, 1], [0, 0, 3]],
            [[4, 0, 0], [5, 6, 0]],
            [[0, 0, 4], [0, 6, 5]],
        ),
        EffectorType.ROT90: (
            [[1, 2], [3, 4]],
            [[2, 4], [1, 3]],
            [[5, 6], [7, 8]],
            [[6, 8], [5, 7]],
        ),
        EffectorType.UPSCALE2: (
            [[1, 2], [3, 4]],
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            [[5, 0], [0, 6]],
            [[5, 5, 0, 0], [5, 5, 0, 0], [0, 0, 6, 6], [0, 0, 6, 6]],
        ),
    }
    for kind, (i1, o1, i2, o2) in cases.items():
        rules = synthesize(_task([(i1, o1), (i2, o2)]))
        assert rules is not None
        assert [r.effector.kind for r in rules] == [kind]


def test_synthesize_gives_nothing_for_no_pairs_or_no_rule():
    assert synthesize({"train": []}) is None
    task = _task(
        [
            ([[1, 0], [0, 0]], [[0, 0], [0, 7]]),
            ([[0, 2], [0, 0]], [[3, 3], [3, 3]]),
        ]
    )
    assert synthesize(task) is None
