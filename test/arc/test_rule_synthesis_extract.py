"""Tests for the effectors that ``extract_positives`` finds from a pair of grids, and for the synthesis built on them."""

from dense_armor.core.graph.graph_diff import diff_graphs, level2_nodes
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects
from dense_armor.core.graph.rule_synthesis import (
    EffectorType,
    extract_positives,
    render,
    synthesize,
)


def _blank(h, w, cells):
    rows = [[0] * w for _ in range(h)]
    for (r, c), v in cells.items():
        rows[r][c] = v
    return rows


def _around(h, w, r, c):
    rows = _blank(h, w, {(r, c): 2})
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                rows[r + dr][c + dc] = 3
    return rows


def _positives(rows_in, rows_out):
    gi, go = Grid.from_list(rows_in), Grid.from_list(rows_out)
    diff = diff_graphs(gi, go)
    return extract_positives(diff, gi, go)


def _by_kind(positives):
    return {e.kind: (e, idx) for e, idx in positives.items()}


def _solve(pairs, test):
    task = {
        "train": [{"input": i, "output": o} for i, o in pairs],
        "test": [{"input": test}],
    }
    rules = synthesize(task)
    if rules is None:
        return None, None
    g = Grid.from_list(test)
    out = render(g, rules, level2_nodes(g, extract_objects(g)))
    return rules, out.to_list()


def test_a_moved_object_gives_a_translate_with_the_offset():
    one = _blank(3, 4, {(0, 0): 1})
    moved = _blank(3, 4, {(1, 1): 1})
    eff, idx = _by_kind(_positives(one, moved))[EffectorType.TRANSLATE]
    assert (eff.dr, eff.dc) == (1, 1) and idx == [0]


def test_a_missing_object_gives_a_delete_on_its_index():
    before = _blank(2, 3, {(0, 0): 1, (0, 2): 2})
    after = _blank(2, 3, {(0, 0): 1})
    eff, idx = _by_kind(_positives(before, after))[EffectorType.DELETE]
    assert idx == [2]


def test_a_new_ring_around_a_single_cell_gives_clone_around():
    eff, idx = _by_kind(_positives(_blank(5, 5, {(2, 2): 2}), _around(5, 5, 2, 2)))[
        EffectorType.CLONE_AROUND
    ]
    assert eff.to_color == 3 and len(eff.deltas) == 8 and (0, 0) not in eff.deltas
    assert idx == [1]


def test_a_new_copy_of_a_shape_gives_a_clone_with_the_offset():
    before = _blank(3, 6, {(0, 0): 1, (0, 1): 1})
    after = _blank(3, 6, {(0, 0): 1, (0, 1): 1, (2, 3): 1, (2, 4): 1})
    eff, idx = _by_kind(_positives(before, after))[EffectorType.CLONE]
    assert (eff.dr, eff.dc, eff.to_color) == (2, 3, 1) and idx == [0]


def test_synthesize_finds_the_ring_around_a_cell_and_draws_it_elsewhere():
    pairs = [
        (_blank(5, 5, {(2, 2): 2}), _around(5, 5, 2, 2)),
        (_blank(6, 6, {(3, 3): 2}), _around(6, 6, 3, 3)),
    ]
    rules, out = _solve(pairs, _blank(6, 6, {(1, 4): 2}))
    assert [r.effector.kind for r in rules] == [EffectorType.CLONE_AROUND]
    assert out == _around(6, 6, 1, 4)


def test_synthesize_finds_the_deletion_of_the_objects_of_one_colour():
    pairs = [
        (_blank(4, 4, {(0, 0): 1, (2, 2): 2}), _blank(4, 4, {(2, 2): 2})),
        (_blank(4, 4, {(1, 1): 1, (3, 3): 2}), _blank(4, 4, {(3, 3): 2})),
    ]
    rules, out = _solve(pairs, _blank(4, 4, {(0, 3): 1, (1, 1): 2}))
    assert [r.effector.kind for r in rules] == [EffectorType.DELETE]
    assert out == _blank(4, 4, {(1, 1): 2})


def test_synthesize_finds_one_recolour_per_colour():
    pairs = [
        (_blank(3, 3, {(0, 0): 1, (2, 2): 2}), _blank(3, 3, {(0, 0): 3, (2, 2): 4})),
        (_blank(3, 3, {(1, 0): 1, (1, 2): 2}), _blank(3, 3, {(1, 0): 3, (1, 2): 4})),
    ]
    rules, out = _solve(pairs, _blank(3, 3, {(0, 2): 1, (2, 0): 2}))
    assert {r.effector.to_color for r in rules} == {3, 4}
    assert out == _blank(3, 3, {(0, 2): 3, (2, 0): 4})
