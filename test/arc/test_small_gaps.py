"""Tests for the small branches of program, graph_diff, solver and vsa_abduction that the larger tests do not reach."""

import numpy as np

from dense_armor.core.graph import dsl
from dense_armor.core.graph.graph_diff import hungarian_match
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects
from dense_armor.core.graph.program import Program
from dense_armor.core.graph.solver import solve
from dense_armor.core.graph.vsa_abduction import (
    _input_object_for,
    _matches_grid,
)
from dense_armor.core.graph.vsa_hypotheses import VSAObject
from dense_armor.core.graph.vsa_objects import ObjectEncoder


def _g(rows):
    return Grid.from_list(rows)


def _raises(g):
    raise ValueError("no object")


def test_a_step_that_raises_counts_the_pair_as_not_solved():
    pairs = [(_g([[1, 0]]), _g([[1, 0]]))]
    assert Program([_raises]).verify(pairs) == 0.0
    assert Program([]).verify(pairs) == 1.0


def test_the_repr_lists_the_steps_and_a_program_is_not_equal_to_a_tuple():
    p = Program([dsl.rot90, dsl.hmirror])
    assert repr(p) == "Program(rot90, hmirror)"
    assert p != (dsl.rot90, dsl.hmirror)
    assert p == Program([dsl.rot90, dsl.hmirror])


def test_the_matching_of_empty_object_lists():
    objs = extract_objects(_g([[1, 0], [0, 0]]))
    assert hungarian_match([], []) == ([], [], [])
    assert hungarian_match([], objs) == ([], list(range(len(objs))), [])
    assert hungarian_match(objs, []) == ([], [], list(range(len(objs))))


def test_the_matching_pairs_a_changed_colour_and_size_with_its_object():
    a = extract_objects(_g([[0, 0, 0], [0, 1, 1], [0, 0, 0]]))
    b = extract_objects(_g([[0, 0, 0], [0, 2, 0], [0, 0, 0]]))
    matched, created, deleted = hungarian_match(a, b)
    assert len(matched) == 2 and created == [] and deleted == []


def test_solve_without_any_primitive_gives_the_empty_program():
    task = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}
    assert solve(task, []) == Program([])


def test_matches_grid_checks_the_bounds_and_the_colour():
    g = _g([[1, 0], [0, 2]])
    assert _matches_grid(frozenset({(0, 0)}), 1, g)
    assert not _matches_grid(frozenset({(0, 0)}), 2, g)
    assert not _matches_grid(frozenset({(0, 2)}), 1, g)
    assert not _matches_grid(frozenset({(-1, 0)}), 1, g)


def test_the_input_object_is_the_most_similar_one_and_none_for_no_objects():
    enc = ObjectEncoder(n=256)
    a = VSAObject(cells=frozenset({(0, 0)}), colour=1, sp=enc.colour_sps[1])
    b = VSAObject(cells=frozenset({(1, 1)}), colour=2, sp=enc.colour_sps[2])
    assert _input_object_for(a, []) is None
    assert _input_object_for(a, [b, a]) is a
    assert isinstance(a.sp, np.ndarray)
