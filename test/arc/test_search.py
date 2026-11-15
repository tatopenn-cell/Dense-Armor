"""Tests for dense_armor.core.graph.search."""

import pytest

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program
from dense_armor.core.graph.search import explore


def test_explore_finds_direct_program():
    g = Grid.from_list([[1, 2], [3, 4]])
    pairs = [(g, dsl.hmirror(g))]
    found = explore([dsl.hmirror], pairs, max_depth=1, beam_width=1)
    assert found[0].verify(pairs) == 1.0


def test_explore_depth_two():
    g = Grid.from_list([[1, 2], [3, 4]])
    target = dsl.rot90(dsl.hmirror(g))
    pairs = [(g, target)]
    found = explore([dsl.hmirror, dsl.rot90], pairs, max_depth=2, beam_width=3)
    assert any(p.verify(pairs) == 1.0 for p in found)


def test_explore_rejects_negative_depth():
    with pytest.raises(ValueError):
        explore([dsl.hmirror], [], max_depth=-1, beam_width=1)


def test_explore_rejects_zero_beam():
    with pytest.raises(ValueError):
        explore([dsl.hmirror], [], max_depth=1, beam_width=0)


def test_explore_empty_primitives_returns_identity():
    g = Grid.from_list([[1, 2], [3, 4]])
    found = explore([], [(g, g)], max_depth=3, beam_width=2)
    assert found == [Program([])]


def test_explore_scores_descending():
    g = Grid.from_list([[1, 2], [3, 4]])
    pairs = [(g, dsl.hmirror(g)), (g, dsl.vmirror(g))]
    found = explore([dsl.hmirror, dsl.vmirror], pairs, max_depth=1, beam_width=2)
    scores = [p.verify(pairs) for p in found]
    assert scores == sorted(scores, reverse=True)


def test_explore_keeps_the_shorter_program_of_an_earlier_depth():
    g = Grid.from_list([[1, 2], [3, 4]])
    pairs = [(g, dsl.rot90(g))]
    found = explore([dsl.rot90, dsl.rot180], pairs, max_depth=2, beam_width=3)
    assert found[0].verify(pairs) == 1.0
    assert len(found[0].steps) == 1
