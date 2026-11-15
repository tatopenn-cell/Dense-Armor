"""Tests for dense_armor.core.graph.refine."""

import pytest

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program
from dense_armor.core.graph.refine import refine


def test_refine_never_worse():
    g = Grid.from_list([[1, 2], [3, 4]])
    start = Program([dsl.rot90])
    pairs = [(g, dsl.hmirror(g))]
    refined = refine(start, [dsl.hmirror], pairs, iterations=5, seed=0)
    assert refined.verify(pairs) >= start.verify(pairs)


def test_refine_reaches_perfect_on_simple_task():
    g = Grid.from_list([[1, 2], [3, 4]])
    start = Program([dsl.rot90])
    pairs = [(g, dsl.hmirror(g))]
    refined = refine(start, [dsl.hmirror], pairs, iterations=50, seed=0)
    assert refined.verify(pairs) == 1.0


def test_refine_no_primitives_returns_unchanged():
    start = Program([dsl.rot90])
    pairs = [(Grid.from_list([[1]]), Grid.from_list([[1]]))]
    refined = refine(start, [], pairs, iterations=10, seed=0)
    assert refined == start


def test_refine_rejects_negative_iterations():
    with pytest.raises(ValueError):
        refine(Program([]), [dsl.rot90], [], iterations=-1, seed=0)


def test_refine_is_deterministic_with_seed():
    g = Grid.from_list([[1, 2], [3, 4]])
    start = Program([dsl.rot90])
    pairs = [(g, dsl.hmirror(g))]
    a = refine(start, [dsl.hmirror, dsl.vmirror], pairs, iterations=30, seed=7)
    b = refine(start, [dsl.hmirror, dsl.vmirror], pairs, iterations=30, seed=7)
    assert a == b
