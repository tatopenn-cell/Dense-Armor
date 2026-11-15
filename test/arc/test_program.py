"""Tests for dense_armor.core.graph.program."""

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program


def test_empty_program_apply_is_identity():
    g = Grid.from_list([[1, 2], [3, 4]])
    assert Program([]).apply(g) == g


def test_apply_in_order():
    g = Grid.from_list([[1, 2], [3, 4]])
    p = Program([dsl.rot90, dsl.hmirror])
    assert p.apply(g) == dsl.hmirror(dsl.rot90(g))


def test_verify_zero_pairs():
    assert Program([dsl.rot90]).verify([]) == 0.0


def test_verify_all_correct():
    g = Grid.from_list([[1, 2], [3, 4]])
    p = Program([dsl.hmirror])
    pairs = [(g, dsl.hmirror(g))]
    assert p.verify(pairs) == 1.0


def test_verify_half_correct():
    g = Grid.from_list([[1, 2], [3, 4]])
    p = Program([dsl.hmirror])
    pairs = [(g, dsl.hmirror(g)), (g, dsl.vmirror(g))]
    assert p.verify(pairs) == 0.5


def test_verify_none_correct():
    g = Grid.from_list([[1, 2], [3, 4]])
    p = Program([dsl.hmirror])
    pairs = [(g, dsl.vmirror(g))]
    assert p.verify(pairs) == 0.0


def test_repr_lists_step_names():
    p = Program([dsl.rot90, dsl.hmirror])
    r = repr(p)
    assert "rot90" in r and "hmirror" in r


def test_eq_and_hash():
    a = Program([dsl.rot90])
    b = Program([dsl.rot90])
    assert a == b
    assert hash(a) == hash(b)
