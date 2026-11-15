"""Tests for core/graph/grid.py."""

from dense_armor.core.graph.grid import Grid


def test_from_list_to_list_roundtrip():
    rows = [[0, 1, 2], [3, 4, 5]]
    g = Grid.from_list(rows)
    assert g.to_list() == rows


def test_hash_equal_for_equal_grids():
    a = Grid.from_list([[0, 1], [1, 0]])
    b = Grid.from_list([[0, 1], [1, 0]])
    assert hash(a) == hash(b)
    assert a == b


def test_shape():
    g = Grid.from_list([[0, 0, 0], [0, 0, 0]])
    assert g.shape() == (2, 3)


def test_at_and_colors():
    g = Grid.from_list([[1, 2], [3, 4]])
    assert g.at(0, 0) == 1
    assert g.at(1, 1) == 4
    assert g.colors() == {1, 2, 3, 4}


def test_crop():
    g = Grid.from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    cropped = g.crop((1, 1, 3, 3))
    assert cropped.to_list() == [[4, 5], [7, 8]]


def test_pad():
    g = Grid.from_list([[1, 2]])
    padded = g.pad(1, color=0)
    assert padded.to_list() == [[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]


def test_invalid_color_raises():
    import pytest

    with pytest.raises(ValueError):
        Grid.from_list([[0, 99]])


def test_ragged_raises():
    import pytest

    with pytest.raises(ValueError):
        Grid.from_list([[0, 1], [0]])
