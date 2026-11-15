"""Tests for dense_armor.core.graph.dsl."""

import pytest

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid


def _g():
    return Grid.from_list([[1, 2], [3, 4]])


def test_rot90():
    assert dsl.rot90(_g()).to_list() == [[2, 4], [1, 3]]


def test_rot180():
    assert dsl.rot180(_g()).to_list() == [[4, 3], [2, 1]]


def test_rot270():
    assert dsl.rot270(_g()).to_list() == [[3, 1], [4, 2]]


def test_rot90_and_rot270_are_inverse():
    assert dsl.rot270(dsl.rot90(_g())) == _g()


def test_hmirror():
    assert dsl.hmirror(_g()).to_list() == [[2, 1], [4, 3]]


def test_vmirror():
    assert dsl.vmirror(_g()).to_list() == [[3, 4], [1, 2]]


def test_dmirror():
    assert dsl.dmirror(_g()).to_list() == [[1, 3], [2, 4]]


def test_hmirror_is_involutive():
    assert dsl.hmirror(dsl.hmirror(_g())) == _g()


def test_fill_all():
    assert dsl.fill(_g(), 7).to_list() == [[7, 7], [7, 7]]


def test_fill_mask():
    assert dsl.fill(_g(), 0, {(0, 0), (1, 1)}).to_list() == [[0, 2], [3, 0]]


def test_fill_mask_clips_out_of_bounds():
    assert dsl.fill(_g(), 0, {(5, 5)}).to_list() == [[1, 2], [3, 4]]


def test_recolor():
    assert dsl.recolor(_g(), 2, 9).to_list() == [[1, 9], [3, 4]]


def test_shift_down():
    assert dsl.shift(_g(), 1, 0).to_list() == [[0, 0], [1, 2]]


def test_shift_right():
    assert dsl.shift(_g(), 0, 1).to_list() == [[0, 1], [0, 3]]


def test_shift_zero_is_identity():
    assert dsl.shift(_g(), 0, 0) == _g()


def test_shift_fill_color():
    assert dsl.shift(_g(), 1, 0, fill_color=9).to_list() == [[9, 9], [1, 2]]


def test_upscale():
    assert dsl.upscale(_g(), 2).to_list() == [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]


def test_upscale_rejects_zero():
    with pytest.raises(ValueError):
        dsl.upscale(_g(), 0)


def test_downscale():
    big = Grid.from_list([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    assert dsl.downscale(big, 2).to_list() == [[1, 2], [3, 4]]


def test_downscale_rejects_zero():
    with pytest.raises(ValueError):
        dsl.downscale(_g(), 0)


def test_upscale_then_downscale_is_identity():
    assert dsl.downscale(dsl.upscale(_g(), 3), 3) == _g()
