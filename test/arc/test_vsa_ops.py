"""Tests for the VSA operations and the rendering of the VSA solver."""

import numpy as np

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_operations import (
    OperationParams,
    OperationResult,
    OperationType,
    execute_action,
)
from dense_armor.core.graph.vsa_solver import _render_into


def _encoder() -> ObjectEncoder:
    return ObjectEncoder(n=64)


def test_generate_puts_the_object_at_the_given_centre():
    grid = Grid.from_list([[0] * 5 for _ in range(5)])
    params = OperationParams(colour=7, centre=(1.5, 1.5), shape_dims=(2, 2))
    out = execute_action(
        frozenset(), 0, OperationType.GENERATE, params, grid, _encoder()
    )
    assert out[0].cells == frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
    assert out[0].colour == 7


def test_generate_odd_size_is_centred_on_a_cell():
    grid = Grid.from_list([[0] * 5 for _ in range(5)])
    params = OperationParams(colour=3, centre=(2.0, 2.0), shape_dims=(3, 3))
    out = execute_action(
        frozenset(), 0, OperationType.GENERATE, params, grid, _encoder()
    )
    assert out[0].cells == frozenset((r, c) for r in (1, 2, 3) for c in (1, 2, 3))


def test_same_size_rendering_keeps_the_coordinates():
    obj = OperationResult(cells=frozenset({(3, 4)}), colour=5, sp=np.zeros(1))
    kept = _render_into([obj], 6, 6, anchor=False)
    anchored = _render_into([obj], 6, 6)
    assert kept.at(3, 4) == 5
    assert anchored.at(0, 0) == 5
