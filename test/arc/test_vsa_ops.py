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


# --------------------------------------------------------------
# The other operations of Table 1, one by one
# --------------------------------------------------------------
BLOCK = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
BIG_BLOCK = frozenset((r, c) for r in (1, 2, 3) for c in (1, 2, 3))
RING = BIG_BLOCK - {(2, 2)}


def _run(op, cells=BLOCK, colour=1, grid=None, **params):
    grid = grid or Grid.from_list([[0] * 6 for _ in range(6)])
    return execute_action(
        cells, colour, op, OperationParams(**params), grid, _encoder()
    )


def test_identity_keeps_the_cells_inside_the_grid():
    out = _run(OperationType.IDENTITY, cells=BLOCK | {(9, 9)})
    assert out[0].cells == BLOCK and out[0].colour == 1


def test_extract_crops_the_object_to_the_origin():
    out = _run(OperationType.EXTRACT)
    assert out[0].cells == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})


def test_recolour_changes_only_the_colour():
    out = _run(OperationType.RECOLOUR, colour=4)
    assert (out[0].cells, out[0].colour) == (BLOCK, 4)
    assert _run(OperationType.RECOLOUR)[0].colour == 1


def test_recentre_moves_the_object_to_the_centre():
    out = _run(OperationType.RECENTRE, centre=(3.5, 3.5))
    assert out[0].cells == frozenset((r + 2, c + 2) for r, c in BLOCK)
    assert _run(OperationType.RECENTRE)[0].cells == BLOCK
    assert _run(OperationType.RECENTRE, centre=(40.0, 40.0)) == []


def test_reshape_draws_the_new_shape_around_the_centre():
    shape = frozenset({(0, 0), (0, 1)})
    out = _run(OperationType.RESHAPE, cells=BIG_BLOCK, shape=shape)
    assert out[0].cells == frozenset({(2, 2), (2, 3)})
    assert _run(OperationType.RESHAPE)[0].cells == BLOCK


def test_move_translates_and_drops_what_leaves_the_grid():
    out = _run(OperationType.MOVE, amount=(1, 2))
    assert out[0].cells == frozenset((r + 1, c + 2) for r, c in BLOCK)
    assert _run(OperationType.MOVE)[0].cells == BLOCK
    assert _run(OperationType.MOVE, amount=(20, 0)) == []


def _grid_with(cells, extra=()):
    rows = [[0] * 6 for _ in range(6)]
    for r, c in cells:
        rows[r][c] = 1
    for r, c in extra:
        rows[r][c] = 5
    return Grid.from_list(rows)


def test_gravity_moves_until_the_border_or_another_object():
    free = _run(OperationType.GRAVITY, grid=_grid_with(BLOCK), direction="down")
    assert free[0].cells == frozenset((r + 3, c) for r, c in BLOCK)
    blocked = _run(
        OperationType.GRAVITY, grid=_grid_with(BLOCK, extra=[(4, 1)]), direction="down"
    )
    assert blocked[0].cells == frozenset((r + 1, c) for r, c in BLOCK)
    assert _run(OperationType.GRAVITY, direction="sideways")[0].cells == BLOCK
    assert _run(OperationType.GRAVITY)[0].cells == BLOCK


def test_grow_extends_the_object_in_one_direction():
    out = _run(OperationType.GROW, grid=_grid_with(BLOCK), direction="right")
    assert out[0].cells == frozenset((r, c) for r in (1, 2) for c in range(1, 6))
    assert _run(OperationType.GROW, direction="sideways")[0].cells == BLOCK
    assert _run(OperationType.GROW)[0].cells == BLOCK


def test_fill_and_hollow_are_opposite():
    assert _run(OperationType.FILL, cells=RING)[0].cells == BIG_BLOCK
    assert _run(OperationType.HOLLOW, cells=BIG_BLOCK)[0].cells == RING


def test_generate_without_parameters_and_outside_the_grid_gives_nothing():
    assert _run(OperationType.GENERATE, cells=frozenset()) == []
    far = OperationParams(colour=2, centre=(50.0, 50.0), shape_dims=(1, 1))
    assert (
        execute_action(
            frozenset(),
            0,
            OperationType.GENERATE,
            far,
            Grid.from_list([[0] * 4 for _ in range(4)]),
            _encoder(),
        )
        == []
    )


def test_an_operation_on_no_cells_gives_nothing():
    assert _run(OperationType.IDENTITY, cells=frozenset()) == []


def test_result_to_grid_draws_the_objects_from_their_top_left_corner():
    from dense_armor.core.graph.vsa_operations import result_to_grid

    assert result_to_grid([]).to_list() == [[0]]
    a = OperationResult(frozenset({(2, 3)}), 5, np.zeros(1))
    b = OperationResult(frozenset({(3, 3)}), 6, np.zeros(1))
    assert result_to_grid([a, b]).to_list() == [[5], [6]]
    wide = OperationResult(frozenset({(0, 0), (0, 9)}), 7, np.zeros(1))
    assert result_to_grid([wide], max_dim=4).to_list() == [[7, 0, 0, 0]]
