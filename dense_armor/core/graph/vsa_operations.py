"""VSA operations, Table 1 of Joffe & Eliasmith (2025), arXiv:2511.08747.

Each operation transforms one object into one or more objects. Operations
are executed at the cell level (so that the final grid is real), and the
resulting object is re-encoded with ObjectEncoder for the VSA predictors.

Operations (Table 1, page 11):
    IDENTITY, EXTRACT, RECOLOUR(COLOUR), RECENTRE(CENTRE),
    RESHAPE(SHAPE), MOVE(AMOUNT), GRAVITY(DIRECTION), GROW(DIRECTION),
    FILL, HOLLOW, GENERATE(COLOUR, CENTRE, SHAPE).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa_objects import ObjectEncoder


class OperationType(str, Enum):
    """The eleven operations of Table 1."""

    IDENTITY = "identity"
    EXTRACT = "extract"
    RECOLOUR = "recolour"
    RECENTRE = "recentre"
    RESHAPE = "reshape"
    MOVE = "move"
    GRAVITY = "gravity"
    GROW = "grow"
    FILL = "fill"
    HOLLOW = "hollow"
    GENERATE = "generate"


@dataclass(frozen=True)
class OperationParams:
    """Concrete parameters for one operation (predicted by the NN in the paper)."""

    colour: int | None = None
    centre: tuple[float, float] | None = None
    shape: frozenset[tuple[int, int]] | None = None
    amount: tuple[int, int] | None = None
    direction: str | None = None  # "up", "down", "left", "right"
    shape_dims: tuple[int, int] | None = None  # (h, w) for GENERATE


@dataclass(frozen=True)
class OperationResult:
    """One object produced by applying an operation to one input object."""

    cells: frozenset[tuple[int, int]]
    colour: int
    sp: np.ndarray  # the re-encoded VSA


def _bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    r = [r for r, _ in cells]
    c = [c for _, c in cells]
    return min(r), min(c), max(r), max(c)


def _translate(
    cells: frozenset[tuple[int, int]], dr: int, dc: int
) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _centre_of(cells: frozenset[tuple[int, int]]) -> tuple[float, float]:
    r0, c0, r1, c1 = _bbox(cells)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _clamp_into_grid(
    cells: frozenset[tuple[int, int]], h: int, w: int
) -> frozenset[tuple[int, int]]:
    return frozenset((r, c) for r, c in cells if 0 <= r < h and 0 <= c < w)


def _apply_identity(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    h, w = grid.shape()
    kept = _clamp_into_grid(cells, h, w)
    sp = encoder.encode_object(sorted(kept), colour, h, w)
    return [OperationResult(kept, colour, sp)]


def _apply_extract(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    r0, c0, r1, c1 = _bbox(cells)
    cropped = frozenset((r - r0, c - c0) for r, c in cells)
    h, w = (r1 - r0 + 1, c1 - c0 + 1)
    sp = encoder.encode_object(sorted(cropped), colour, h, w)
    return [OperationResult(cropped, colour, sp)]


def _apply_recolour(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    new_colour = params.colour if params.colour is not None else colour
    h, w = grid.shape()
    sp = encoder.encode_object(sorted(cells), new_colour, h, w)
    return [OperationResult(cells, new_colour, sp)]


def _apply_recentre(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.centre is None:
        return _apply_identity(cells, colour, params, grid, encoder)
    target = params.centre
    cur_r, cur_c = _centre_of(cells)
    dr = int(round(target[0] - cur_r))
    dc = int(round(target[1] - cur_c))
    moved = _translate(cells, dr, dc)
    h, w = grid.shape()
    moved = _clamp_into_grid(moved, h, w)
    if not moved:
        return []
    sp = encoder.encode_object(sorted(moved), colour, h, w)
    return [OperationResult(moved, colour, sp)]


def _apply_reshape(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.shape is None:
        return _apply_identity(cells, colour, params, grid, encoder)
    cur_r, cur_c = _centre_of(cells)
    new_cells = frozenset(
        (int(round(cur_r + dr)), int(round(cur_c + dc))) for dr, dc in params.shape
    )
    h, w = grid.shape()
    new_cells = _clamp_into_grid(new_cells, h, w)
    if not new_cells:
        return []
    sp = encoder.encode_object(sorted(new_cells), colour, h, w)
    return [OperationResult(new_cells, colour, sp)]


def _apply_move(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.amount is None:
        return _apply_identity(cells, colour, params, grid, encoder)
    dr, dc = params.amount
    moved = _translate(cells, dr, dc)
    h, w = grid.shape()
    moved = _clamp_into_grid(moved, h, w)
    if not moved:
        return []
    sp = encoder.encode_object(sorted(moved), colour, h, w)
    return [OperationResult(moved, colour, sp)]


def _apply_gravity(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.direction is None:
        return _apply_identity(cells, colour, params, grid, encoder)
    h, w = grid.shape()
    # Occlusion: all cells of the grid that are not this object
    occupied = set()
    for r in range(h):
        for c in range(w):
            if (r, c) not in cells and grid.at(r, c) != 0:
                occupied.add((r, c))
    # Direction deltas
    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    if params.direction not in dirs:
        return _apply_identity(cells, colour, params, grid, encoder)
    dr, dc = dirs[params.direction]
    moved = cells
    for _ in range(max(h, w) + 1):
        candidate = _translate(moved, dr, dc)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in candidate):
            break
        if any((r, c) in occupied for r, c in candidate):
            break
        moved = candidate
    sp = encoder.encode_object(sorted(moved), colour, h, w)
    return [OperationResult(moved, colour, sp)]


def _apply_grow(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.direction is None:
        return _apply_identity(cells, colour, params, grid, encoder)
    h, w = grid.shape()
    occupied = set()
    for r in range(h):
        for c in range(w):
            if (r, c) not in cells and grid.at(r, c) != 0:
                occupied.add((r, c))
    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    if params.direction not in dirs:
        return _apply_identity(cells, colour, params, grid, encoder)
    dr, dc = dirs[params.direction]
    grown = set(cells)
    for _ in range(max(h, w) + 1):
        frontier = set()
        for r, c in grown:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in occupied:
                continue
            frontier.add((nr, nc))
        if not frontier:
            break
        grown |= frontier
    grown_f = frozenset(grown)
    sp = encoder.encode_object(sorted(grown_f), colour, h, w)
    return [OperationResult(grown_f, colour, sp)]


def _apply_fill(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    r0, c0, r1, c1 = _bbox(cells)
    filled = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    h, w = grid.shape()
    sp = encoder.encode_object(sorted(filled), colour, h, w)
    return [OperationResult(filled, colour, sp)]


def _apply_hollow(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    r0, c0, r1, c1 = _bbox(cells)
    hollow = frozenset(
        (r, c) for r, c in cells if r == r0 or r == r1 or c == c0 or c == c1
    )
    h, w = grid.shape()
    sp = encoder.encode_object(sorted(hollow), colour, h, w)
    return [OperationResult(hollow, colour, sp)]


def _apply_generate(
    cells: frozenset[tuple[int, int]],
    colour: int,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    if params.colour is None or params.centre is None or params.shape_dims is None:
        return []
    new_colour = params.colour
    cr, cc = params.centre
    h, w = grid.shape()
    sh, sw = params.shape_dims
    new_cells = frozenset(
        (int(round(cr - sh / 2 + r)), int(round(cc - sw / 2 + c)))
        for r in range(sh)
        for c in range(sw)
    )
    new_cells = _clamp_into_grid(new_cells, h, w)
    if not new_cells:
        return []
    sp = encoder.encode_object(sorted(new_cells), new_colour, h, w)
    return [OperationResult(new_cells, new_colour, sp)]


def execute_action(
    cells: frozenset[tuple[int, int]],
    colour: int,
    op_type: OperationType,
    params: OperationParams,
    grid: Grid,
    encoder: ObjectEncoder,
) -> list[OperationResult]:
    """Execute one operation on one input object.

    Args:
        cells: the cells of the input object.
        colour: the colour of the input object.
        op_type: which of the eleven operations.
        params: the concrete parameters (from the NN predictor).
        grid: the grid the object lives in.
        encoder: the VSA encoder.

    Returns:
        the list of output objects (some operations may produce none).
    """
    if not cells and op_type != OperationType.GENERATE:
        return []
    dispatch = {
        OperationType.IDENTITY: _apply_identity,
        OperationType.EXTRACT: _apply_extract,
        OperationType.RECOLOUR: _apply_recolour,
        OperationType.RECENTRE: _apply_recentre,
        OperationType.RESHAPE: _apply_reshape,
        OperationType.MOVE: _apply_move,
        OperationType.GRAVITY: _apply_gravity,
        OperationType.GROW: _apply_grow,
        OperationType.FILL: _apply_fill,
        OperationType.HOLLOW: _apply_hollow,
        OperationType.GENERATE: _apply_generate,
    }
    fn = dispatch.get(op_type)
    if fn is None:
        return []
    return fn(cells, colour, params, grid, encoder)


def result_to_grid(
    results: list[OperationResult],
    background_colour: int = 0,
    max_dim: int = 30,
) -> Grid:
    """Render the list of OperationResult into a grid.

    Args:
        results: the objects produced by the operations.
        background_colour: fill colour for empty cells.
        max_dim: maximum dimension of the output grid.

    Returns:
        a Grid.
    """
    if not results:
        return Grid.from_list([[background_colour]])
    all_cells: set[tuple[int, int]] = set()
    for r in results:
        all_cells |= r.cells
    r_min = min(r for r, _ in all_cells)
    r_max = max(r for r, _ in all_cells)
    c_min = min(c for _, c in all_cells)
    c_max = max(c for _, c in all_cells)
    h = min(max_dim, r_max - r_min + 1)
    w = min(max_dim, c_max - c_min + 1)
    grid = [[background_colour] * w for _ in range(h)]
    for r in results:
        for rr, cc in r.cells:
            nr, nc = rr - r_min, cc - c_min
            if 0 <= nr < h and 0 <= nc < w:
                grid[nr][nc] = r.colour
    return Grid.from_list(grid)
