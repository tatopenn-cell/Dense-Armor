"""A TypedGraph of the cells and the objects of a grid."""

from __future__ import annotations

from dense_armor.core.graph.ontology import Constraint, TypedGraph

from .grid import Grid
from .objects import extract_objects

SCHEMA = {
    "cell": {"row": int, "col": int, "color": int},
    "object": {
        "color": int,
        "area": int,
        "bbox": tuple,
        "cells": tuple,
        "touches_border": bool,
        "has_hole": bool,
    },
}


def _cell_name(r: int, c: int) -> str:
    """Return the node name for a cell.

    Args:
        r: row index.
        c: column index.

    Returns:
        the node name ``cell:r,c``.
    """
    return f"cell:{r},{c}"


def _obj_name(i: int) -> str:
    """Return the node name for an object.

    Args:
        i: object index.

    Returns:
        the node name ``obj:i``.
    """
    return f"obj:{i}"


def build_grid_graph(
    grid: Grid,
    connectivity: int = 4,
    by_color: bool = True,
) -> TypedGraph:
    """Build a TypedGraph with one node per cell and one per object,
    plus edges ``cell -belongs_to-> object`` and
    ``object -adjacent_to-> object``.

    Every ``belongs_to`` edge carries a constraint that checks the
    cell is really in the object's cells (and the colors agree, when
    the object has a single color). Every ``adjacent_to`` edge
    carries a constraint that checks the two objects really touch.

    Args:
        grid: the grid to analyse.
        connectivity: 4 or 8 (default 4).
        by_color: True to split cells by color (default True).

    Returns:
        a TypedGraph with cell and object nodes and their edges.
    """
    g = TypedGraph(SCHEMA)
    objects = extract_objects(grid, connectivity=connectivity, by_color=by_color)
    shape = grid.shape()

    for i, obj in enumerate(objects):
        cells = tuple(sorted(obj.cells))
        g.add_node(
            _obj_name(i),
            "object",
            color=obj.color if obj.color is not None else -1,
            area=obj.area,
            bbox=obj.bbox,
            cells=cells,
            touches_border=obj.touches_border(shape),
            has_hole=bool(obj.holes(grid)),
        )

    cell_to_obj: dict[tuple[int, int], int] = {}
    for i, obj in enumerate(objects):
        for r, c in obj.cells:
            cell_to_obj[(r, c)] = i
    for r in range(shape[0]):
        for c in range(shape[1]):
            g.add_node(
                _cell_name(r, c),
                "cell",
                row=r,
                col=c,
                color=grid.at(r, c),
            )
    constraint_in = Constraint(
        "cell_in_object_and_color",
        lambda cell, obj: (
            (cell["row"], cell["col"]) in obj["cells"]
            and (obj["color"] == -1 or cell["color"] == obj["color"])
        ),
    )
    for (r, c), i in cell_to_obj.items():
        g.add_edge(_cell_name(r, c), _obj_name(i), "belongs_to", constraint_in)

    constraint_adj = Constraint(
        "objects_are_4_adjacent",
        lambda a, b: _share_cell_boundary(a["cells"], b["cells"]),
    )
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _share_cell_boundary(
                tuple(sorted(objects[i].cells)),
                tuple(sorted(objects[j].cells)),
            ):
                g.add_edge(_obj_name(i), _obj_name(j), "adjacent_to", constraint_adj)
                g.add_edge(_obj_name(j), _obj_name(i), "adjacent_to", constraint_adj)
    return g


def _share_cell_boundary(a: tuple, b: tuple) -> bool:
    """Check whether two cell sets are 4-adjacent.

    Args:
        a: the cells of the first object.
        b: the cells of the second object.

    Returns:
        True if the two cell sets are 4-adjacent.
    """
    set_b = set(b)
    for r, c in a:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in set_b:
                return True
    return False
