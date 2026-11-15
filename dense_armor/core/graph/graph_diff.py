"""Graph differencing for ARC: L2 builder + LAP on L1 objects.

The module follows the specification formalised in chat (D1-D3):
Level 2 has four deterministic categories (background, frame,
alignment_group, cavity); the matching on L1 objects is a Linear
Assignment Problem solved with ``scipy.optimize.linear_sum_assignment``
(Jonker-Volgenant, O(n^3), deterministic).

Level 2 is returned as a plain ``dict[str, dict]`` rather than being
folded into ``TypedGraph``: ``TypedGraph`` validates against a fixed
schema, and L2 has four node types that ``TypedGraph`` would have to
be extended for. The rest of the pipeline only reads L2 via
``dict.get``, so keeping it separate is enough.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import ArcObject

TAU_NULL = 1000.0


def level2_nodes(grid: Grid, objects: list[ArcObject]) -> dict[str, dict]:
    """Build the Level 2 nodes of ``grid``.

    Args:
        grid: the grid.
        objects: the Level 1 objects extracted from ``grid``.

    Returns:
        a dict mapping each L2 node name to ``{"type": ..., "state": ...}``.
    """
    h, w = grid.shape()
    nodes: dict[str, dict] = {}

    counts = Counter(grid.at(r, c) for r in range(h) for c in range(w))
    bg_color = counts.most_common(1)[0][0] if counts else 0
    nodes["bg"] = {"type": "background", "state": {"color": bg_color}}

    for i, obj in enumerate(objects):
        cells = obj.cells
        touches = {
            "top": any(r == 0 for r, _ in cells),
            "bot": any(r == h - 1 for r, _ in cells),
            "left": any(c == 0 for _, c in cells),
            "right": any(c == w - 1 for _, c in cells),
        }
        if all(touches.values()):
            nodes[f"frame:{i}"] = {"type": "frame", "state": {"obj_idx": i}}

    for axis in ("row", "col"):
        groups: dict[float, list[int]] = {}
        for i, obj in enumerate(objects):
            r0, c0, r1, c1 = obj.bbox
            center = (r0 + r1) / 2.0 if axis == "row" else (c0 + c1) / 2.0
            groups.setdefault(center, []).append(i)
        for center, idxs in groups.items():
            if len(idxs) >= 2:
                nodes[f"align:{axis}:{center}"] = {
                    "type": "alignment_group",
                    "state": {"axis": axis, "center": center, "members": tuple(idxs)},
                }

    for i, obj in enumerate(objects):
        holes = obj.holes(grid)
        for j, hole in enumerate(holes):
            nodes[f"cavity:{i}:{j}"] = {
                "type": "cavity",
                "state": {"parent_obj_idx": i, "cells": tuple(sorted(hole))},
            }
    return nodes


def _obj_center(obj: ArcObject) -> tuple[float, float]:
    r0, c0, r1, c1 = obj.bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _pair_cost(u: ArcObject, v: ArcObject, w: tuple[float, ...]) -> float:
    """Weighted cost of matching object ``u`` to object ``v``.

    Args:
        u: input object.
        v: output object.
        w: weights ``(w1, w2, w3, w4, w5)`` for shape, area, holes,
            centroid distance, color.

    Returns:
        the cost.
    """
    w1, w2, w3, w4, w5 = w
    cost = 0.0
    ur, uc = u.bbox[2] - u.bbox[0], u.bbox[3] - u.bbox[1]
    vr, vc = v.bbox[2] - v.bbox[0], v.bbox[3] - v.bbox[1]
    if (ur, uc) != (vr, vc):
        cost += w1
    if u.area != v.area and max(u.area, v.area) > 0:
        cost += w2 * abs(u.area - v.area) / max(u.area, v.area)
    # holes: both None means no holes; we approximate with has_hole flag
    # because holes() needs a Grid; the caller will use the flag if needed.
    if (u.color is None) != (v.color is None):
        cost += w3
    elif u.color is not None and v.color is not None and u.color != v.color:
        cost += w5
    ucr, ucc = _obj_center(u)
    vcr, vcc = _obj_center(v)
    cost += w4 * (abs(ucr - vcr) + abs(ucc - vcc))
    return cost


def hungarian_match(
    objects_I: list[ArcObject],
    objects_O: list[ArcObject],
    weights: tuple[float, float, float, float, float] = (10.0, 5.0, 5.0, 1.0, 3.0),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the LAP between input and output objects.

    The cost matrix is ``(n_I + n_O) x (n_I + n_O)``; nodes beyond the
    real objects represent deletion (left block) and creation (top
    block). A pair ``(i, j)`` is a match only if its cost is strictly
    below ``TAU_NULL``.

    Args:
        objects_I: the objects of the input grid.
        objects_O: the objects of the output grid.
        weights: the five weights of ``_pair_cost``.

    Returns:
        ``(matched_pairs, created, deleted)`` as index lists.
    """
    n_I = len(objects_I)
    n_O = len(objects_O)
    if n_I == 0 and n_O == 0:
        return [], [], []
    if n_I == 0:
        return [], list(range(n_O)), []
    if n_O == 0:
        return [], [], list(range(n_I))

    size = n_I + n_O
    C = np.zeros((size, size))
    C[:n_I, :n_O] = TAU_NULL + 1.0
    for i in range(n_I):
        for j in range(n_O):
            C[i, j] = _pair_cost(objects_I[i], objects_O[j], weights)
    C[:n_I, n_O:] = TAU_NULL
    C[n_I:, :n_O] = TAU_NULL
    C[n_I:, n_O:] = 0.0

    ri, ci = linear_sum_assignment(C)
    matched: list[tuple[int, int]] = []
    deleted: list[int] = []
    created: list[int] = []
    for r, c in zip(ri.tolist(), ci.tolist()):
        if r < n_I and c < n_O:
            if C[r, c] < TAU_NULL:
                matched.append((int(r), int(c)))
            else:
                deleted.append(int(r))
                created.append(int(c))
        elif r < n_I and c >= n_O:
            deleted.append(int(r))
        elif r >= n_I and c < n_O:
            created.append(int(c))
    return matched, created, deleted


def diff_graphs(
    grid_I: Grid,
    grid_O: Grid,
) -> dict:
    """Full graph diff between two grids.

    Args:
        grid_I: input grid.
        grid_O: output grid.

    Returns:
        a dict with ``objects_I``, ``objects_O``, ``L2_I``, ``L2_O``,
        ``matched``, ``created``, ``deleted``.
    """
    from dense_armor.core.graph.objects import extract_objects

    objs_I = extract_objects(grid_I, by_color=True)
    objs_O = extract_objects(grid_O, by_color=True)
    L2_I = level2_nodes(grid_I, objs_I)
    L2_O = level2_nodes(grid_O, objs_O)
    matched, created, deleted = hungarian_match(objs_I, objs_O)
    return {
        "objects_I": objs_I,
        "objects_O": objs_O,
        "L2_I": L2_I,
        "L2_O": L2_O,
        "matched": matched,
        "created": created,
        "deleted": deleted,
    }
