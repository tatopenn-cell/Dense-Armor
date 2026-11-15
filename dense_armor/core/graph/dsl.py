"""Pure Grid -> Grid transformations.

The primitives follow the arc-dsl writeup (Chollet, "The ARC-DSL",
Section 2.3): rotation, mirroring, fill, recolor, shift, scaling.
Unlike the writeup, which uses plain data structures, these operate
on :class:`~dense_armor.core.graph.grid.Grid` and preserve its
invariants: the result is immutable, hashable, and has colors 0..9.

Every primitive is a pure function. Composition lives in
:mod:`dense_armor.core.graph.program`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import ArcObject


def rot90(g: Grid) -> Grid:
    """Rotate the grid 90 degrees counter-clockwise.

    Args:
        g: the input grid.

    Returns:
        a grid of shape ``(g.width(), g.height())``.
    """
    h, w = g.shape()
    return Grid.from_list([[g.at(j, w - 1 - i) for j in range(h)] for i in range(w)])


def rot180(g: Grid) -> Grid:
    """Rotate the grid 180 degrees.

    Args:
        g: the input grid.

    Returns:
        the grid rotated by two quarter-turns.
    """
    h, w = g.shape()
    return Grid.from_list(
        [[g.at(h - 1 - i, w - 1 - j) for j in range(w)] for i in range(h)]
    )


def rot270(g: Grid) -> Grid:
    """Rotate the grid 90 degrees clockwise.

    Args:
        g: the input grid.

    Returns:
        a grid of shape ``(g.width(), g.height())``.
    """
    h, w = g.shape()
    return Grid.from_list([[g.at(h - 1 - j, i) for j in range(h)] for i in range(w)])


def hmirror(g: Grid) -> Grid:
    """Mirror the grid horizontally (reverse each row).

    Args:
        g: the input grid.

    Returns:
        a grid of the same shape.
    """
    h, w = g.shape()
    return Grid.from_list([[g.at(i, w - 1 - j) for j in range(w)] for i in range(h)])


def vmirror(g: Grid) -> Grid:
    """Mirror the grid vertically (reverse the rows).

    Args:
        g: the input grid.

    Returns:
        a grid of the same shape.
    """
    h, w = g.shape()
    return Grid.from_list([[g.at(h - 1 - i, j) for j in range(w)] for i in range(h)])


def dmirror(g: Grid) -> Grid:
    """Transpose the grid (mirror along the main diagonal).

    Args:
        g: the input grid.

    Returns:
        a grid of shape ``(g.width(), g.height())``.
    """
    h, w = g.shape()
    return Grid.from_list([[g.at(j, i) for j in range(h)] for i in range(w)])


def fill(g: Grid, color: int, mask: set[tuple[int, int]] | None = None) -> Grid:
    """Set cells of the grid to a given color.

    Args:
        g: the input grid.
        color: the color to write (0..9).
        mask: the set of ``(row, col)`` cells to fill; ``None`` fills all.

    Returns:
        a grid of the same shape.
    """
    h, w = g.shape()
    out = [[g.at(i, j) for j in range(w)] for i in range(h)]
    if mask is None:
        return Grid.from_list([[color] * w for _ in range(h)])
    for r, c in mask:
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = color
    return Grid.from_list(out)


def recolor(g: Grid, from_color: int, to_color: int) -> Grid:
    """Replace one color with another.

    Args:
        g: the input grid.
        from_color: the color to replace (0..9).
        to_color: the replacement color (0..9).

    Returns:
        a grid of the same shape.
    """
    h, w = g.shape()
    return Grid.from_list(
        [
            [to_color if g.at(i, j) == from_color else g.at(i, j) for j in range(w)]
            for i in range(h)
        ]
    )


def shift(g: Grid, dr: int, dc: int, fill_color: int = 0) -> Grid:
    """Translate the grid by ``(dr, dc)``, padding with ``fill_color``.

    Args:
        g: the input grid.
        dr: the row offset (positive = down).
        dc: the column offset (positive = right).
        fill_color: the color of the cells that have no source (0..9).

    Returns:
        a grid of the same shape.
    """
    h, w = g.shape()
    out = [[fill_color] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            si, sj = i - dr, j - dc
            if 0 <= si < h and 0 <= sj < w:
                out[i][j] = g.at(si, sj)
    return Grid.from_list(out)


def upscale(g: Grid, k: int) -> Grid:
    """Scale the grid up by an integer factor.

    Args:
        g: the input grid.
        k: the integer factor (>= 1).

    Returns:
        a grid of shape ``(g.height() * k, g.width() * k)``.

    Raises:
        ValueError: if ``k`` is smaller than 1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    h, w = g.shape()
    out: list[list[int]] = []
    for i in range(h):
        row: list[int] = []
        for j in range(w):
            row.extend([g.at(i, j)] * k)
        for _ in range(k):
            out.append(list(row))
    return Grid.from_list(out)


def downscale(g: Grid, k: int) -> Grid:
    """Scale the grid down by an integer factor, keeping the top-left of each block.

    Args:
        g: the input grid.
        k: the integer factor (>= 1).

    Returns:
        a grid of shape ``(g.height() // k, g.width() // k)``.

    Raises:
        ValueError: if ``k`` is smaller than 1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    h, w = g.shape()
    return Grid.from_list(
        [[g.at(i * k, j * k) for j in range(w // k)] for i in range(h // k)]
    )


__all__ = [
    "rot90",
    "rot180",
    "rot270",
    "hmirror",
    "vmirror",
    "dmirror",
    "fill",
    "recolor",
    "shift",
    "upscale",
    "downscale",
    "fill_objects",
    "recolor_objects",
    "move_objects",
    "delete_objects",
    "crop_to_object",
    "apply_per_object",
    "default_primitives",
]

# --- primitive che operano sugli oggetti della griglia ---

Predicate: TypeAlias = Callable[["ArcObject"], bool]


def _is_foreground(obj: ArcObject) -> bool:
    """Default predicate: true for objects that are not background color 0.

    Args:
        obj: the object to check.

    Returns:
        ``True`` if the object's color is not 0.
    """
    return obj.color is not None and obj.color != 0


def _named(fn: Callable[..., Grid], name: str) -> Callable[..., Grid]:
    """Wrap a closure so that ``Program.__repr__`` shows a step name.

    Args:
        fn: the callable to wrap.
        name: the name to expose as ``__name__``.

    Returns:
        the same callable, with ``__name__`` set.
    """
    fn.__name__ = name
    return fn


def fill_objects(
    g: Grid,
    color: int,
    predicate: Predicate | None = None,
) -> Grid:
    """Fill every object that matches ``predicate`` with ``color``.

    Args:
        g: the input grid.
        color: the fill color (0..9).
        predicate: the condition on the object; ``None`` fills every
            foreground object (color != 0).

    Returns:
        a grid of the same shape.
    """
    from dense_armor.core.graph.objects import extract_objects

    pred = predicate or _is_foreground
    h, w = g.shape()
    out = [[g.at(i, j) for j in range(w)] for i in range(h)]
    for obj in extract_objects(g):
        if pred(obj):
            for r, c in obj.cells:
                out[r][c] = color
    return Grid.from_list(out)


def recolor_objects(
    g: Grid,
    mapping: dict[int, int],
    predicate: Predicate | None = None,
) -> Grid:
    """Recolor every object that matches ``predicate`` with ``mapping``.

    Args:
        g: the input grid.
        mapping: ``{from_color: to_color}``.
        predicate: the condition on the object; ``None`` uses every
            foreground object.

    Returns:
        a grid of the same shape.
    """
    from dense_armor.core.graph.objects import extract_objects

    pred = predicate or _is_foreground
    h, w = g.shape()
    out = [[g.at(i, j) for j in range(w)] for i in range(h)]
    for obj in extract_objects(g):
        if pred(obj) and obj.color in mapping:
            new = mapping[obj.color]
            for r, c in obj.cells:
                out[r][c] = new
    return Grid.from_list(out)


def move_objects(
    g: Grid,
    dr: int,
    dc: int,
    predicate: Predicate | None = None,
) -> Grid:
    """Translate every object that matches ``predicate`` by ``(dr, dc)``.

    Objects that would fall outside the grid are dropped; the cells
    they came from are left as background 0.

    Args:
        g: the input grid.
        dr: the row offset (positive = down).
        dc: the column offset (positive = right).
        predicate: the condition on the object; ``None`` uses every
            foreground object.

    Returns:
        a grid of the same shape.
    """
    from dense_armor.core.graph.objects import extract_objects

    pred = predicate or _is_foreground
    h, w = g.shape()
    out = [[0] * w for _ in range(h)]
    for obj in extract_objects(g):
        if pred(obj):
            continue
        for r, c in obj.cells:
            out[r][c] = g.at(r, c)
    for obj in extract_objects(g):
        if not pred(obj):
            continue
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g.at(r, c)
    return Grid.from_list(out)


def delete_objects(
    g: Grid,
    predicate: Predicate | None = None,
) -> Grid:
    """Delete every object that matches ``predicate`` (background 0).

    Args:
        g: the input grid.
        predicate: the condition on the object; ``None`` deletes every
            foreground object.

    Returns:
        a grid of the same shape.
    """
    from dense_armor.core.graph.objects import extract_objects

    pred = predicate or _is_foreground
    h, w = g.shape()
    out = [[g.at(i, j) for j in range(w)] for i in range(h)]
    for obj in extract_objects(g):
        if pred(obj):
            for r, c in obj.cells:
                out[r][c] = 0
    return Grid.from_list(out)


def crop_to_object(
    g: Grid,
    predicate: Predicate | None = None,
) -> Grid:
    """Crop the grid to the bounding box of the first object matching ``predicate``.

    The first match is the object with the smallest ``(min_row, min_col)``.

    Args:
        g: the input grid.
        predicate: the condition on the object; ``None`` uses the
            first foreground object.

    Returns:
        a grid of shape ``(r1 - r0, c1 - c0)`` for the matched bbox.

    Raises:
        ValueError: if no object matches.
    """
    from dense_armor.core.graph.objects import extract_objects

    pred = predicate or _is_foreground
    for obj in extract_objects(g):
        if pred(obj):
            return g.crop(obj.bbox)
    raise ValueError("no object matches the predicate")


def apply_per_object(
    g: Grid,
    fn: Callable[[ArcObject], list[tuple[int, int, int]]],
) -> Grid:
    """Apply ``fn`` to every object and rebuild the grid from the returned cells.

    ``fn`` receives an ``ArcObject`` and returns a list of
    ``(row, col, color)`` cells. The grid starts from background 0 and
    is filled with the cells returned for every object (last write
    wins).

    Args:
        g: the input grid.
        fn: the per-object function.

    Returns:
        a grid of the same shape.
    """
    from dense_armor.core.graph.objects import extract_objects

    h, w = g.shape()
    out = [[0] * w for _ in range(h)]
    for obj in extract_objects(g):
        for r, c, color in fn(obj):
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = color
    return Grid.from_list(out)


def default_primitives() -> list[Callable[[Grid], Grid]]:
    """A small, general menu of primitives for the solver.

    Returns:
        a list of ``Grid -> Grid`` functions: the original rotations and
        mirrors, plus the object primitives introduced for the ARC task.
    """
    return [
        rot90,
        rot180,
        rot270,
        hmirror,
        vmirror,
        dmirror,
        _named(lambda g: fill(g, 0), "fill0"),
        _named(lambda g: fill(g, 1), "fill1"),
        _named(lambda g: recolor(g, 0, 1), "recolor01"),
        _named(lambda g: shift(g, 1, 0), "shift_down"),
        _named(lambda g: shift(g, 0, 1), "shift_right"),
        _named(lambda g: upscale(g, 2), "upscale2"),
        _named(lambda g: downscale(g, 2), "downscale2"),
        _named(lambda g: fill_objects(g, 0), "fill_objects0"),
        _named(
            lambda g: recolor_objects(
                g, {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}
            ),
            "swap_pairs",
        ),
        _named(lambda g: delete_objects(g, lambda o: o.area == 1), "delete_size1"),
        _named(lambda g: delete_objects(g, lambda o: o.area < 4), "delete_lt4"),
        _named(
            lambda g: crop_to_object(
                g,
                lambda o: (
                    o.color != 0
                    and o.area
                    == max((oo.area for oo in _foreground_objects(g)), default=0)
                ),
            ),
            "crop_largest",
        ),
        _named(lambda g: move_objects(g, 1, 0), "move_down_1"),
        _named(lambda g: move_objects(g, 0, 1), "move_right_1"),
        _named(
            lambda g: fill_objects(g, 0, lambda o: o.touches_border(g.shape())),
            "clear_border_objs",
        ),
    ]


def _foreground_objects(g: Grid):
    """Yield the foreground objects of ``g``.

    Args:
        g: the input grid.

    Yields:
        each ``ArcObject`` whose color is not 0.
    """
    from dense_armor.core.graph.objects import extract_objects

    for o in extract_objects(g):
        if o.color is not None and o.color != 0:
            yield o
