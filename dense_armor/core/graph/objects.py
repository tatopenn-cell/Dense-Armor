"""Objects in a grid: connected cells of the same color, with properties."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .grid import Grid


@dataclass(frozen=True)
class ArcObject:
    """A set of cells of a grid. ``bbox`` is ``(r0, c0, r1, c1)``,
    ``r1`` and ``c1`` exclusive."""

    cells: frozenset[tuple[int, int]]
    color: int | None
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return len(self.cells)

    def bbox_area(self) -> int:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0) * (c1 - c0)

    def touches_border(self, shape: tuple[int, int]) -> bool:
        h, w = shape
        for r, c in self.cells:
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                return True
        return False

    def is_rectangular(self) -> bool:
        return self.area == self.bbox_area()

    def holes(self, grid: Grid) -> list[frozenset[tuple[int, int]]]:
        """Connected components of non-object cells inside the bbox
        that do not touch the bbox border."""
        r0, c0, r1, c1 = self.bbox
        inside = {
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if (r, c) not in self.cells
        }
        on_bbox_border = {
            (r, c)
            for (r, c) in inside
            if r == r0 or r == r1 - 1 or c == c0 or c == c1 - 1
        }
        holes: list[frozenset[tuple[int, int]]] = []
        visited: set[tuple[int, int]] = set()
        for start in sorted(inside):
            if start in visited:
                continue
            comp: set[tuple[int, int]] = set()
            q = deque([start])
            visited.add(start)
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in inside and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            if not comp & on_bbox_border:
                holes.append(frozenset(comp))
        return holes


def _bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Compute the bounding box of a list of cells.

    Args:
        cells: the cells of the object.

    Returns:
        (min_row, min_col, max_row, max_col) of the object.
    """
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs) + 1, max(cs) + 1)


def extract_objects(
    grid: Grid,
    connectivity: int = 4,
    by_color: bool = True,
) -> list[ArcObject]:
    """Extract connected components.

    ``connectivity`` is 4 or 8. ``by_color`` means that two cells
    belong to the same object only if they have the same color.
    With ``by_color=False`` every cell, background included, joins its
    neighbours: the whole grid is one object with ``color=None``.

    Args:
        grid: the grid to analyse.
        connectivity: 4 or 8 (default 4).
        by_color: True to split cells by color (default True).

    Returns:
        the list of ArcObject instances.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    deltas: tuple[tuple[int, int], ...]
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    seen: set[tuple[int, int]] = set()
    objects: list[ArcObject] = []
    h, w = grid.shape()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            color = grid.at(r, c)
            comp: set[tuple[int, int]] = set()
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if by_color and grid.at(nr, nc) != color:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            cells = frozenset(comp)
            objects.append(
                ArcObject(
                    cells=cells,
                    color=color if by_color else None,
                    bbox=_bbox(cells),
                )
            )
    return objects
