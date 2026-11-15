"""A grid of ARC colors: immutable, hashable, with crop and pad."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    """A rectangular grid of colors 0..9, immutable and hashable."""

    cells: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("empty grid")
        width = len(self.cells[0])
        for row in self.cells:
            if len(row) != width:
                raise ValueError("ragged grid")
            for value in row:
                if not isinstance(value, int) or not 0 <= value <= 9:
                    raise ValueError(f"color out of range: {value!r}")

    @classmethod
    def from_list(cls, rows: list[list[int]]) -> Grid:
        return cls(tuple(tuple(int(v) for v in row) for row in rows))

    def to_list(self) -> list[list[int]]:
        return [list(row) for row in self.cells]

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0])

    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def at(self, row: int, col: int) -> int:
        return self.cells[row][col]

    def crop(self, box: tuple[int, int, int, int]) -> Grid:
        r0, c0, r1, c1 = box
        return Grid(tuple(row[c0:c1] for row in self.cells[r0:r1]))

    def pad(self, n: int, color: int = 0) -> Grid:
        if n < 0:
            raise ValueError("n must be >= 0")
        w = self.width + 2 * n
        top = (color,) * w
        mid = tuple((color,) * n + row + (color,) * n for row in self.cells)
        bot = (color,) * w
        return Grid((top,) * n + mid + (bot,) * n)

    def colors(self) -> set[int]:
        return {v for row in self.cells for v in row}
