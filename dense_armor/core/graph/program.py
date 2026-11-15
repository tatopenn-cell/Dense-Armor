"""Composition of Grid -> Grid primitives into a program.

A ``Program`` is a sequence of pure functions
``Grid -> Grid``. It can be applied to a grid and verified against
a list of ``(input, output)`` training pairs. The verification score
is the fraction of pairs the program solves correctly; it is the
feedback signal that a refinement loop (see ARC Prize 2025, page 5)
uses to decide whether to keep or replace a candidate program.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from dense_armor.core.graph.grid import Grid

Step = Callable[[Grid], Grid]


class Program:
    """A sequence of Grid -> Grid steps.

    Args:
        steps: the callables to apply in order.

    Example:
        >>> from dense_armor.core.graph.dsl import rot90, hmirror
        >>> from dense_armor.core.graph.grid import Grid
        >>> p = Program([rot90, hmirror])
        >>> p.apply(Grid.from_list([[1, 0], [0, 0]])).to_list()
        [[1, 0], [0, 0]]
    """

    def __init__(self, steps: Iterable[Step]) -> None:
        self.steps: tuple[Step, ...] = tuple(steps)

    def apply(self, g: Grid) -> Grid:
        """Apply every step in order.

        Args:
            g: the input grid.

        Returns:
            the grid after the last step.
        """
        out = g
        for step in self.steps:
            out = step(out)
        return out

    def verify(self, pairs: Sequence[tuple[Grid, Grid]]) -> float:
        """Fraction of training pairs the program solves correctly.

        Args:
            pairs: the ``(input, output)`` pairs to check.

        Returns:
            a number in ``[0.0, 1.0]``; ``0.0`` when ``pairs`` is empty.
        """
        if not pairs:
            return 0.0
        ok = 0
        for x, y in pairs:
            if self.apply(x) == y:
                ok += 1
        return ok / len(pairs)

    def __repr__(self) -> str:
        """Human-readable form listing the step names."""
        names = [getattr(s, "__name__", repr(s)) for s in self.steps]
        return f"Program({', '.join(names)})"

    def __eq__(self, other: object) -> bool:
        """Two programs are equal when their steps are equal, in order."""
        if not isinstance(other, Program):
            return NotImplemented
        return self.steps == other.steps

    def __hash__(self) -> int:
        """Hash based on the step tuple."""
        return hash(self.steps)


__all__ = ["Program"]
