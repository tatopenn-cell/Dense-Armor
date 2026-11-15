"""Breadth-first exploration of the space of Grid -> Grid programs.

Given a small set of primitives and a list of training pairs, the
exploration generates programs by composing primitives up to
``max_depth`` and keeps the ``beam_width`` candidates with the highest
verification score. The score is :meth:`Program.verify`, the fraction
of training pairs the program solves correctly (ARC Prize 2025,
page 1: the refinement loop uses a verifiable feedback signal).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program, Step


def explore(
    primitives: Iterable[Step],
    pairs: Sequence[tuple[Grid, Grid]],
    max_depth: int = 3,
    beam_width: int = 5,
) -> list[Program]:
    """Breadth-first search over programs, keeping a beam of the best.

    Args:
        primitives: the callables that can appear as a step.
        pairs: the ``(input, output)`` pairs used to score candidates.
        max_depth: the maximum number of steps in a candidate.
        beam_width: the number of candidates kept at each depth.

    Returns:
        a list of ``Program`` ordered by descending verification score.

    Raises:
        ValueError: if ``max_depth`` is negative or ``beam_width`` is not
            positive.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    prims = tuple(primitives)
    if not prims:
        return [Program([])]
    scored: list[tuple[float, Program]] = [(Program([]).verify(pairs), Program([]))]
    for _ in range(max_depth):
        candidates: list[tuple[float, Program]] = []
        for _, p in scored:
            for prim in prims:
                candidate = Program(p.steps + (prim,))
                candidates.append((candidate.verify(pairs), candidate))
        candidates.sort(key=lambda t: -t[0])
        scored = candidates[:beam_width]
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored]
