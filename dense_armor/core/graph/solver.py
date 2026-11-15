"""Solve an ARC task: explore, refine, return the best program.

A task is ``{"train": [{"input": ..., "output": ...}, ...],
"test": [{"input": ...}, ...]}``. The solver converts the training
pairs to ``(Grid, Grid)`` tuples, runs :func:`explore`, and then
refines every candidate in the beam. The test inputs are not seen
during solving; :func:`to_submission` fills the answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program, Step
from dense_armor.core.graph.refine import refine
from dense_armor.core.graph.search import explore


def _pairs_from_task(task: dict[str, Any]) -> list[tuple[Grid, Grid]]:
    """Convert the ``"train"`` list of a task to ``(Grid, Grid)`` pairs.

    Args:
        task: the task dict.

    Returns:
        the list of pairs.
    """
    pairs: list[tuple[Grid, Grid]] = []
    for ex in task.get("train", []):
        pairs.append((Grid.from_list(ex["input"]), Grid.from_list(ex["output"])))
    return pairs


def _score(program: Program, pairs: list[tuple[Grid, Grid]]) -> tuple[float, int]:
    """Score used to rank candidates: verify first, then shorter is better.

    ARC Prize 2025, page 5, and the arc-dsl writeup, both note that
    shorter programs are less likely to overfit the training pairs.
    When two candidates solve the same number of pairs, prefer the
    shorter one.

    Args:
        program: the candidate.
        pairs: the training pairs.

    Returns:
        a pair ``(verify, -len(steps))``, ordered lexicographically.
    """
    return (program.verify(pairs), -len(program.steps))


def solve(
    task: dict[str, Any],
    primitives: Iterable[Step],
    max_depth: int = 3,
    beam_width: int = 5,
    refine_iterations: int = 10,
    seed: int = 0,
) -> Program:
    """Solve ``task`` with the given primitives.

    Explore the space with :func:`explore`, then refine every candidate
    of the beam with :func:`refine` (one seed per candidate) and return
    the best by ``(verify, -length)``.

    Args:
        task: the ARC task dict with a ``"train"`` list.
        primitives: the callables that can appear as a step.
        max_depth: the search depth.
        beam_width: the search width, and the number of candidates
            refined.
        refine_iterations: the number of refinement mutations per
            candidate.
        seed: the base random seed; candidate ``i`` uses ``seed + i``.

    Returns:
        the best program found. The caller checks ``program.verify``
        against the training pairs.
    """
    pairs = _pairs_from_task(task)
    if not pairs:
        return Program([])
    prims = tuple(primitives)
    candidates = explore(prims, pairs, max_depth=max_depth, beam_width=beam_width)
    if not candidates:
        return Program([])
    best_program = candidates[0]
    best_key = _score(best_program, pairs)
    for i, start in enumerate(candidates):
        refined = refine(
            start,
            prims,
            pairs,
            iterations=refine_iterations,
            seed=seed + i,
        )
        key = _score(refined, pairs)
        if key > best_key:
            best_program, best_key = refined, key
    return best_program
