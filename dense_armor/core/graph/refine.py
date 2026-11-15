"""Local refinement of a Program by random mutations.

The refinement loop (ARC Prize 2025, page 5) explores candidates,
verifies them, and refines the best one. This module implements the
last step without an LLM: a hill climb where each iteration tries a
random neighbor (add, remove, or replace a step) and keeps the
candidate if it strictly improves the verification score.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program, Step


def _mutate(
    program: Program, primitives: tuple[Step, ...], rng: random.Random
) -> Program:
    """Return a random neighbor of ``program``.

    The neighbor is obtained by adding, removing or replacing one
    step. When ``program`` is empty only addition is possible.

    Args:
        program: the program to mutate.
        primitives: the callables that can appear as a step.
        rng: the random source, for reproducibility.

    Returns:
        a new ``Program``.
    """
    steps = list(program.steps)
    if not steps:
        return Program([rng.choice(primitives)])
    action = rng.choice(("add", "remove", "replace"))
    if action == "add":
        pos = rng.randint(0, len(steps))
        steps.insert(pos, rng.choice(primitives))
    elif action == "remove":
        steps.pop(rng.randrange(len(steps)))
    else:
        steps[rng.randrange(len(steps))] = rng.choice(primitives)
    return Program(steps)


def refine(
    program: Program,
    primitives: Iterable[Step],
    pairs: Sequence[tuple[Grid, Grid]],
    iterations: int = 10,
    seed: int = 0,
) -> Program:
    """Hill climb from ``program`` by local mutations, keeping the best.

    At each iteration, a random neighbor is tried. If its verification
    score is strictly greater than the current best, it becomes the
    best. The best program seen is returned.

    Args:
        program: the starting program.
        primitives: the callables that can appear as a step.
        pairs: the ``(input, output)`` pairs used to score candidates.
        iterations: the number of mutations to try.
        seed: the random seed, for reproducibility.

    Returns:
        the best program found; its verification score is never less
        than the starting program's.

    Raises:
        ValueError: if ``iterations`` is negative.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    prims = tuple(primitives)
    if not prims:
        return program
    rng = random.Random(seed)
    best = program
    best_score = best.verify(pairs)
    for _ in range(iterations):
        candidate = _mutate(best, prims, rng)
        score = candidate.verify(pairs)
        if score > best_score:
            best, best_score = candidate, score
    return best
