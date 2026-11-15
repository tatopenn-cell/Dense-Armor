"""VSA abduction (Algorithm 1) and grid-size hypotheses (Table 2).

References: Joffe & Eliasmith (2025), arXiv:2511.08747.
    - Table 2 (page 11): grid size hypotheses.
    - Algorithm 1 (page 13): demonstration abduction.
    - Section 3.2.1 (page 12): heuristics and minimum hitting set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa import similarity
from dense_armor.core.graph.vsa_hypotheses import (
    ObjectHypothesis,
    VSAObject,
    object_hypothesis_heuristic,
    perceive,
)
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_operations import (
    OperationParams,
    OperationType,
)


class GridSizeHypothesis(str, Enum):
    """Table 2: how does the output size relate to the input size?"""

    IDENTITY = "identity"
    CONSTANT = "constant"
    FUNCTION = "function"


def grid_size_hypothesis(
    pairs: list[tuple[Grid, Grid]],
) -> tuple[GridSizeHypothesis, tuple[int, int] | None]:
    """Predict the output grid size from the training pairs (Table 2).

    Args:
        pairs: the (input, output) training pairs.

    Returns:
        the hypothesis, and for CONSTANT the constant size.
    """
    if not pairs:
        return GridSizeHypothesis.IDENTITY, None
    same = all(g_in.shape() == g_out.shape() for g_in, g_out in pairs)
    if same:
        return GridSizeHypothesis.IDENTITY, None
    out_sizes = {g_out.shape() for _, g_out in pairs}
    if len(out_sizes) == 1:
        return GridSizeHypothesis.CONSTANT, next(iter(out_sizes))
    return GridSizeHypothesis.FUNCTION, None


@dataclass(frozen=True)
class ConcreteAction:
    """One operation with its concrete parameters."""

    op_type: OperationType
    params: OperationParams


def _candidate_actions(
    in_obj: VSAObject,
    out_obj: VSAObject,
    grid_in: Grid,
    encoder: ObjectEncoder,
) -> list[ConcreteAction]:
    """Derive the concrete actions mapping ``in_obj`` to ``out_obj``.

    The parameters are read from the output object, not searched.

    Args:
        in_obj: the input object.
        out_obj: the output object.
        grid_in: the input grid.
        encoder: the VSA encoder (unused, but kept for API symmetry).

    Returns:
        the list of candidate actions, from the simplest to the most generic.
    """
    actions: list[ConcreteAction] = []
    in_cells = in_obj.cells
    out_cells = out_obj.cells
    in_colour = in_obj.colour
    out_colour = out_obj.colour
    in_centre = _centre_of(in_cells)
    out_centre = _centre_of(out_cells)
    in_shape = _normalize_shape(in_cells)
    out_shape = _normalize_shape(out_cells)

    # IDENTITY: same cells and same colour
    if in_cells == out_cells and in_colour == out_colour:
        actions.append(ConcreteAction(OperationType.IDENTITY, OperationParams()))
        return actions

    # RECOLOUR: same cells, different colour
    if in_cells == out_cells and in_colour != out_colour:
        actions.append(
            ConcreteAction(OperationType.RECOLOUR, OperationParams(colour=out_colour))
        )
        return actions

    # MOVE: same shape, different centre
    if in_shape == out_shape and in_cells != out_cells:
        dr = int(round(out_centre[0] - in_centre[0]))
        dc = int(round(out_centre[1] - in_centre[1]))
        actions.append(
            ConcreteAction(OperationType.MOVE, OperationParams(amount=(dr, dc)))
        )

    # RECENTRE: same shape, different centre
    if in_shape == out_shape and in_cells != out_cells:
        actions.append(
            ConcreteAction(
                OperationType.RECENTRE,
                OperationParams(centre=(out_centre[0], out_centre[1])),
            )
        )

    # FILL: out is the bounding-box fill of in
    r0, c0, r1, c1 = _bbox(in_cells)
    filled = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    if out_cells == filled and in_cells != filled:
        actions.append(ConcreteAction(OperationType.FILL, OperationParams()))

    # HOLLOW: out is the border of in
    hollow = frozenset(
        (r, c) for r, c in in_cells if r == r0 or r == r1 or c == c0 or c == c1
    )
    if out_cells == hollow and in_cells != hollow:
        actions.append(ConcreteAction(OperationType.HOLLOW, OperationParams()))

    # RESHAPE: same colour, new shape centred at in_centre (or out_centre)
    if in_colour == out_colour and in_shape != out_shape:
        rel_shape = frozenset(
            (r - int(round(out_centre[0])), c - int(round(out_centre[1])))
            for r, c in out_cells
        )
        actions.append(
            ConcreteAction(
                OperationType.RESHAPE,
                OperationParams(shape=rel_shape),
            )
        )

    # EXTRACT: out is the in bbox cropped to its own top-left
    if in_cells != out_cells:
        cropped = frozenset((r - r0, c - c0) for r, c in in_cells)
        if cropped == out_cells:
            actions.append(ConcreteAction(OperationType.EXTRACT, OperationParams()))

    # GENERATE: always possible (most generic)
    sh = max(r for r, _ in out_cells) - min(r for r, _ in out_cells) + 1
    sw = max(c for _, c in out_cells) - min(c for _, c in out_cells) + 1
    actions.append(
        ConcreteAction(
            OperationType.GENERATE,
            OperationParams(
                colour=out_colour,
                centre=(out_centre[0], out_centre[1]),
                shape_dims=(sh, sw),
            ),
        )
    )

    return actions


def _bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    r = [r for r, _ in cells]
    c = [c for _, c in cells]
    return min(r), min(c), max(r), max(c)


def _centre_of(cells: frozenset[tuple[int, int]]) -> tuple[float, float]:
    r0, c0, r1, c1 = _bbox(cells)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _normalize_shape(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _input_object_for(
    out_obj: VSAObject,
    objs_in: list[VSAObject],
) -> VSAObject | None:
    """Heuristic 2: pick the input object most similar to ``out_obj``."""
    if not objs_in:
        return None
    sims = [(similarity(out_obj.sp, o.sp), i) for i, o in enumerate(objs_in)]
    sims.sort(reverse=True)
    return objs_in[sims[0][1]]


def _matches_grid(cells: frozenset[tuple[int, int]], colour: int, grid: Grid) -> bool:
    """Check that the cells of an object appear in ``grid`` with ``colour``."""
    h, w = grid.shape()
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid.at(r, c) != colour:
            return False
    return True


def minimum_hitting_set(sets: list[set]) -> set:
    """Smallest set that intersects every set in ``sets``.

    Exact, by branch-and-bound on the smallest uncovered set.

    Args:
        sets: the sets to hit.

    Returns:
        the minimum hitting set.
    """
    sets = [s for s in sets if s]
    if not sets:
        return set()
    best: list[set] = [set().union(*sets)]  # upper bound

    def _rec(chosen: set, remaining: list[set]) -> None:
        if not remaining:
            if len(chosen) < len(best[0]):
                best[0] = set(chosen)
            return
        if len(chosen) >= len(best[0]):
            return
        # pick the smallest uncovered set
        smallest = min(remaining, key=len)
        for candidate in smallest:
            new_chosen = chosen | {candidate}
            new_remaining = [s for s in remaining if candidate not in s]
            _rec(new_chosen, new_remaining)

    _rec(set(), sets)
    return best[0]


@dataclass
class AbductionResult:
    """Output of Algorithm 1: hypothesis, action set, per-object mapping."""

    hypothesis: ObjectHypothesis
    action_set: set[ConcreteAction]
    per_output_actions: list[list[ConcreteAction]]  # one per output object


def abduct(
    task: dict,
    encoder: ObjectEncoder,
    max_hypotheses: int = 3,
) -> AbductionResult | None:
    """Run Algorithm 1: abduction of the simplest object hypothesis and action set.

    Args:
        task: the ARC task dict.
        encoder: the VSA encoder.
        max_hypotheses: the number of top hypotheses to try (ranked by score).

    Returns:
        the abduction result, or None if no hypothesis works.
    """
    pairs = [
        (Grid.from_list(ex["input"]), Grid.from_list(ex["output"]))
        for ex in task["train"]
    ]
    if not pairs:
        return None

    # Rank hypotheses on the first pair (Section 3.2.1)
    hyp_scores = object_hypothesis_heuristic(pairs[0][0], pairs[0][1], encoder)
    top_hyps = [h for h, _ in hyp_scores[:max_hypotheses]]

    for hyp in top_hyps:
        per_output_actions: list[list[ConcreteAction]] = []
        ok = True
        for grid_in, grid_out in pairs:
            objs_in = perceive(grid_in, hyp, encoder)
            objs_out = perceive(grid_out, hyp, encoder)
            if not objs_in or not objs_out:
                ok = False
                break
            for out_obj in objs_out:
                in_obj = _input_object_for(out_obj, objs_in)
                if in_obj is None:
                    ok = False
                    break
                candidates = _candidate_actions(in_obj, out_obj, grid_in, encoder)
                if not candidates:
                    ok = False
                    break
                per_output_actions.append(candidates)
            if not ok:
                break
        if not ok:
            continue
        # Minimum hitting set: one action set covering every output object
        action_set = minimum_hitting_set([set(c) for c in per_output_actions])
        # Sanity: every output must be reachable by at least one action in the set
        return AbductionResult(
            hypothesis=hyp,
            action_set=action_set,
            per_output_actions=per_output_actions,
        )
    return None
