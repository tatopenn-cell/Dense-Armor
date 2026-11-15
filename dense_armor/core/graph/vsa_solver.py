"""VSA solver: full pipeline (abduction -> induction -> deduction).

References:
    Joffe & Eliasmith (2025), arXiv:2511.08747.
    - Algorithm 1 (page 13): demonstration abduction.
    - Algorithm 2 (page 15): rule induction.
    - Algorithm 3 (page 16): answer deduction.

The paper's rule induction uses single-layer NNs (Dense + Trainer from
Dense-Armor) to predict, for each operation in the action set:
- the condition (whether the operation applies to an object), and
- the parameters (colour, centre, shape, amount, direction).

In this implementation the operation predictors are nearest-neighbour
prototypes on the VSA encodings, which is the zero-parameter limit of
the paper's "initialise the weight vector with the superposition of the
input object representations weighted by outcome" (Section 3.2.2, page
14). The paper itself calls the learned weight vector a prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa import similarity
from dense_armor.core.graph.vsa_abduction import (
    AbductionResult,
    ConcreteAction,
    GridSizeHypothesis,
    abduct,
    grid_size_hypothesis,
)
from dense_armor.core.graph.vsa_hypotheses import (
    ObjectHypothesis,
    VSAObject,
    perceive,
)
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_operations import (
    execute_action,
    result_to_grid,
)


@dataclass
class Rule:
    """One learned rule: a condition (prototype) and an action.

    ``threshold`` is a cosine similarity chosen for this implementation. It is
    not the paper's rule (Algorithm 3: the operation predictor, a neural network,
    gives a probability, and the operation is executed from 50%).
    """

    prototype: np.ndarray
    action: ConcreteAction
    threshold: float = 0.3


@dataclass
class VSAProgram:
    """A program: object hypothesis, grid size hypothesis, list of rules."""

    hypothesis: ObjectHypothesis
    size_hypothesis: GridSizeHypothesis
    constant_size: tuple[int, int] | None
    rules: list[Rule]


def _learn_rules(
    task: dict,
    abduction: AbductionResult,
    encoder: ObjectEncoder,
) -> list[Rule]:
    """Algorithm 2: induce one rule per action in the action set.

    Each rule stores a prototype (the average VSA of objects the
    operation was applied to) and the concrete action. At deduction,
    an object activates the rule if its VSA similarity to the prototype
    exceeds the threshold.

    Args:
        task: the ARC task dict.
        abduction: the abduction result (Algorithm 1).
        encoder: the VSA encoder.

    Returns:
        the list of rules.
    """
    # Collect, per action, the VSA of the input objects it was applied to
    per_action_objs: dict[ConcreteAction, list[np.ndarray]] = {}
    pairs = [
        (Grid.from_list(ex["input"]), Grid.from_list(ex["output"]))
        for ex in task["train"]
    ]
    for grid_in, grid_out in pairs:
        objs_in = perceive(grid_in, abduction.hypothesis, encoder)
        objs_out = perceive(grid_out, abduction.hypothesis, encoder)
        for out_obj in objs_out:
            in_obj = _match_input(out_obj, objs_in)
            if in_obj is None:
                continue
            # find which action of the set actually maps in_obj to out_obj
            for action in abduction.action_set:
                results = execute_action(
                    in_obj.cells,
                    in_obj.colour,
                    action.op_type,
                    action.params,
                    grid_in,
                    encoder,
                )
                if not results:
                    continue
                # match by cells
                for res in results:
                    if res.cells == out_obj.cells and res.colour == out_obj.colour:
                        per_action_objs.setdefault(action, []).append(in_obj.sp)
                        break

    rules: list[Rule] = []
    for action, sps in per_action_objs.items():
        prototype = np.mean(sps, axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        rules.append(Rule(prototype=prototype, action=action))
    return rules


def _match_input(out_obj: VSAObject, objs_in: list[VSAObject]) -> VSAObject | None:
    if not objs_in:
        return None
    sims = [(similarity(out_obj.sp, o.sp), i) for i, o in enumerate(objs_in)]
    sims.sort(reverse=True)
    return objs_in[sims[0][1]]


def _deduce(
    grid_in: Grid,
    program: VSAProgram,
    encoder: ObjectEncoder,
) -> Grid:
    """Algorithm 3: apply the rules to the input, produce the output grid.

    Args:
        grid_in: the input grid.
        program: the learned program.
        encoder: the VSA encoder.

    Returns:
        the predicted output grid.
    """
    objs_in = perceive(grid_in, program.hypothesis, encoder)
    results = []
    for obj in objs_in:
        for rule in program.rules:
            sim = similarity(obj.sp, rule.prototype)
            if sim < rule.threshold:
                continue
            out = execute_action(
                obj.cells,
                obj.colour,
                rule.action.op_type,
                rule.action.params,
                grid_in,
                encoder,
            )
            results.extend(out)

    # Render output
    if program.size_hypothesis == GridSizeHypothesis.CONSTANT and program.constant_size:
        h, w = program.constant_size
        return _render_into(results, h, w)
    if program.size_hypothesis == GridSizeHypothesis.IDENTITY:
        return _render_into(results, *grid_in.shape(), anchor=False)
    return result_to_grid(results)


def _render_into(results, h: int, w: int, anchor: bool = True) -> Grid:
    """Render objects into a fixed-size grid.

    With ``anchor`` the top-left cell of all the objects goes to ``(0, 0)``;
    without it the cells keep their coordinates (same size as the input).
    """
    if not results:
        return Grid.from_list([[0] * w for _ in range(h)])
    all_cells = set()
    for r in results:
        all_cells |= r.cells
    r_min = min(r for r, _ in all_cells) if anchor else 0
    c_min = min(c for _, c in all_cells) if anchor else 0
    grid = [[0] * w for _ in range(h)]
    for r in results:
        for rr, cc in r.cells:
            nr, nc = rr - r_min, cc - c_min
            if 0 <= nr < h and 0 <= nc < w:
                grid[nr][nc] = r.colour
    return Grid.from_list(grid)


def solve(task: dict, encoder: ObjectEncoder) -> VSAProgram | None:
    """Full VSA solve: abduction -> rule induction -> deduction.

    Args:
        task: the ARC task dict.
        encoder: the VSA encoder.

    Returns:
        the learned VSAProgram, or None if abduction fails.
    """
    abduction = abduct(task, encoder, max_hypotheses=3)
    if abduction is None:
        return None
    pairs = [
        (Grid.from_list(ex["input"]), Grid.from_list(ex["output"]))
        for ex in task["train"]
    ]
    size_hyp, const_size = grid_size_hypothesis(pairs)
    rules = _learn_rules(task, abduction, encoder)
    if not rules:
        return None
    return VSAProgram(
        hypothesis=abduction.hypothesis,
        size_hypothesis=size_hyp,
        constant_size=const_size,
        rules=rules,
    )


def apply(program: VSAProgram, grid: Grid, encoder: ObjectEncoder) -> Grid:
    """Apply a learned program to a grid.

    Args:
        program: the learned program.
        grid: the input grid.
        encoder: the VSA encoder.

    Returns:
        the predicted output grid.
    """
    return _deduce(grid, program, encoder)
