"""VSA solver with property selection (cross-validation).

Implements the property-selection step of the paper §3.2.2, page 14:
"for each demonstration, our solver, using a particular property
hypothesis, trains a predictor on data from all other demonstrations
and tests the predictor on data from the held out demonstration".

The 'predictor' here is the full VSAProgram of this task. The property
hypotheses are which combination of {colour, centre, shape} is used in
the VSA encodings during abduction and rule learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa_hypotheses import (
    VSAObject,
    perceive,
)
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_solver import (
    VSAProgram,
)
from dense_armor.core.graph.vsa_solver import (
    apply as vsa_apply,
)


@dataclass
class VSAVariant:
    """One property hypothesis (which VSA component to use).

    The projection (colour, centre or shape) is used where ``learn_variant``
    chooses the input object that corresponds to each output object. The rules
    (``_learn_rules``) and the deduction (``apply``) use the full vector of the
    object in every variant, so the variants differ only in that matching. A
    variant adds no operation: a task that needs one outside Table 1 (for
    example a comparison of objects) is out of reach for all of them (paper,
    Section 5.2).
    """

    name: str
    encoder: ObjectEncoder

    def project(self, obj: VSAObject, grid: Grid) -> np.ndarray:
        """Return the VSA of ``obj`` under this variant's projection."""
        if self.name == "all":
            return obj.sp
        if self.name == "colour":
            # Colour only: the colour SP
            return self.encoder.colour_sps[obj.colour]
        if self.name == "centre":
            # Centre only: recompute the centre SSP
            h, w = grid.shape()
            r_min = min(r for r, _ in obj.cells)
            r_max = max(r for r, _ in obj.cells)
            c_min = min(c for _, c in obj.cells)
            c_max = max(c for _, c in obj.cells)
            cx = (r_min + r_max) / 2.0
            cy = (c_min + c_max) / 2.0
            rel_x = cx - (h - 1) / 2.0
            rel_y = cy - (w - 1) / 2.0
            return self.encoder.ssp.encode(np.array([rel_x, rel_y]))
        if self.name == "shape":
            # Shape only: recompute the shape SSP
            r_min = min(r for r, _ in obj.cells)
            c_min = min(c for _, c in obj.cells)
            r_max = max(r for r, _ in obj.cells)
            c_max = max(c for _, c in obj.cells)
            cx = (r_min + r_max) / 2.0
            cy = (c_min + c_max) / 2.0
            rel_positions = np.array(
                [(r - cx, c - cy) for r, c in obj.cells], dtype=float
            )
            from dense_armor.core.graph.vsa import bundle

            pixel_sps = self.encoder.ssp.encode(rel_positions)
            return bundle(*pixel_sps)
        return obj.sp


def _project_objs(
    objs: list[VSAObject], grid: Grid, variant: VSAVariant
) -> list[VSAObject]:
    """Re-emit the objects with the projected VSA."""
    return [
        VSAObject(cells=o.cells, colour=o.colour, sp=variant.project(o, grid))
        for o in objs
    ]


def learn_variant(
    task: dict,
    variant: VSAVariant,
) -> VSAProgram | None:
    """Abduct + rule induction, using the variant's projection.

    Args:
        task: the ARC task dict.
        variant: the property hypothesis.

    Returns:
        the learned program, or None if abduction fails.
    """
    # Re-implement abduct with the projected objects
    from dense_armor.core.graph.vsa_abduction import (
        _candidate_actions,
        _input_object_for,
        action_cost,
        grid_size_hypothesis,
        minimum_hitting_set,
    )
    from dense_armor.core.graph.vsa_hypotheses import object_hypothesis_heuristic
    from dense_armor.core.graph.vsa_solver import _learn_rules

    pairs = [
        (Grid.from_list(ex["input"]), Grid.from_list(ex["output"]))
        for ex in task["train"]
    ]
    if not pairs:
        return None
    hyp_scores = object_hypothesis_heuristic(pairs[0][0], pairs[0][1], variant.encoder)
    top_hyps = [h for h, _ in hyp_scores[:3]]
    for hyp in top_hyps:
        per_output_actions: list[list] = []
        ok = True
        for grid_in, grid_out in pairs:
            objs_in_raw = perceive(grid_in, hyp, variant.encoder)
            objs_out_raw = perceive(grid_out, hyp, variant.encoder)
            objs_in = _project_objs(objs_in_raw, grid_in, variant)
            objs_out = _project_objs(objs_out_raw, grid_out, variant)
            if not objs_in or not objs_out:
                ok = False
                break
            for out_obj in objs_out:
                in_obj = _input_object_for(out_obj, objs_in)
                if in_obj is None:
                    ok = False
                    break
                # candidate actions are computed on the RAW objects (real cells)
                raw_in = next((o for o in objs_in_raw if o.cells == in_obj.cells), None)
                raw_out = next(
                    (o for o in objs_out_raw if o.cells == out_obj.cells), None
                )
                if raw_in is None or raw_out is None:
                    ok = False
                    break
                candidates = _candidate_actions(
                    raw_in, raw_out, grid_in, variant.encoder
                )
                if not candidates:
                    ok = False
                    break
                per_output_actions.append(candidates)
            if not ok:
                break
        if not ok:
            continue
        action_set = minimum_hitting_set(
            [set(c) for c in per_output_actions], cost=action_cost
        )
        from dense_armor.core.graph.vsa_abduction import AbductionResult

        abduction = AbductionResult(
            hypothesis=hyp,
            action_set=action_set,
            per_output_actions=per_output_actions,
        )
        size_hyp, const_size = grid_size_hypothesis(pairs)
        rules = _learn_rules(task, abduction, variant.encoder)
        if rules:
            return VSAProgram(
                hypothesis=hyp,
                size_hypothesis=size_hyp,
                constant_size=const_size,
                rules=rules,
            )
    return None


def predict_variant(program: VSAProgram, grid: Grid, variant: VSAVariant) -> Grid:
    """Apply a program learned with this variant."""
    return vsa_apply(program, grid, variant.encoder)


VARIANTS = ("all", "colour", "centre", "shape")


def select_best_variant(
    task: dict,
    encoder: ObjectEncoder,
) -> tuple[str, VSAProgram | None, float]:
    """LOO cross-validation over the 4 property hypotheses.

    For each variant, leave one training pair out, learn on the others,
    predict the held-out output, and score. Return the variant with
    the highest mean accuracy.

    Args:
        task: the ARC task dict.
        encoder: the VSA encoder.

    Returns:
        (best variant name, its full-training program, mean LOO accuracy)
    """
    pairs = task["train"]
    if len(pairs) < 2:
        # Cannot do LOO with a single pair
        v = VSAVariant("all", encoder)
        return ("all", learn_variant(task, v), 0.0)
    best_name = "all"
    best_acc = -1.0
    for name in VARIANTS:
        variant = VSAVariant(name, encoder)
        correct = 0
        for i in range(len(pairs)):
            train_subset = {
                "train": [ex for j, ex in enumerate(pairs) if j != i],
                "test": [],
            }
            program = learn_variant(train_subset, variant)
            if program is None:
                continue
            g_in = Grid.from_list(pairs[i]["input"])
            g_out = Grid.from_list(pairs[i]["output"])
            pred = predict_variant(program, g_in, variant)
            if pred == g_out:
                correct += 1
        acc = correct / len(pairs)
        if acc > best_acc:
            best_acc = acc
            best_name = name
    best_variant = VSAVariant(best_name, encoder)
    return (best_name, learn_variant(task, best_variant), best_acc)
