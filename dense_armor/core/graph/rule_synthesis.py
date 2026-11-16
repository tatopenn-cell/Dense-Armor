"""Rule synthesis from graph diffs.

Synthesizes a list of ``Rule(selector, effector)`` from the training
pairs of an ARC task. A rule pairs a selector (at most three predicates
on ``ArcObject``) with an effector (a change on the selected objects).

Effectors cover:

- object-scoped: RECOLOR, RECOLOR_BY_MAP (stable bijection),
  RECOLOR_RELATIONAL (colour of the nearest anchor, or of an isolated
  anchor per target), RECOLOR_NEAREST (nearest anchor by Chebyshev
  distance), DELETE, TRANSLATE, CLONE_AROUND, CLONE, FILL_HOLES,
  FILL_BG, FILL_L2.
- global: ROT90, ROT180, ROT270, HMIRROR, VMIRROR, DMIRROR,
  UPSCALE2, UPSCALE3, DOWNSCALE2, DOWNSCALE3.

When no single effector is common to every training pair,
``RECOLOR_NEAREST`` is tried via ``_try_buco_b_recolor``: each target
object takes the colour of the nearest object matching an anchor
predicate, and the target predicate isolates the targets.

Known limits: merge (an output object is the union of several inputs),
generic LGG, and any rule whose selector would need more than three
predicates are not covered.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import ArcObject

# --- vocabularies ---


class PredicateType(str, Enum):
    """Finite vocabulary Σ_S, lexicographically ordered for tie-break."""

    AREA_GT = "area_gt"
    AREA_LT = "area_lt"
    HAS_COLOR = "has_color"
    HAS_HOLE = "has_hole"
    IS_LARGEST = "is_largest"
    IS_RECTANGULAR = "is_rectangular"
    IS_SMALLEST = "is_smallest"
    IS_SQUARE = "is_square"
    RANK = "rank"
    TOUCHES_BORDER = "touches_border"


class EffectorType(str, Enum):
    """Finite vocabulary Σ_E."""

    CLONE = "clone"
    CLONE_AROUND = "clone_around"
    DELETE = "delete"
    FILL_BG = "fill_bg"
    FILL_HOLES = "fill_holes"
    FILL_L2 = "fill_l2"
    DMIRROR = "dmirror"
    DOWNSCALE2 = "downscale2"
    DOWNSCALE3 = "downscale3"
    HMIRROR = "hmirror"
    RECOLOR = "recolor"
    RECOLOR_BY_MAP = "recolor_by_map"
    RECOLOR_NEAREST = "recolor_nearest"
    RECOLOR_RELATIONAL = "recolor_relational"
    ROT90 = "rot90"
    ROT180 = "rot180"
    ROT270 = "rot270"
    TRANSLATE = "translate"
    UPSCALE2 = "upscale2"
    UPSCALE3 = "upscale3"
    VMIRROR = "vmirror"


GLOBAL_EFFECTORS: frozenset[EffectorType] = frozenset(
    {
        EffectorType.DMIRROR,
        EffectorType.DOWNSCALE2,
        EffectorType.DOWNSCALE3,
        EffectorType.HMIRROR,
        EffectorType.ROT90,
        EffectorType.ROT180,
        EffectorType.ROT270,
        EffectorType.UPSCALE2,
        EffectorType.UPSCALE3,
        EffectorType.VMIRROR,
    }
)


@dataclass(frozen=True)
class Predicate:
    """Atomic predicate on an object."""

    kind: PredicateType
    arg: int | None = None


@dataclass(frozen=True)
class Effector:
    """Effector with concrete parameters."""

    kind: EffectorType
    to_color: int | None = None
    dr: int | None = None
    dc: int | None = None
    source_idx: int | None = None
    l2_name: str | None = None
    mapping: tuple[tuple[int, int], ...] | None = None
    deltas: tuple[tuple[int, int], ...] | None = None
    anchor_predicate: tuple[Predicate, ...] | None = None
    target_predicate: tuple[Predicate, ...] | None = None


@dataclass(frozen=True)
class Rule:
    """A selector-effector rule."""

    selector: tuple[Predicate, ...]
    effector: Effector

    def mdl_bits(self) -> float:
        params = 0
        if self.effector.to_color is not None:
            params += 1
        if self.effector.dr is not None:
            params += 1
        if self.effector.dc is not None:
            params += 1
        if self.effector.mapping is not None:
            params += 2 * len(self.effector.mapping)
        if self.effector.deltas is not None:
            params += len(self.effector.deltas)
        sel_bits = len(self.selector) * math.log2(len(SIGMA_S))
        if self.effector.anchor_predicate is not None:
            sel_bits += len(self.effector.anchor_predicate) * math.log2(len(SIGMA_S))
        if self.effector.target_predicate is not None:
            sel_bits += len(self.effector.target_predicate) * math.log2(len(SIGMA_S))
        return sel_bits + math.log2(len(SIGMA_E)) + params * math.log2(30)


SIGMA_S: list[Predicate] = []
for c in range(1, 10):
    SIGMA_S.append(Predicate(PredicateType.HAS_COLOR, c))
for n in (1, 2, 3, 5, 10):
    SIGMA_S.append(Predicate(PredicateType.AREA_GT, n))
    SIGMA_S.append(Predicate(PredicateType.AREA_LT, n))
SIGMA_S += [
    Predicate(PredicateType.HAS_HOLE),
    Predicate(PredicateType.IS_LARGEST),
    Predicate(PredicateType.IS_RECTANGULAR),
    Predicate(PredicateType.IS_SMALLEST),
    Predicate(PredicateType.IS_SQUARE),
    Predicate(PredicateType.TOUCHES_BORDER),
]
for k in (1, 2, 3, 4, 5):
    SIGMA_S.append(Predicate(PredicateType.RANK, k))
# sort per tie-break lessicografico
SIGMA_S.sort(key=lambda p: (p.kind.value, p.arg or 0))

SIGMA_E: list[Effector] = [Effector(EffectorType.DELETE)]
for c in range(1, 10):
    SIGMA_E.append(Effector(EffectorType.RECOLOR, to_color=c))
    SIGMA_E.append(Effector(EffectorType.FILL_HOLES, to_color=c))
    SIGMA_E.append(Effector(EffectorType.FILL_BG, to_color=c))
for dr in (-1, 1):
    for dc in (-1, 1):
        SIGMA_E.append(Effector(EffectorType.TRANSLATE, dr=dr, dc=dc))


# --- predicates ---


def _eval(p: Predicate, obj: ArcObject, all_objs: list[ArcObject], grid: Grid) -> bool:
    if p.kind == PredicateType.HAS_COLOR:
        return obj.color == p.arg
    if p.kind == PredicateType.AREA_GT:
        return obj.area > (p.arg or 0)
    if p.kind == PredicateType.AREA_LT:
        return obj.area < (p.arg or 0)
    if p.kind == PredicateType.HAS_HOLE:
        return len(obj.holes(grid)) > 0
    if p.kind == PredicateType.IS_LARGEST:
        if obj.color is None or obj.color == 0:
            return False
        fg = [o for o in all_objs if o.color is not None and o.color != 0]
        return all(o.area <= obj.area for o in fg)
    if p.kind == PredicateType.IS_RECTANGULAR:
        return obj.is_rectangular()
    if p.kind == PredicateType.IS_SMALLEST:
        if obj.color is None or obj.color == 0:
            return False
        fg = [o for o in all_objs if o.color is not None and o.color != 0]
        return all(o.area >= obj.area for o in fg)
    if p.kind == PredicateType.RANK:
        if obj.color is None or obj.color == 0:
            return False
        target = p.arg or 1
        if target < 1:
            return False
        larger = sum(
            1
            for o in all_objs
            if o.color is not None and o.color != 0 and o.area > obj.area
        )
        return larger == target - 1
    if p.kind == PredicateType.IS_SQUARE:
        r0, c0, r1, c1 = obj.bbox
        return (r1 - r0) == (c1 - c0)
    if p.kind == PredicateType.TOUCHES_BORDER:
        return obj.touches_border(grid.shape())
    return False


def _selector_matches(
    sel: tuple[Predicate, ...], obj: ArcObject, all_objs: list[ArcObject], grid: Grid
) -> bool:
    return all(_eval(p, obj, all_objs, grid) for p in sel)


# --- shape signature and anchor ---


def shape_signature(obj: ArcObject) -> frozenset[tuple[int, int]]:
    """Normalized cell mask, top-left at (0, 0)."""
    r0, c0 = obj.bbox[0], obj.bbox[1]
    return frozenset((r - r0, c - c0) for r, c in obj.cells)


def anchor(obj: ArcObject) -> tuple[float, float]:
    r0, c0, r1, c1 = obj.bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


# --- positives extraction with CLONE and FILL_L2 ---


def extract_positives(
    diff: dict, grid_I: Grid, grid_O: Grid
) -> dict[Effector, list[int]]:
    """Map effectors to the input-side positive indices (or -1 for bg/L2).

    Args:
        diff: result of ``diff_graphs``.
        grid_I: input grid.
        grid_O: output grid.

    Returns:
        a dict ``{effector: [indices into objects_I, or -1 for bg/L2]}``.
    """
    objs_I = diff["objects_I"]
    objs_O = diff["objects_O"]
    positives: dict[Effector, list[int]] = {}

    # matched: recolor, translate, fill_holes
    for i, j in diff["matched"]:
        u, v = objs_I[i], objs_O[j]
        if u.color != v.color and v.color is not None:
            eff = Effector(EffectorType.RECOLOR, to_color=v.color)
            positives.setdefault(eff, []).append(i)
        ur, uc = anchor(u)
        vr, vc = anchor(v)
        dr, dc = int(round(vr - ur)), int(round(vc - uc))
        if (dr, dc) != (0, 0):
            eff = Effector(EffectorType.TRANSLATE, dr=dr, dc=dc)
            positives.setdefault(eff, []).append(i)
        if len(u.holes(grid_I)) > len(v.holes(grid_O)):
            hO, wO = grid_O.shape()
            for hole in u.holes(grid_I):
                cols = {grid_O.at(r, c) for r, c in hole if 0 <= r < hO and 0 <= c < wO}
                for color in cols:
                    eff = Effector(EffectorType.FILL_HOLES, to_color=color)
                    positives.setdefault(eff, []).append(i)

    mapping_dict: dict[int, int] = {}
    all_matched_recolor = bool(diff["matched"])
    for i, j in diff["matched"]:
        u, v = objs_I[i], objs_O[j]
        if u.color is None or v.color is None or u.color == v.color:
            all_matched_recolor = False
            break
        if u.color in mapping_dict and mapping_dict[u.color] != v.color:
            all_matched_recolor = False
            break
        mapping_dict[u.color] = v.color
    if all_matched_recolor and len(mapping_dict) >= 2:
        eff = Effector(
            EffectorType.RECOLOR_BY_MAP,
            mapping=tuple(sorted(mapping_dict.items())),
        )
        for i in range(len(objs_I)):
            positives.setdefault(eff, []).append(i)

    # deleted
    for i in diff["deleted"]:
        positives.setdefault(Effector(EffectorType.DELETE), []).append(i)

    # created: CLONE_AROUND, CLONE, or FILL_L2
    created_around: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for j in diff["created"]:
        v = objs_O[j]
        v_cells = frozenset(v.cells)
        v_color = v.color if v.color is not None else 0
        around_found = False
        for i, u in enumerate(objs_I):
            if u.color is None or u.color == 0 or u.area != 1:
                continue
            ur, uc = next(iter(u.cells))
            deltas = tuple(sorted((r - ur, c - uc) for r, c in v_cells))
            if any(d == (0, 0) for d in deltas):
                continue
            if any(max(abs(dr_), abs(dc_)) > 2 for dr_, dc_ in deltas):
                continue
            if len(deltas) < 2:
                continue
            key = (u.color, v_color, deltas)
            created_around[key] = created_around.get(key, 0) + 1
            around_found = True
            break
        if around_found:
            continue
        sig_v = shape_signature(v)
        clone_source = None
        for i, u in enumerate(objs_I):
            if shape_signature(u) == sig_v:
                if clone_source is None or (u.bbox[0], u.bbox[1]) < (
                    objs_I[clone_source].bbox[0],
                    objs_I[clone_source].bbox[1],
                ):
                    clone_source = i
        if clone_source is not None:
            u = objs_I[clone_source]
            ur, uc = anchor(u)
            vr, vc = anchor(v)
            dr, dc = int(round(vr - ur)), int(round(vc - uc))
            color = v.color if v.color is not None else u.color
            eff = Effector(
                EffectorType.CLONE,
                dr=dr,
                dc=dc,
                to_color=color,
                source_idx=clone_source,
            )
            positives.setdefault(eff, []).append(clone_source)
            continue
        # FILL_L2: cell set of v matches a cavity node in L2_I
        for name, node in diff["L2_I"].items():
            if node["type"] != "cavity":
                continue
            if frozenset(node["state"]["cells"]) == v_cells:
                color = v.color if v.color is not None else 0
                eff = Effector(EffectorType.FILL_L2, to_color=color, l2_name=name)
                positives.setdefault(eff, []).append(-1)
                break

    # emit CLONE_AROUND effectors discovered above
    for (u_color, v_color, deltas), _count in created_around.items():
        eff = Effector(
            EffectorType.CLONE_AROUND,
            to_color=v_color,
            deltas=deltas,
        )
        for i, u in enumerate(objs_I):
            if u.color == u_color and u.area == 1:
                positives.setdefault(eff, []).append(i)

    # global effectors: if the whole output is a rot/mirror/scale of the
    # input, emit the corresponding global effector. Selector is ().
    for glob_eff in (
        Effector(EffectorType.ROT90),
        Effector(EffectorType.ROT180),
        Effector(EffectorType.ROT270),
        Effector(EffectorType.HMIRROR),
        Effector(EffectorType.VMIRROR),
        Effector(EffectorType.DMIRROR),
        Effector(EffectorType.UPSCALE2),
        Effector(EffectorType.UPSCALE3),
        Effector(EffectorType.DOWNSCALE2),
        Effector(EffectorType.DOWNSCALE3),
    ):
        try:
            if apply_global(grid_I, glob_eff) == grid_O:
                positives.setdefault(glob_eff, []).append(-1)
        except Exception:
            continue

    # fill_bg: same-dimension only, background color changes
    hI, wI = grid_I.shape()
    hO, wO = grid_O.shape()
    if (hI, wI) == (hO, wO):
        bg_I = Counter(
            grid_I.at(r, c) for r in range(hI) for c in range(wI)
        ).most_common(1)[0][0]
        bg_O = Counter(
            grid_O.at(r, c) for r in range(hO) for c in range(wO)
        ).most_common(1)[0][0]
        if bg_I != bg_O:
            eff = Effector(EffectorType.FILL_BG, to_color=bg_O)
            positives.setdefault(eff, []).append(-1)

    return positives


# --- global selector synthesis on (P, N) ---


def synthesize_selector_global(
    per_pair_pos: list[list[int]],
    per_pair_neg: list[list[int]],
    per_pair_objs: list[list[ArcObject]],
    per_pair_grids: list[Grid],
) -> tuple[Predicate, ...] | None:
    """Synthesize a selector that matches all positives and no negative.

    Args:
        per_pair_pos: positive indices per pair.
        per_pair_neg: negative indices per pair.
        per_pair_objs: objects per pair.
        per_pair_grids: grids per pair.

    Returns:
        a selector of size <= 3, or ``None``.
    """
    for size in (0, 1, 2, 3):
        candidates = list(combinations(SIGMA_S, size))
        for sel in candidates:
            ok = True
            for pos_idxs, neg_idxs, objs, grid in zip(
                per_pair_pos, per_pair_neg, per_pair_objs, per_pair_grids
            ):
                for i in pos_idxs:
                    if not _selector_matches(sel, objs[i], objs, grid):
                        ok = False
                        break
                if not ok:
                    break
                for i in neg_idxs:
                    if _selector_matches(sel, objs[i], objs, grid):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return sel
    return None


# --- anchor discovery for parametrized effectors ---


# --- render ---


def apply_global(grid: Grid, eff: Effector) -> Grid:
    """Apply one global effector (rot/mirror/scale) to a grid.

    Args:
        grid: the input grid.
        eff: the effector.

    Returns:
        the transformed grid.
    """
    from dense_armor.core.graph import dsl

    if eff.kind == EffectorType.ROT90:
        return dsl.rot90(grid)
    if eff.kind == EffectorType.ROT180:
        return dsl.rot180(grid)
    if eff.kind == EffectorType.ROT270:
        return dsl.rot270(grid)
    if eff.kind == EffectorType.HMIRROR:
        return dsl.hmirror(grid)
    if eff.kind == EffectorType.VMIRROR:
        return dsl.vmirror(grid)
    if eff.kind == EffectorType.DMIRROR:
        return dsl.dmirror(grid)
    if eff.kind == EffectorType.UPSCALE2:
        return dsl.upscale(grid, 2)
    if eff.kind == EffectorType.UPSCALE3:
        return dsl.upscale(grid, 3)
    if eff.kind == EffectorType.DOWNSCALE2:
        return dsl.downscale(grid, 2)
    if eff.kind == EffectorType.DOWNSCALE3:
        return dsl.downscale(grid, 3)
    raise ValueError(f"not a global effector: {eff.kind}")


def _render_effector(
    eff: Effector,
    obj: ArcObject,
    grid: Grid,
    objs: list[ArcObject],
) -> list[tuple[int, int, int]]:
    """Apply one effector to one object, returning (r, c, color) cells."""
    if eff.kind == EffectorType.RECOLOR:
        return [(r, c, eff.to_color or 0) for r, c in obj.cells]
    if eff.kind == EffectorType.RECOLOR_BY_MAP:
        if eff.mapping is None or obj.color is None:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        mapping = dict(eff.mapping)
        return [(r, c, mapping.get(obj.color, obj.color)) for r, c in obj.cells]
    if eff.kind == EffectorType.RECOLOR_RELATIONAL:
        if eff.anchor_predicate is None:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        if eff.target_predicate is not None:
            if not _selector_matches(eff.target_predicate, obj, objs, grid):
                return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        anchor_color = None
        for other in objs:
            if other is obj:
                continue
            if _selector_matches(eff.anchor_predicate, other, objs, grid):
                anchor_color = other.color
                break
        if anchor_color is None:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        return [(r, c, anchor_color) for r, c in obj.cells]
    if eff.kind == EffectorType.RECOLOR_NEAREST:
        if eff.anchor_predicate is None:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        if eff.target_predicate is not None:
            if not _selector_matches(eff.target_predicate, obj, objs, grid):
                return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        anchors = [
            other
            for other in objs
            if other is not obj
            and _selector_matches(eff.anchor_predicate, other, objs, grid)
        ]
        if not anchors:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        best_anchor = None
        best_dist = float("inf")
        obj_cells = list(obj.cells)
        for anchor in anchors:
            anchor_cells = list(anchor.cells)
            d = min(
                max(abs(r1 - r2), abs(c1 - c2))
                for r1, c1 in obj_cells
                for r2, c2 in anchor_cells
            )
            if d < best_dist:
                best_dist = d
                best_anchor = anchor
        if best_anchor is None or best_anchor.color is None:
            return [(r, c, grid.at(r, c)) for r, c in obj.cells]
        return [(r, c, best_anchor.color) for r, c in obj.cells]
    if eff.kind == EffectorType.DELETE:
        return [(r, c, 0) for r, c in obj.cells]
    if eff.kind == EffectorType.TRANSLATE:
        dr, dc = eff.dr or 0, eff.dc or 0
        return [(r + dr, c + dc, grid.at(r, c)) for r, c in obj.cells]
    if eff.kind == EffectorType.CLONE_AROUND:
        if eff.deltas is None:
            return []
        cells: list[tuple[int, int, int]] = []
        color = eff.to_color or 0
        for r, c in obj.cells:
            for dr, dc in eff.deltas:
                cells.append((r + dr, c + dc, color))
        return cells
    if eff.kind == EffectorType.FILL_HOLES:
        cells = [(r, c, grid.at(r, c)) for r, c in obj.cells]
        for hole in obj.holes(grid):
            for r, c in hole:
                cells.append((r, c, eff.to_color or 0))
        return cells
    return [(r, c, grid.at(r, c)) for r, c in obj.cells]


def _try_buco_b_recolor(
    per_pair_pos: list[dict[Effector, list[int]]],
    per_pair_objs: list[list[ArcObject]],
    per_pair_grids: list[Grid],
) -> tuple[tuple[Predicate, ...], tuple[Predicate, ...]] | None:
    """Buco B colore, versione nearest-match (v9).

    Anti-tautology: the anchor predicate must not contain HAS_COLOR(c)
    for any c that appears as a to_color in any pair.

    Relaxed condition: for every target, among all non-target objects
    matching phi, the **nearest one** must have color == to_color. This
    is what RECOLOR_NEAREST does at render time. The stronger v6-v8
    condition (phi isolates exactly one anchor per target) is not
    required, because RECOLOR_NEAREST chooses the nearest anchor for
    each target separately, so multiple anchors per target are fine as
    long as the nearest one has the right color.

    The target predicate must match exactly the targets (and no
    non-target) in every pair.

    Args:
        per_pair_pos: per-pair positives.
        per_pair_objs: per-pair objects.
        per_pair_grids: per-pair grids.

    Returns:
        (anchor_predicate, target_predicate), or None.
    """
    per_pair_target_color: list[dict[int, int]] = []
    forbidden_colors: set[int] = set()
    for pos in per_pair_pos:
        mapping: dict[int, int] = {}
        for eff, idxs in pos.items():
            if eff.kind != EffectorType.RECOLOR:
                continue
            if eff.to_color is not None:
                forbidden_colors.add(eff.to_color)
            for i in idxs:
                if i < 0:
                    continue
                if eff.to_color is not None:
                    mapping[i] = eff.to_color
        per_pair_target_color.append(mapping)

    if not all(m for m in per_pair_target_color):
        return None

    def _chebyshev(a: ArcObject, b: ArcObject) -> int:
        return min(
            max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a.cells for r2, c2 in b.cells
        )

    anchor_sel: tuple[Predicate, ...] | None = None
    for size in (1, 2):
        for sel in combinations(SIGMA_S, size):
            if any(
                p.kind == PredicateType.HAS_COLOR and p.arg in forbidden_colors
                for p in sel
            ):
                continue
            ok = True
            for k, (mapping, objs, grid) in enumerate(
                zip(per_pair_target_color, per_pair_objs, per_pair_grids)
            ):
                for target_idx, to_color in mapping.items():
                    target = objs[target_idx]
                    anchors = [
                        r
                        for r_idx, r in enumerate(objs)
                        if r_idx != target_idx and _selector_matches(sel, r, objs, grid)
                    ]
                    if not anchors:
                        ok = False
                        break
                    nearest = min(anchors, key=lambda r: _chebyshev(target, r))
                    if nearest.color != to_color:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                anchor_sel = sel
                break
        if anchor_sel is not None:
            break

    if anchor_sel is None:
        return None

    for size in (1, 2):
        for sel in combinations(SIGMA_S, size):
            ok = True
            for k, (mapping, objs, grid) in enumerate(
                zip(per_pair_target_color, per_pair_objs, per_pair_grids)
            ):
                for i, o in enumerate(objs):
                    is_target = i in mapping
                    is_match = _selector_matches(sel, o, objs, grid)
                    if is_target != is_match:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return (anchor_sel, sel)
    return None


def render(grid: Grid, rules: list[Rule], L2_I: dict) -> Grid:
    """Apply a rule list to a grid."""
    from dense_armor.core.graph.objects import extract_objects

    # Phase 1: global effectors, in D1 order (one at most, per spec)
    g = grid
    object_rules = []
    for rule in rules:
        if rule.effector.kind in GLOBAL_EFFECTORS:
            g = apply_global(g, rule.effector)
        else:
            object_rules.append(rule)

    # Phase 2: object effectors, on the current grid
    grid = g
    h, w = grid.shape()
    out = [[grid.at(r, c) for c in range(w)] for r in range(h)]
    objs = extract_objects(grid, by_color=True)

    for rule in object_rules:
        eff = rule.effector
        if eff.kind == EffectorType.FILL_BG:
            bg = Counter(grid.at(r, c) for r in range(h) for c in range(w)).most_common(
                1
            )[0][0]
            for r in range(h):
                for c in range(w):
                    if out[r][c] == bg:
                        out[r][c] = eff.to_color or 0
            continue
        if eff.kind == EffectorType.FILL_L2:
            node = L2_I.get(eff.l2_name or "")
            if node is None:
                continue
            for r, c in node["state"]["cells"]:
                if 0 <= r < h and 0 <= c < w:
                    out[r][c] = eff.to_color or 0
            continue
        if eff.kind == EffectorType.CLONE:
            src_idx = eff.source_idx
            if src_idx is None or src_idx >= len(objs):
                continue
            src = objs[src_idx]
            for r, c in src.cells:
                nr, nc = r + (eff.dr or 0), c + (eff.dc or 0)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = eff.to_color or grid.at(r, c)
            continue
        # object-scoped effectors
        for obj in objs:
            if not _selector_matches(rule.selector, obj, objs, grid):
                continue
            for r, c, color in _render_effector(eff, obj, grid, objs):
                if 0 <= r < h and 0 <= c < w:
                    out[r][c] = color
    return Grid.from_list(out)


# --- top-level synthesis ---


def synthesize(task: dict, max_rules: int = 5) -> list[Rule] | None:
    """Synthesize a rule list for ``task``.

    Args:
        task: the ARC task dict.
        max_rules: maximum number of rules to try.

    Returns:
        a list of ``Rule`` reproducing training exactly, or ``None``.
    """
    from dense_armor.core.graph.graph_diff import diff_graphs

    pairs = [
        (Grid.from_list(ex["input"]), Grid.from_list(ex["output"]))
        for ex in task["train"]
    ]
    if not pairs:
        return None

    per_pair_pos: list[dict[Effector, list[int]]] = []
    per_pair_objs: list[list[ArcObject]] = []
    per_pair_grids: list[Grid] = []
    per_pair_L2: list[dict] = []

    for grid_I, grid_O in pairs:
        d = diff_graphs(grid_I, grid_O)
        pos = extract_positives(d, grid_I, grid_O)
        per_pair_pos.append(pos)
        per_pair_objs.append(d["objects_I"])
        per_pair_grids.append(grid_I)
        per_pair_L2.append(d["L2_I"])

    # effectors that appear in every pair
    common = set(per_pair_pos[0])
    for p in per_pair_pos[1:]:
        common &= set(p)

    # Buco B colore: if common is empty, try RECOLOR_RELATIONAL then RECOLOR_NEAREST
    if not common:
        result = _try_buco_b_recolor(per_pair_pos, per_pair_objs, per_pair_grids)
        if result is None:
            return None
        anchor_sel, target_sel = result
        eff = Effector(
            EffectorType.RECOLOR_NEAREST,
            anchor_predicate=anchor_sel,
            target_predicate=target_sel,
        )
        for k, (grid_I, grid_O) in enumerate(pairs):
            L2 = per_pair_L2[k]
            rendered = render(grid_I, [Rule(selector=(), effector=eff)], L2)
            if rendered != grid_O:
                return None
        return [Rule(selector=(), effector=eff)]

    rules: list[Rule] = []
    for eff in sorted(common, key=repr):  # a fixed order, not the order of the set
        # collect per-pair positives and negatives (excluding -1)
        per_pair_pos_list: list[list[int]] = []
        per_pair_neg_list: list[list[int]] = []
        skip_eff = False
        for k in range(len(pairs)):
            idxs = per_pair_pos[k][eff]
            pos_idxs = [i for i in idxs if i >= 0]
            if not pos_idxs and -1 not in idxs:
                skip_eff = True
                break
            per_pair_pos_list.append(pos_idxs)
            n_obj = len(per_pair_objs[k])
            per_pair_neg_list.append([i for i in range(n_obj) if i not in pos_idxs])
        if skip_eff:
            continue

        sel: tuple[Predicate, ...]
        if (
            eff.kind
            in (
                EffectorType.FILL_BG,
                EffectorType.FILL_L2,
                EffectorType.RECOLOR_BY_MAP,
            )
            or eff.kind in GLOBAL_EFFECTORS
        ):
            sel = ()
        elif eff.kind == EffectorType.CLONE:
            if len({eff.source_idx}) != 1:
                continue
            sel = ()
        elif eff.kind == EffectorType.CLONE_AROUND:
            source_colors: set[int] = set()
            for k in range(len(pairs)):
                for idx in per_pair_pos[k].get(eff, []):
                    if idx >= 0 and idx < len(per_pair_objs[k]):
                        c = per_pair_objs[k][idx].color
                        if c is not None:
                            source_colors.add(c)
            if len(source_colors) != 1:
                continue
            source_color = next(iter(source_colors))
            sel = (Predicate(PredicateType.HAS_COLOR, source_color),)
        else:
            maybe_sel = synthesize_selector_global(
                per_pair_pos_list,
                per_pair_neg_list,
                per_pair_objs,
                per_pair_grids,
            )
            if maybe_sel is None:
                continue
            sel = maybe_sel
        rules.append(Rule(selector=sel, effector=eff))

    if not rules:
        return None

    order = {
        EffectorType.ROT90: 0,
        EffectorType.ROT180: 1,
        EffectorType.ROT270: 2,
        EffectorType.HMIRROR: 3,
        EffectorType.VMIRROR: 4,
        EffectorType.DMIRROR: 5,
        EffectorType.UPSCALE2: 6,
        EffectorType.UPSCALE3: 7,
        EffectorType.DOWNSCALE2: 8,
        EffectorType.DOWNSCALE3: 9,
        EffectorType.DELETE: 10,
        EffectorType.TRANSLATE: 11,
        EffectorType.RECOLOR: 12,
        EffectorType.RECOLOR_BY_MAP: 13,
        EffectorType.RECOLOR_RELATIONAL: 13,
        EffectorType.RECOLOR_NEAREST: 13,
        EffectorType.FILL_HOLES: 14,
        EffectorType.FILL_L2: 15,
        EffectorType.CLONE: 16,
        EffectorType.CLONE_AROUND: 17,
        EffectorType.FILL_BG: 18,
    }
    rules.sort(key=lambda r: order.get(r.effector.kind, 99))

    for size in range(1, min(len(rules), max_rules) + 1):
        for subset in combinations(rules, size):
            ok = True
            for (grid_I, grid_O), L2_I in zip(pairs, per_pair_L2):
                if render(grid_I, list(subset), L2_I) != grid_O:
                    ok = False
                    break
            if ok:
                return list(subset)
    return None
