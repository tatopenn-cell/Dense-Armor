"""VSA: the six object hypotheses and the VSA cleanup decoders.

References: Joffe & Eliasmith (2025), arXiv:2511.08747.
    - Table 3 (page 13): the six object hypotheses.
    - Section 3.2.1 (page 12): the object hypothesis heuristic.
    - Section 3.1.1 (page 8): object properties (colour, centre, shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa import similarity
from dense_armor.core.graph.vsa_objects import ObjectEncoder


class ObjectHypothesis(str, Enum):
    """The six object hypotheses of Table 3."""

    EIGHT_CONNECTED = "8-connected"
    FOUR_CONNECTED = "4-connected"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    COLOUR = "colour"
    PIXEL = "pixel"


@dataclass(frozen=True)
class VSAObject:
    """One object under one hypothesis: its cells, colour, and encoded vector."""

    cells: frozenset[tuple[int, int]]
    colour: int
    sp: np.ndarray  # the encoded object


def _bfs_components(
    grid: Grid, deltas: tuple[tuple[int, int], ...], same_colour: bool
) -> list[frozenset[tuple[int, int]]]:
    """BFS on the grid with the given neighbour pattern.

    Args:
        grid: the grid.
        deltas: the neighbour offsets.
        same_colour: if True, only cells of the same colour are connected.

    Returns:
        the connected components, as frozensets of (row, col).
    """
    h, w = grid.shape()
    seen: set[tuple[int, int]] = set()
    components: list[frozenset[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            colour = grid.at(r, c)
            comp: set[tuple[int, int]] = {(r, c)}
            frontier = [(r, c)]
            seen.add((r, c))
            while frontier:
                cr, cc = frontier.pop()
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if same_colour and grid.at(nr, nc) != colour:
                        continue
                    seen.add((nr, nc))
                    comp.add((nr, nc))
                    frontier.append((nr, nc))
            components.append(frozenset(comp))
    return components


def _connected_components_colour(
    grid: Grid, deltas: tuple[tuple[int, int], ...]
) -> list[frozenset[tuple[int, int]]]:
    """Connected components of the same colour with the given deltas."""
    return _bfs_components(grid, deltas, same_colour=True)


def _vertical_stripes(grid: Grid) -> list[frozenset[tuple[int, int]]]:
    """Each maximal vertical run of the same colour is an object."""
    h, w = grid.shape()
    components: list[frozenset[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for c in range(w):
        for r in range(h):
            if (r, c) in seen:
                continue
            colour = grid.at(r, c)
            comp: set[tuple[int, int]] = set()
            rr = r
            while rr < h and grid.at(rr, c) == colour:
                comp.add((rr, c))
                seen.add((rr, c))
                rr += 1
            components.append(frozenset(comp))
    return components


def _horizontal_stripes(grid: Grid) -> list[frozenset[tuple[int, int]]]:
    """Each maximal horizontal run of the same colour is an object."""
    h, w = grid.shape()
    components: list[frozenset[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            colour = grid.at(r, c)
            comp: set[tuple[int, int]] = set()
            cc = c
            while cc < w and grid.at(r, cc) == colour:
                comp.add((r, cc))
                seen.add((r, cc))
                cc += 1
            components.append(frozenset(comp))
    return components


def _colour_objects(grid: Grid) -> list[frozenset[tuple[int, int]]]:
    """All cells of the same colour form one object, regardless of position."""
    h, w = grid.shape()
    by_colour: dict[int, set[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            by_colour.setdefault(grid.at(r, c), set()).add((r, c))
    return [frozenset(s) for s in by_colour.values()]


def _pixel_objects(grid: Grid) -> list[frozenset[tuple[int, int]]]:
    """Each non-zero pixel is its own object."""
    h, w = grid.shape()
    return [
        frozenset({(r, c)}) for r in range(h) for c in range(w) if grid.at(r, c) != 0
    ]


HYPOTHESIS_BUILDERS = {
    ObjectHypothesis.EIGHT_CONNECTED: lambda g: _connected_components_colour(
        g, ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    ),
    ObjectHypothesis.FOUR_CONNECTED: lambda g: _connected_components_colour(
        g, ((-1, 0), (1, 0), (0, -1), (0, 1))
    ),
    ObjectHypothesis.VERTICAL: _vertical_stripes,
    ObjectHypothesis.HORIZONTAL: _horizontal_stripes,
    ObjectHypothesis.COLOUR: _colour_objects,
    ObjectHypothesis.PIXEL: _pixel_objects,
}


def perceive(
    grid: Grid,
    hypothesis: ObjectHypothesis,
    encoder: ObjectEncoder,
) -> list[VSAObject]:
    """Perceive the objects of ``grid`` under ``hypothesis``, with their VSA.

    Args:
        grid: the grid.
        hypothesis: which of the six object hypotheses to use.
        encoder: the VSA encoder.

    Returns:
        the list of VSAObject, one per component.
    """
    builder = HYPOTHESIS_BUILDERS[hypothesis]
    components = builder(grid)
    h, w = grid.shape()
    out: list[VSAObject] = []
    for cells in components:
        if not cells:
            continue
        # colour: the most frequent colour in the component (usually all the same)
        counts: dict[int, int] = {}
        for r, c in cells:
            v = grid.at(r, c)
            counts[v] = counts.get(v, 0) + 1
        colour = max(counts, key=lambda k: counts[k])
        sp = encoder.encode_object(list(cells), colour, h, w)
        out.append(VSAObject(cells=cells, colour=colour, sp=sp))
    return out


def cleanup_colour(colour_sp: np.ndarray, encoder: ObjectEncoder) -> int:
    """Find the colour whose SP is most similar to ``colour_sp``.

    Args:
        colour_sp: the (approximate) colour SP.
        encoder: the encoder holding the 10 colour SPs.

    Returns:
        the recovered ARC colour 0..9.
    """
    sims = [similarity(colour_sp, c) for c in encoder.colour_sps]
    return int(np.argmax(sims))


def cleanup_centre(centre_sp: np.ndarray, encoder: ObjectEncoder) -> tuple[int, int]:
    """Find the (x, y) whose SSP is most similar to ``centre_sp``.

    Args:
        centre_sp: the (approximate) centre SSP.
        encoder: the encoder holding the SSP samples.

    Returns:
        the recovered (x, y) centre, relative to the grid centre.
    """
    sims = encoder.centre_samples @ centre_sp
    idx = int(np.argmax(sims))
    pt = encoder.centre_index[idx]
    return (int(pt[0]), int(pt[1]))


def object_hypothesis_heuristic(
    grid_in: Grid,
    grid_out: Grid,
    encoder: ObjectEncoder,
) -> list[tuple[ObjectHypothesis, float]]:
    """Score each of the six hypotheses, after Section 3.2.1 (a variant).

    For every output object, compute the distribution of similarities to
    the input objects under this hypothesis, take the softmax, and record
    the maximum. The average maximum across output objects is the score.
    Higher is better: it means each output object is clearly similar to
    few input objects.

    Differences from the paper: the similarities are not padded with 0 and 1
    (with one input object the maximum is always 1), the softmax has a
    temperature of 10, the score is divided by ``1 + 0.01 * (n_in + n_out)``,
    and only the pair given is scored (``abduct`` passes the first pair).

    Args:
        grid_in: the input grid.
        grid_out: the output grid.
        encoder: the VSA encoder.

    Returns:
        the hypotheses sorted by score, descending.
    """
    scores: list[tuple[ObjectHypothesis, float]] = []
    for hyp in ObjectHypothesis:
        objs_in = perceive(grid_in, hyp, encoder)
        objs_out = perceive(grid_out, hyp, encoder)
        if not objs_in or not objs_out:
            scores.append((hyp, 0.0))
            continue
        maxima: list[float] = []
        for o_out in objs_out:
            sims = np.array([similarity(o_out.sp, o_in.sp) for o_in in objs_in])
            # softmax
            sims = sims - sims.max()
            p = np.exp(sims * 10.0)  # temperature 10, as a sharpening
            p = p / p.sum()
            maxima.append(float(p.max()))
        # penalize too many objects
        n_obj = len(objs_in) + len(objs_out)
        scores.append((hyp, float(np.mean(maxima)) / (1.0 + 0.01 * n_obj)))
    scores.sort(key=lambda t: -t[1])
    return scores
