"""Structural change between two graphs.

References:
    Jenkins, Variddhisai, El-Medany, Ng, Mandic (2024), "Online Graph
    Topology Learning via Time-Vertex Adaptive Filters: From Theory to
    Cardiac Fibrillation", arXiv:2411.01567, Section VI: the paper
    follows the changes of the weights and of the topology of an
    estimated graph over time. The two measures below are a choice of
    this module; the paper plots those changes and does not define these
    functions.
"""

from __future__ import annotations

import math

from dense_armor.core.graph.core import Graph


def _weights(graph: Graph) -> dict[tuple[int, int], float]:
    return {(min(i, j), max(i, j)): w for i, j, w in graph.edges}


def structural_change(before: Graph, after: Graph, tol: float = 0.0) -> dict:
    """Compare two graphs on the same nodes.

    Args:
        before: the graph at the first time.
        after: the graph at the second time, with the same ``n_nodes``.
        tol: a weight that changes by no more than ``tol`` is not
            reported as reweighted.

    Returns:
        A dict with:

        * ``added``: edges ``(i, j)`` that are in ``after`` and not in
          ``before``, sorted;
        * ``removed``: edges that are in ``before`` and not in ``after``;
        * ``reweighted``: ``(i, j, w_before, w_after)`` for the edges
          in both whose weight changed by more than ``tol``;
        * ``topology_change``: the Jaccard distance of the two edge sets,
          ``1 - |A and B| / |A or B|``: 0 for the same edges, 1 for
          edges with nothing in common, 0 when neither graph has an edge;
        * ``weight_change``: ``sqrt(sum (w_after - w_before)^2)`` over
          all the edges of either graph (a missing edge weighs 0),
          divided by the larger of the two weight norms; 0 for equal
          graphs and 0 when neither graph has an edge.

    Raises:
        ValueError: if the graphs have a different number of nodes or
            ``tol`` is negative.

    Example:
        >>> a = Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
        >>> b = Graph(3, [(0, 1, 1.0)])
        >>> structural_change(a, b)["removed"]
        [(1, 2)]
    """
    if before.n_nodes != after.n_nodes:
        raise ValueError(
            f"graphs have {before.n_nodes} and {after.n_nodes} nodes, expected the same"
        )
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    wb, wa = _weights(before), _weights(after)
    keys = set(wb) | set(wa)
    common = set(wb) & set(wa)
    reweighted = [
        (i, j, wb[(i, j)], wa[(i, j)])
        for i, j in sorted(common)
        if abs(wa[(i, j)] - wb[(i, j)]) > tol
    ]
    diff = math.sqrt(sum((wa.get(k, 0.0) - wb.get(k, 0.0)) ** 2 for k in keys))
    den = max(
        math.sqrt(sum(w * w for w in wb.values())),
        math.sqrt(sum(w * w for w in wa.values())),
    )
    return {
        "added": sorted(set(wa) - set(wb)),
        "removed": sorted(set(wb) - set(wa)),
        "reweighted": reweighted,
        "topology_change": 1.0 - len(common) / len(keys) if keys else 0.0,
        "weight_change": diff / den if den > 0 else 0.0,
    }
