"""Topological order vs Kahn written by hand."""

from collections import deque

import pytest

from dense_armor.core.graph import ExecutionDAG


class _Pass:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return dict(x) if isinstance(x, dict) else {"x": float(x)}


def _kahn(nodes, edges):
    g = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    for s, d in edges:
        g[s].append(d)
        indeg[d] += 1
    q = deque(sorted(n for n in nodes if indeg[n] == 0))
    out = []
    while q:
        n = q.popleft()
        out.append(n)
        for m in sorted(g[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return out


@pytest.mark.parametrize(
    "nodes,edges",
    [
        (["a", "b", "c"], [("a", "b"), ("b", "c")]),
        (["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]),
        (["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("a", "d")]),
    ],
)
def test_kahn_matches(nodes, edges):
    dag = ExecutionDAG(nodes={n: _Pass() for n in nodes}, edges=edges)
    ours = dag.topological_order()
    ref = _kahn(nodes, edges)
    # every edge goes from earlier to later
    for s, d in edges:
        assert ours.index(s) < ours.index(d)
    # sorted by name, the two lists are equal
    assert sorted(ours) == sorted(ref)
