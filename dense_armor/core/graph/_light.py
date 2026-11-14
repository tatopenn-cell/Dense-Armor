"""A small directed multigraph: the part of networkx that TypedGraph and ExecutionDAG use.

The ontology and the execution DAG need nodes with attributes, edges with attributes, the order of
a topological sort and the ancestors of a node. That is a few dozen lines, so they do not import
networkx (importing it takes about 0.6 s). ``to_networkx`` builds a networkx graph, and only then
networkx is imported.

The order of ``topological_order`` is the one of ``networkx.topological_sort``: the nodes of in-degree
zero in the order they were added, then the nodes they free, generation by generation.

References:
    Kahn (1962), "Topological sorting of large networks", CACM (bibliographic only).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from typing import Any


class _Nodes(dict):
    """``graph.nodes[name]`` is the attribute dict; ``graph.nodes(data=True)`` lists ``(name, attrs)``."""

    def __call__(self, data: bool = False):
        return list(self.items()) if data else list(self)


class LightGraph:
    """Nodes with attributes and directed edges with attributes; parallel edges get their own key."""

    def __init__(self) -> None:
        self.nodes: _Nodes = _Nodes()
        self._out: dict[Any, list[tuple[Any, int, dict]]] = {}
        self._in: dict[Any, list[tuple[Any, int, dict]]] = {}
        self._next_key: dict[tuple[Any, Any], int] = {}

    def __contains__(self, name: Any) -> bool:
        return name in self.nodes

    def __iter__(self) -> Iterator:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return sum(len(v) for v in self._out.values())

    def add_node(self, name: Any, **attrs: Any) -> None:
        if name not in self.nodes:
            self.nodes[name] = {}
            self._out[name] = []
            self._in[name] = []
        self.nodes[name].update(attrs)

    def add_nodes_from(self, names: Iterable) -> None:
        for name in names:
            self.add_node(name)

    def add_edge(self, u: Any, v: Any, key: int | None = None, **attrs: Any) -> int:
        self.add_node(u)
        self.add_node(v)
        if key is None:
            key = self._next_key.get((u, v), 0)
        self._next_key[(u, v)] = max(self._next_key.get((u, v), 0), key + 1)
        data = dict(attrs)
        self._out[u].append((v, key, data))
        self._in[v].append((u, key, data))
        return key

    @staticmethod
    def _shape(u, v, key, data, keys, want):
        row = (u, v, key) if keys else (u, v)
        if want is True:
            return (*row, data)
        if want:
            return (*row, data.get(want))
        return row

    def edges(self, keys: bool = False, data: bool | str = False) -> list[tuple]:
        return [
            self._shape(u, v, key, d, keys, data)
            for u in self.nodes
            for v, key, d in self._out[u]
        ]

    def out_edges(self, name: Any, keys: bool = False, data: bool | str = False) -> list[tuple]:
        return [self._shape(name, v, key, d, keys, data) for v, key, d in self._out.get(name, [])]

    def in_edges(self, name: Any, keys: bool = False, data: bool | str = False) -> list[tuple]:
        return [self._shape(u, name, key, d, keys, data) for u, key, d in self._in.get(name, [])]

    def successors(self, name: Any) -> list:
        return list(dict.fromkeys(v for v, _k, _d in self._out.get(name, [])))

    def predecessors(self, name: Any) -> list:
        return list(dict.fromkeys(u for u, _k, _d in self._in.get(name, [])))

    def in_degree(self, name: Any) -> int:
        return len(self._in.get(name, []))

    def out_degree(self, name: Any) -> int:
        return len(self._out.get(name, []))

    def copy(self) -> LightGraph:
        return copy.deepcopy(self)

    def topological_order(self) -> list:
        """The order of ``networkx.topological_sort``. Raises ``ValueError`` on a cycle."""
        indegree = {v: self.in_degree(v) for v in self.nodes if self.in_degree(v) > 0}
        zero = [v for v in self.nodes if self.in_degree(v) == 0]
        order: list = []
        while zero:
            generation, zero = zero, []
            for node in generation:
                for child in self.successors(node):
                    indegree[child] -= sum(1 for v, _k, _d in self._out[node] if v == child)
                    if indegree[child] == 0:
                        zero.append(child)
                        del indegree[child]
            order.extend(generation)
        if indegree:
            raise ValueError("graph has a cycle")
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
        except ValueError:
            return False
        return True

    def _reach(self, name: Any, step) -> set:
        seen: set = set()
        stack = [name]
        while stack:
            for nxt in step(stack.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def ancestors(self, name: Any) -> set:
        return self._reach(name, self.predecessors)

    def descendants(self, name: Any) -> set:
        return self._reach(name, self.successors)

    def to_networkx(self, directed_simple: bool = False):
        """The same graph as a networkx ``MultiDiGraph`` (or ``DiGraph``). Imports networkx."""
        import networkx as nx  # type: ignore[import-untyped]

        g = nx.DiGraph() if directed_simple else nx.MultiDiGraph()
        for name, attrs in self.nodes.items():
            g.add_node(name, **copy.deepcopy(attrs))
        for u in self.nodes:
            for v, key, data in self._out[u]:
                if directed_simple:
                    g.add_edge(u, v, **data)
                else:
                    g.add_edge(u, v, key=key, **data)
        return g
