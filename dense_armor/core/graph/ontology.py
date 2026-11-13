"""TypedGraph: a graph with node types, schema and constraints.

A node has a ``type`` and a ``state`` (a dict). An edge has a
``relation`` and an optional ``constraint`` (a function
``(state_src, state_dst) -> bool``).

References:
    Vijay, Ethiraj (2025), "Graph-Symbolic Policy Enforcement and Control
    (G-SPEC): A Neuro-Symbolic Framework for Safe Agentic AI in 5G
    Autonomous Networks", arXiv:2512.20275, Sections 1-3: a knowledge graph
    used as an executable state machine, where an action is allowed only
    if it operates on a valid subgraph and satisfies the policy; the
    validation cost follows the size of the subgraph. Introduction and
    problem formulation read; the evaluation was not.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any


def _nx():
    import networkx as nx  # type: ignore[import-untyped]

    return nx


class ConstraintViolation(ValueError):
    """Raised by ``TypedGraph.validate(raise_on_violation=True)``.

    ``failed`` lists the broken relations as ``(src, dst, relation)``.
    """

    def __init__(self, failed: list[tuple]) -> None:
        self.failed = failed
        super().__init__(f"{len(failed)} constraint(s) violated: {failed[:3]}")


class Constraint:
    """A named rule on the two ends of an edge.

    Args:
        name: what the rule says, used in messages.
        check: ``check(state_src, state_dst) -> bool``, ``False`` when the
            relation is broken.

    Example:
        >>> c = Constraint("no test db in production", lambda s, d: not (s["tier"] == "production" and d["env"] == "test"))
        >>> c({"tier": "production"}, {"env": "test"})
        False
    """

    def __init__(self, name: str, check: Callable[[dict, dict], bool]) -> None:
        self.name = name
        self.check = check

    def __call__(self, src: dict, dst: dict) -> bool:
        return bool(self.check(src, dst))

    def __repr__(self) -> str:
        return f"Constraint({self.name!r})"


def type_ok(value: Any, expected: type) -> bool:
    """``isinstance`` with the two numeric rules of the schema.

    An ``int`` is a valid ``float``, and a ``bool`` is neither an
    ``int`` nor a ``float``.
    """
    if isinstance(value, bool):
        return expected is bool or expected is object
    if expected is float:
        return isinstance(value, (int, float))
    return isinstance(value, expected)


class TypedGraph:
    """A typed graph with a schema and constraints.

    Args:
        schema: dict ``{type_name: {field_name: type}}``.

    Raises:
        ValueError: on an unknown type, a node name already used, a field
            not in the schema, or a field value of the wrong type (an
            ``int`` is accepted for a ``float`` field; a ``bool`` only
            for a ``bool`` field).

    Choices stated explicitly:
        The schema is a plain ``dict[str, dict[str, type]]``; the
        edges carry a ``relation`` and an optional ``constraint``
        function. This is a design choice of this module: a full OWL/SHACL
        implementation would be a different choice (heavier, with its own
        parser).
    """

    def __init__(self, schema: dict[str, dict[str, type]]) -> None:
        self.schema = {k: dict(v) for k, v in schema.items()}
        self.graph = _nx().MultiDiGraph()
        self._relations: set[str] = set()

    def add_node(self, name: str, node_type: str, **state: Any) -> TypedGraph:
        if node_type not in self.schema:
            raise ValueError(f"unknown type {node_type!r}")
        if name in self.graph:
            raise ValueError(f"node {name!r} already exists")
        expected = self.schema[node_type]
        for field, value in state.items():
            if field not in expected:
                raise ValueError(
                    f"field {field!r} not in schema for type {node_type!r}"
                )
            if not type_ok(value, expected[field]):
                raise ValueError(
                    f"field {field!r} must be {expected[field].__name__}, "
                    f"got {builtins.type(value).__name__}"
                )
        self.graph.add_node(name, type=node_type, state=dict(state))
        return self

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        constraint: Constraint | Callable[[dict, dict], bool] | None = None,
    ) -> TypedGraph:
        if src not in self.graph or dst not in self.graph:
            raise ValueError(f"unknown node in edge ({src}, {dst})")
        self.graph.add_edge(src, dst, relation=relation, constraint=constraint)
        self._relations.add(relation)
        return self

    def update(self, name: str, **state: Any) -> TypedGraph:
        if name not in self.graph:
            raise ValueError(f"unknown node {name!r}")
        node = self.graph.nodes[name]
        typ = node["type"]
        current = dict(node["state"])
        for field, value in state.items():
            if field not in self.schema[typ]:
                raise ValueError(f"field {field!r} not in schema")
            if not type_ok(value, self.schema[typ][field]):
                raise ValueError(
                    f"field {field!r} must be {self.schema[typ][field].__name__}, "
                    f"got {builtins.type(value).__name__}"
                )
            current[field] = value
        node["state"] = current
        return self

    def validate(self, raise_on_violation: bool = False, nodes=None) -> list[tuple]:
        """Check the edge constraints. Return the failed ones.

        Args:
            raise_on_violation: raise :class:`ConstraintViolation` if any
                constraint is violated, instead of returning the list.
            nodes: check only the edges that touch one of these nodes. A
                constraint reads the states of its two ends, so after a change
                of some nodes only their edges can change status: the cost
                follows the size of that subgraph, not of the whole graph.
        """
        if nodes is None:
            edges = self.graph.edges(keys=True, data=True)
        else:
            seen: dict = {}
            for n in nodes:
                if n not in self.graph:
                    continue
                for u, v, key, data in self.graph.in_edges(n, keys=True, data=True):
                    seen[(u, v, key)] = data
                for u, v, key, data in self.graph.out_edges(n, keys=True, data=True):
                    seen[(u, v, key)] = data
            edges = [(u, v, key, data) for (u, v, key), data in seen.items()]
        failures: list[tuple] = []
        for u, v, key, data in edges:
            fn = data.get("constraint")
            if fn is None:
                continue
            if not fn(self.graph.nodes[u]["state"], self.graph.nodes[v]["state"]):
                failures.append((u, v, data.get("relation", "?")))
        if failures and raise_on_violation:
            raise ConstraintViolation(failures)
        return failures

    def to_networkx(self):
        return self.graph.copy()
