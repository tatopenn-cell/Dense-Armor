"""TypedGraph: a graph with node types, schema and constraints.

A node has a ``type`` and a ``state`` (a dict). An edge has a
``relation`` and an optional ``constraint`` (a function
``(state_src, state_dst) -> bool``).

References:
    Babaei Giglou, D'Souza, Auer (2023), "LLMs4OL: Large Language
    Models for Ontology Learning", arXiv:2307.16648, Section 2 "The
    LLMs4OL Task Paradigm" and the "Types" subsection (schema of the
    ontology).
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import networkx as nx  # type: ignore[import-untyped]


class TypedGraph:
    """A typed graph with a schema and constraints.

    Args:
        schema: dict ``{type_name: {field_name: type}}``.

    Raises:
        ValueError: on an unknown type, a field not in the schema, or
            a field value of the wrong type.

    Choices stated explicitly:
        The schema is a plain ``dict[str, dict[str, type]]``; the
        edges carry a ``relation`` and an optional ``constraint``
        function. This is a choice of the kit: the paper (LLMs4OL)
        describes ontology *learning*, not a typed graph API, and a
        full OWL/SHACL implementation would be a different choice
        (heavier, with its own parser).
    """

    def __init__(self, schema: dict[str, dict[str, type]]) -> None:
        self.schema = {k: dict(v) for k, v in schema.items()}
        self.graph = nx.MultiDiGraph()
        self._relations: set[str] = set()

    def add_node(self, name: str, node_type: str, **state: Any) -> TypedGraph:
        if node_type not in self.schema:
            raise ValueError(f"unknown type {node_type!r}")
        expected = self.schema[node_type]
        for field, value in state.items():
            if field not in expected:
                raise ValueError(
                    f"field {field!r} not in schema for type {node_type!r}"
                )
            if not isinstance(value, expected[field]):
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
        constraint: Callable[[dict, dict], bool] | None = None,
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
            if not isinstance(value, self.schema[typ][field]):
                raise ValueError(f"field {field!r} has wrong type")
            current[field] = value
        node["state"] = current
        return self

    def validate(self) -> list[tuple]:
        """Check every edge constraint. Return the failed ones."""
        failures: list[tuple] = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            fn = data.get("constraint")
            if fn is None:
                continue
            if not fn(self.graph.nodes[u]["state"], self.graph.nodes[v]["state"]):
                failures.append((u, v, data.get("relation", "?")))
        return failures

    def to_networkx(self):
        return self.graph.copy()
