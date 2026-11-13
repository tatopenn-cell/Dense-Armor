"""Action: a STRIPS action with preconditions and effects.

An action is a function from a graph (and parameters) to a graph,
with preconditions checked before the effects are applied.

References:
    Fikes, Nilsson (1971), "STRIPS: A New Approach to the Application
    of Theorem Proving to Problem Solving", Artificial Intelligence
    (bibliographic only).
    Vijay, Ethiraj (2025), arXiv:2512.20275, Section 3.2: an action is
    validated on the subgraph it operates on before it is executed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from dense_armor.core.graph.ontology import TypedGraph, type_ok


class ActionRejected(Exception):
    """Raised when an Action is refused.

    ``what`` says why: a failed precondition (the default), or an edge
    constraint broken before or after the action.
    """

    def __init__(
        self, action: str, failed: list[tuple], what: str = "precondition(s) failed"
    ) -> None:
        self.action = action
        self.failed = failed
        msg = f"action {action!r} rejected: {len(failed)} {what}"
        super().__init__(msg)


@dataclass
class Action:
    """A STRIPS action.

    Args:
        name: action name.
        preconditions: list of ``(node, field, op, value)``.
        effects: list of ``(node, field, kind, value)``, ``kind`` in
            ``{"set", "add", "delete"}``.
    """

    name: str
    preconditions: list[tuple] = field(default_factory=list)
    effects: list[tuple] = field(default_factory=list)

    def check(
        self, graph: TypedGraph, params: dict[str, Any] | None = None
    ) -> list[tuple]:
        """Return the preconditions that fail on ``graph``."""
        params = params or {}
        failed = []
        for node, field_name, op, value in self.preconditions:
            node_name = params.get(node, node) if node in params else node
            if node_name not in graph.graph:
                failed.append((node, field_name, op, value, "no such node"))
                continue
            actual = graph.graph.nodes[node_name]["state"].get(field_name)
            if not _cmp(actual, op, value):
                failed.append((node_name, field_name, op, value, actual))
        return failed

    def apply(
        self, graph: TypedGraph, params: dict[str, Any] | None = None
    ) -> TypedGraph:
        """Check preconditions, then apply effects; graph unchanged on failure."""
        params = params or {}
        failed = self.check(graph, params)
        if failed:
            raise ActionRejected(self.name, failed)
        new = TypedGraph(schema=graph.schema)
        new.graph = copy.deepcopy(graph.graph)
        for node, field_name, kind, value in self.effects:
            node_name = params.get(node, node) if node in params else node
            if node_name not in new.graph:
                raise ActionRejected(
                    self.name, [(node, field_name, kind, value, "no such node")]
                )
            typ = new.graph.nodes[node_name]["type"]
            expected = new.schema[typ].get(field_name)
            if expected is None:
                raise ValueError(f"field {field_name!r} not in schema for type {typ!r}")
            state = dict(new.graph.nodes[node_name]["state"])
            if kind == "set":
                if not type_ok(value, expected):
                    raise ValueError(
                        f"field {field_name!r} must be {expected.__name__}, "
                        f"got {type(value).__name__}"
                    )
                state[field_name] = value
            elif kind in ("add", "delete"):
                current = state.get(field_name, [])
                if not isinstance(current, list):
                    raise ValueError(
                        f"effect {kind!r} needs a list field, {field_name!r} "
                        f"holds {type(current).__name__}"
                    )
                lst = list(current)
                if kind == "add":
                    lst.append(value)
                elif value in lst:
                    lst.remove(value)
                state[field_name] = lst
            else:
                raise ValueError(f"unknown effect kind {kind!r}")
            new.graph.nodes[node_name]["state"] = state
        return new


def apply_action(
    graph: TypedGraph, action: Action, params: dict[str, Any] | None = None
) -> TypedGraph:
    """Apply ``action`` to ``graph``; refuse it if it breaks a rule that held.

    A large system always has some broken rules, so the graph is not
    required to be valid. The rules broken before the action are the
    baseline: the action is refused only if it breaks a rule that held.

    Only the relations of the nodes that the effects change are checked
    (the validation of the affected subgraph, as in arXiv:2512.20275,
    Section 3.2): a rule reads the states of its two ends, so no other
    relation can change status, and the result is the one of checking the
    whole graph.
    """
    params = params or {}
    touched = {params.get(node, node) for node, *_ in action.effects}
    before = set(graph.validate(nodes=touched))
    new = action.apply(graph, params)
    introduced = [v for v in new.validate(nodes=touched) if v not in before]
    if introduced:
        raise ActionRejected(
            action.name, introduced, "new constraint(s) broken by the action"
        )
    return new


def _cmp(actual: Any, op: str, value: Any) -> bool:
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "<":
        return actual is not None and actual < value
    if op == ">":
        return actual is not None and actual > value
    if op == "<=":
        return actual is not None and actual <= value
    if op == ">=":
        return actual is not None and actual >= value
    raise ValueError(f"unknown op {op!r}")
