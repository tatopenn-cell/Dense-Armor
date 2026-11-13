"""Dataflow: describe where each signal comes from, then build the DAG.

References:
    Kahn (1962), "Topological sorting of large networks", CACM
    (bibliographic only).
"""

from __future__ import annotations

from typing import Any

from dense_armor.core.graph.dag import ExecutionDAG
from dense_armor.core.graph.provenance import Provenance


class Dataflow:
    """A description of estimators and of the flow of signals between them.

    A source takes a signal given to ``ExecutionDAG.run``; every other
    node takes the output of the node it comes ``after``. ``build()``
    gives the :class:`ExecutionDAG` that runs it, and ``lineage(name)``
    says through which nodes a signal reached ``name``.

    Raises:
        ValueError: on a name used twice, or on a node that comes after
            a name that was not declared before it.

    Example:
        >>> class E:
        ...     pass
        >>> d = Dataflow().source("raw", E()).then("clean", E(), after="raw")
        >>> d.lineage("clean")
        ['raw', 'clean']
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._after: dict[str, str | None] = {}

    def source(self, name: str, estimator: Any) -> Dataflow:
        """Declare a node that receives a signal from outside."""
        self._declare(name, estimator, None)
        return self

    def then(self, name: str, estimator: Any, after: str) -> Dataflow:
        """Declare a node that receives the output of ``after``."""
        if after not in self._nodes:
            raise ValueError(f"unknown node {after!r}: declare it before {name!r}")
        self._declare(name, estimator, after)
        return self

    def _declare(self, name: str, estimator: Any, after: str | None) -> None:
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        self._nodes[name] = estimator
        self._after[name] = after

    def lineage(self, name: str) -> list[str]:
        """The nodes from the source down to ``name``, in order."""
        if name not in self._nodes:
            raise ValueError(f"unknown node {name!r}")
        chain = [name]
        while self._after[chain[-1]] is not None:
            chain.append(self._after[chain[-1]])  # type: ignore[arg-type]
        return chain[::-1]

    def edges(self) -> list[tuple[str, str]]:
        """The flows as ``(from, to)``, in the order they were declared."""
        return [(a, n) for n, a in self._after.items() if a is not None]

    def build(self, provenance: Provenance | None = None) -> ExecutionDAG:
        """The :class:`ExecutionDAG` of this flow."""
        return ExecutionDAG(self._nodes, self.edges(), provenance=provenance)
