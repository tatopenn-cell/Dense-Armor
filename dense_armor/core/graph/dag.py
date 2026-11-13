"""ExecutionDAG: a deterministic network of estimators.

Every node is an estimator, every edge is a ``Signal`` flow. The
execution order is the topological order.

References:
    Cheng, Wu, Hu (2024), "Dataflow-Guided Retrieval Augmentation for
    Repository-Level Code Completion", arXiv:2405.19782, Section 3.2
    (the repo-specific context graph as a DAG of dependencies).
    Kahn (1962), "Topological sorting of large networks", CACM
    (bibliographic only).
"""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from dense_armor.core.graph.provenance import Provenance
from dense_armor.roles import Signal


class ExecutionDAG:
    """A DAG of estimators.

    Args:
        nodes: dict ``{name: estimator}``.
        edges: list of ``(src, dst)``, every name must be in ``nodes``.

    Raises:
        ValueError: on duplicate node, unknown node in an edge, cycle.
    """

    def __init__(
        self,
        nodes: dict[str, Any],
        edges: list[tuple[str, str]],
        provenance: Provenance | None = None,
    ) -> None:
        self._nodes = dict(nodes)
        self._edges = list(edges)
        self._provenance = provenance if provenance is not None else Provenance()
        self._t = 0.0
        self._t_step = 1.0
        self._g = nx.DiGraph()
        self._g.add_nodes_from(self._nodes.keys())
        for src, dst in self._edges:
            if src not in self._nodes:
                raise ValueError(f"unknown node in edge: {src}")
            if dst not in self._nodes:
                raise ValueError(f"unknown node in edge: {dst}")
            self._g.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(self._g):
            raise ValueError("graph has a cycle")

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    @property
    def nodes(self) -> dict[str, Any]:
        return self._nodes

    @property
    def edges(self) -> list[tuple[str, str]]:
        return self._edges

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._g))

    def ancestors(self, name: str) -> list[str]:
        return sorted(nx.ancestors(self._g, name))

    def descendants(self, name: str) -> list[str]:
        return sorted(nx.descendants(self._g, name))

    def to_networkx(self):
        return self._g.copy()

    def run(self, signals: dict[str, Signal]) -> dict[str, Signal | None]:
        """Run the DAG on ``signals``, one per source node.

        Every node receives the signal of its single predecessor and
        produces one output. A node with no predecessor uses
        ``signals[name]``. A node with more than one predecessor is
        refused at run time. If the predecessor is a ``GateNode`` that
        returned ``None``, the node is skipped (``out[name] = None``).
        """
        order = self.topological_order()
        out: dict[str, Signal | None] = {}
        for name in order:
            preds = list(self._g.predecessors(name))
            if not preds:
                if name not in signals:
                    raise ValueError(f"no signal for source {name}")
                x = signals[name]
            else:
                if len(preds) > 1:
                    raise ValueError(
                        f"node {name} has {len(preds)} predecessors, expected at most 1"
                    )
                x = out.get(preds[0])
                if x is None:
                    out[name] = None
                    continue
            est = self._nodes[name]
            out[name] = self._apply(name, est, x)
            self._provenance.record(
                t=self._t,
                node=name,
                method="run",
                signal_in=x,
                output=out[name],
                parents=preds,
            )
            self._t += self._t_step
        return out

    @staticmethod
    def _apply(name: str, est: Any, x: Signal) -> Signal | None:
        # GateNode: non chiama learn_one del detector (protect=False)
        if hasattr(est, "blocked") and callable(est):
            return est(x)
        # AnomalyDetector: score_one
        if hasattr(est, "score_one") and not hasattr(est, "predict_one"):
            score = float(est.score_one(x))
            if hasattr(est, "learn_one"):
                est.learn_one(x)
            return Signal([score], names=[name], units=["score"])
        # Transformer: learn_one + transform_one
        if hasattr(est, "transform_one"):
            if hasattr(est, "learn_one"):
                est.learn_one(x)
            trans = est.transform_one(x)
            return Signal(
                list(trans.values()),
                names=[str(k) for k in trans.keys()],
                units=[""] * len(trans),
            )
        # Regressor / Classifier: learn_one + predict_one
        if hasattr(est, "predict_one"):
            if hasattr(est, "learn_one"):
                est.learn_one(x, None)
            pred = est.predict_one(x)
            if isinstance(pred, dict):
                return Signal(
                    list(pred.values()),
                    names=[str(k) for k in pred.keys()],
                    units=[""] * len(pred),
                )
            return Signal([float(pred)], names=[name], units=["pred"])
        return x
