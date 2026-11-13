"""ExecutionDAG: a deterministic network of estimators.

Every node is an estimator, every edge is a ``Signal`` flow. The
execution order is the topological order.

References:
    Cheng, Wu, Hu (2024), "Dataflow-Guided Retrieval Augmentation for
    Repository-Level Code Completion", arXiv:2405.19782
    (bibliographic only: no PDF in ``papers/``, no section verified; the paper is about code-completion
    retrieval, not estimator networks).
    Kahn (1962), "Topological sorting of large networks", CACM
    (bibliographic only).
"""

from __future__ import annotations

from typing import Any

from dense_armor.core.graph.provenance import Provenance
from dense_armor.roles import Signal


def _nx():
    import networkx as nx  # type: ignore[import-untyped]

    return nx


class ExecutionDAG:
    """A DAG of estimators.

    Args:
        nodes: dict ``{name: estimator}``.
        edges: list of ``(src, dst)``, every name must be in ``nodes``.

    Raises:
        ValueError: on an unknown node in an edge, or a cycle.

    Choices stated explicitly:
        Each node has at most one predecessor (a tree, not a general
        DAG): the execution is a linear pass over the topological
        order, and a node with two predecessors is refused at run
        time. This is a design choice of this module: a general dataflow graph
        with joins (Naiad-style differential dataflow) is a different
        design, out of scope here. The provenance is recorded in the
        same pass, so the log order is the topological order.
    """

    def __init__(
        self,
        nodes: dict[str, Any],
        edges: list[tuple[str, str]],
        provenance: Provenance | None = None,
    ) -> None:
        self._nodes = dict(nodes)
        self._edges = list(dict.fromkeys((s, d) for s, d in edges))
        self._provenance = provenance if provenance is not None else Provenance()
        self._t = 0.0
        self._t_step = 1.0
        self._g = _nx().DiGraph()
        self._g.add_nodes_from(self._nodes.keys())
        for src, dst in self._edges:
            if src not in self._nodes:
                raise ValueError(f"unknown node in edge: {src}")
            if dst not in self._nodes:
                raise ValueError(f"unknown node in edge: {dst}")
            self._g.add_edge(src, dst)
        if not _nx().is_directed_acyclic_graph(self._g):
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
        return _nx().is_directed_acyclic_graph(self._g)

    def topological_order(self) -> list[str]:
        return list(_nx().topological_sort(self._g))

    def ancestors(self, name: str) -> list[str]:
        return sorted(_nx().ancestors(self._g, name))

    def descendants(self, name: str) -> list[str]:
        return sorted(_nx().descendants(self._g, name))

    def to_networkx(self):
        return self._g.copy()

    def run(
        self,
        signals: dict[str, Signal],
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Signal | None]:
        """Run the DAG on ``signals``, one per source node.

        Every node receives the signal of its single predecessor and
        produces one output. A node with no predecessor uses
        ``signals[name]``. A node with more than one predecessor is
        refused at run time. If the predecessor is a ``GateNode`` that
        returned ``None``, the node is skipped (``out[name] = None``).

        ``labels`` maps a node name to the true label of this sample. A
        regressor or classifier node predicts first and learns after,
        only if it has a label in ``labels``.
        """
        labels = labels or {}
        order = self.topological_order()
        for name in order:
            n_pred = self._g.in_degree(name)
            if n_pred > 1:
                raise ValueError(
                    f"node {name} has {n_pred} predecessors, expected at most 1"
                )
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
            out[name] = self._apply(name, est, x, labels.get(name))
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
    def _apply(name: str, est: Any, x: Signal, y: Any = None) -> Signal | None:
        # GateNode: scores, learns only the samples that pass
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
        # Regressor / Classifier: predict_one, then learn_one if there is a label
        if hasattr(est, "predict_one"):
            pred = est.predict_one(x)
            if y is not None and hasattr(est, "learn_one"):
                est.learn_one(x, y)
            if isinstance(pred, dict):
                return Signal(
                    list(pred.values()),
                    names=[str(k) for k in pred.keys()],
                    units=[""] * len(pred),
                )
            return Signal(
                [float("nan") if pred is None else float(pred)],
                names=[name],
                units=["pred"],
            )
        return x
