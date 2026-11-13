"""CausalGraph: a DAG with the DoWhy do-calculus.

References:
    Zan et al. (2026), "EvoCause: LLM-Guided Evolution of Causal
    Graphs for Root Cause Analysis", arXiv:2607.27290, Section 3.1
    "Problem Setup" and 3.2 "Identifiability under Root-Cause
    Feedback" (causal graph and intervention).
    Pearl (2009), "Causality: Models, Reasoning, and Inference",
    Cambridge University Press (bibliographic only).
    DoWhy 0.14 (MIT): CausalModel, identify_effect,
    estimate_effect(method_name='backdoor.linear_regression').
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]


class CausalGraph:
    """A DAG of causal relations.

    Args:
        nodes: list of variable names.
        edges: list of ``(cause, effect)``.

    Raises:
        ValueError: on an unknown node in an edge, or a cycle.

    Choices stated explicitly:
        The effect is estimated with DoWhy's
        ``backdoor.linear_regression`` on the parents of the treatment
        as ``common_causes``. This is a choice of the kit: DoWhy also
        offers instrumental variables, front-door, and propensity
        scores, and the paper (EvoCause) describes a different
        estimator (LLM-guided causal graph evolution). The kit picks
        the simplest DoWhy estimator on purpose, so that the effect
        is identifiable and testable on a known DAG.
    """

    def __init__(self, nodes: list[str], edges: list[tuple[str, str]]) -> None:
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._g = nx.DiGraph()
        self._g.add_nodes_from(self._nodes)
        for u, v in self._edges:
            if u not in self._g or v not in self._g:
                raise ValueError(f"unknown node in edge ({u}, {v})")
            self._g.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(self._g):
            raise ValueError("causal graph has a cycle")

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self._edges)

    def parents(self, name: str) -> list[str]:
        return sorted(self._g.predecessors(name))

    def is_valid(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def estimate_effect(
        self, treatment: str, value: float, data, outcome: str
    ) -> float:
        """Estimate the causal effect of ``treatment = value`` on
        ``outcome`` using DoWhy (backdoor.linear_regression).

        Args:
            treatment: name of the treatment variable.
            value: the value to which the treatment is set.
            data: a ``pandas.DataFrame`` with all the variables.
            outcome: name of the outcome variable.

        Returns:
            The estimated average causal effect.
        """
        from dowhy import CausalModel  # type: ignore[import-untyped]

        if treatment not in self._g or outcome not in self._g:
            raise ValueError(f"unknown variable ({treatment}, {outcome})")
        common = [p for p in self.parents(treatment) if p in data.columns]
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common if common else None,
        )
        identified = model.identify_effect()
        est = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        return float(est.value)
