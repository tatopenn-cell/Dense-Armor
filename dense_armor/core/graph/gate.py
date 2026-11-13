"""GateNode: an AnomalyGate as a node of an ExecutionDAG."""

from __future__ import annotations

from typing import Any

from dense_armor.roles import AnomalyGate
from dense_armor.roles._util import sum_budgets, worst_memory_class


class GateNode:
    """A node that blocks the downstream flow when the detector fires.

    Args:
        detector: an anomaly detector with ``score_one``.
        threshold: optional, otherwise the detector's own threshold.

    Choices stated explicitly:
        ``GateNode`` classifies with an ``AnomalyGate`` and scores every
        sample once: the detector learns from the samples that pass and
        does not learn from the blocked ones, which are only scored.
        This is a design choice of this module.
    """

    def __init__(self, detector: Any, threshold: float | None = None) -> None:
        self._gate = AnomalyGate(detector, protect=True, threshold=threshold)
        self.blocked: int = 0
        self.passed: int = 0

    @property
    def budget_s(self) -> float | None:
        """Latency of one call: a ``score_one`` and a ``learn_one`` of the detector."""
        return sum_budgets([(2.0, self._gate.detector)])

    @property
    def memory_class(self) -> str:
        """The memory class of the detector."""
        return worst_memory_class([self._gate.detector])

    def __call__(self, signal) -> Any | None:
        score = self._gate.score_one(signal)
        blocked = self._gate.classify(score)
        if blocked:
            self.blocked += 1
            return None
        self._gate.detector.learn_one(signal)
        self.passed += 1
        return signal
