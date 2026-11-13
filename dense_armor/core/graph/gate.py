"""GateNode: an AnomalyGate as a node of an ExecutionDAG."""

from __future__ import annotations

from typing import Any

from dense_armor.roles import AnomalyGate


class GateNode:
    """A node that blocks the downstream flow when the detector fires.

    Args:
        detector: an anomaly detector with ``score_one``.
        threshold: optional, otherwise the detector's own threshold.
    """

    def __init__(self, detector: Any, threshold: float | None = None) -> None:
        self._gate = AnomalyGate(detector, protect=False, threshold=threshold)
        self.blocked: int = 0
        self.passed: int = 0

    def __call__(self, signal) -> Any | None:
        score = self._gate.score_one(signal)
        if self._gate.classify(score):
            self.blocked += 1
            return None
        self.passed += 1
        return signal
