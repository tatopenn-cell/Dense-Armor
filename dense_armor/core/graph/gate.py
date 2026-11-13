"""GateNode: an AnomalyGate as a node of an ExecutionDAG."""

from __future__ import annotations

from typing import Any

from dense_armor.roles import AnomalyGate


class GateNode:
    """A node that blocks the downstream flow when the detector fires.

    Args:
        detector: an anomaly detector with ``score_one``.
        threshold: optional, otherwise the detector's own threshold.

    Choices stated explicitly:
        ``GateNode`` wraps ``AnomalyGate(protect=True)`` and calls its
        ``learn_one`` on every sample: the detector learns from the
        samples that pass and does not learn from the blocked ones,
        which are only scored. This is a design choice of this module.
    """

    def __init__(self, detector: Any, threshold: float | None = None) -> None:
        self._gate = AnomalyGate(detector, protect=True, threshold=threshold)
        self.blocked: int = 0
        self.passed: int = 0

    def __call__(self, signal) -> Any | None:
        score = self._gate.score_one(signal)
        blocked = self._gate.classify(score)
        self._gate.learn_one(signal)
        if blocked:
            self.blocked += 1
            return None
        self.passed += 1
        return signal
