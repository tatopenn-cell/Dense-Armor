"""GraphSignal: a :class:`Signal` whose values live on the graph nodes."""

from __future__ import annotations

import numpy as np

from dense_armor.roles import Signal


class GraphSignal:
    """A :class:`Signal` (one channel per node) with a :class:`Graph`.

    Args:
        signal: a :class:`Signal`, one channel per node.
        graph: a :class:`Graph` with ``len(signal.names)`` nodes.

    Raises:
        ValueError: if the number of channels differs from ``n_nodes``.
    """

    def __init__(self, signal: Signal, graph) -> None:
        if len(signal.names) != graph.n_nodes:
            raise ValueError(
                f"signal has {len(signal.names)} channels, "
                f"graph has {graph.n_nodes} nodes"
            )
        self._signal = signal
        self._graph = graph
        # valori come array float64, nell'ordine dei names
        self.values = np.array([signal[n] for n in signal.names], dtype=float)

    @classmethod
    def from_signal(cls, signal: Signal, graph) -> GraphSignal:
        return cls(signal, graph)

    @property
    def n_nodes(self) -> int:
        return self._graph.n_nodes

    def to_signal(self) -> Signal:
        """Return the original :class:`Signal`."""
        return self._signal
