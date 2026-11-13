"""Provenance: a JSONL log of every learn / predict on a DAG node.

Each record is ``{t, node, method, input_hash, output_hash,
parent_nodes, elapsed_ms}``. ``input_hash`` and ``output_hash`` are
SHA-256 of the values in ``Signal.names`` order.

References:
    Cheney, Chiticariu, Tan (2009), "Provenance in Databases: Why,
    How, and Where", Foundations and Trends in Databases
    (bibliographic only: no PDF in ``papers/``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _signal_hash(signal) -> str:
    if signal is None:
        return "none"
    if hasattr(signal, "names"):
        vals = [float(signal[n]) for n in signal.names]
    elif isinstance(signal, dict):
        vals = [float(v) for v in signal.values()]
    else:
        vals = [float(signal)]
    data = json.dumps(vals, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class Provenance:
    """A log of learn / predict calls, with a chain per output hash.

    Choices stated explicitly:
        The record format is a JSONL of
        ``{t, node, method, input_hash, output_hash, parent_nodes,
        elapsed_ms}``; the hashes are SHA-256 of the values in
        ``Signal.names`` order. This is a choice of the kit. The
        paper cited in the module (Cheney, Chiticariu, Tan 2009)
        formalises provenance for databases, not a concrete log
        format. A PROV-JSON export or a different hash (BLAKE2) would
        be a different choice.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        t: float,
        node: str,
        method: str,
        signal_in,
        output,
        parents: list[str],
        elapsed_ms: float = 0.0,
    ) -> None:
        self._records.append(
            {
                "t": float(t),
                "node": str(node),
                "method": str(method),
                "input_hash": _signal_hash(signal_in),
                "output_hash": _signal_hash(output),
                "parent_nodes": list(parents),
                "elapsed_ms": float(elapsed_ms),
            }
        )

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def chain(self, output_hash: str) -> list[dict[str, Any]]:
        """All records that led to ``output_hash``, in topological order."""
        out = []
        stack = [output_hash]
        seen = set()
        while stack:
            h = stack.pop()
            for r in self._records:
                key = (h, r["node"], r["method"])
                if r["output_hash"] == h and key not in seen:
                    seen.add(key)
                    out.append(r)
                    stack.append(r["input_hash"])
        return out[::-1]

    def to_jsonl(self, path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    @classmethod
    def from_jsonl(cls, path) -> Provenance:
        p = Path(path)
        prov = cls()
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prov._records.append(json.loads(line))
        return prov
