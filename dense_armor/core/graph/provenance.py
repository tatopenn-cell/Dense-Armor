"""Provenance: a JSONL log of every run of a DAG node.

Each record is ``{t, node, method, input_hash, output_hash,
parent_nodes, elapsed_ms}``. ``input_hash`` and ``output_hash`` are
SHA-256 of the values in ``Signal.names`` order. ``elapsed_ms`` is written
as 0.0 by ``ExecutionDAG`` (the time is not measured).

References:
    Wang and others (2026), "From Agent Traces to Trust: A Survey of
    Evidence Tracing and Execution Provenance in LLM Agents",
    arXiv:2606.04990: execution provenance as a typed graph of a run, and
    the gap of traces that are not tamper-evident (abstract only).
    Cheney, Chiticariu, Tan (2009), "Provenance in Databases: Why,
    How, and Where", Foundations and Trends in Databases
    (bibliographic only).
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


_GENESIS = "0" * 64
_FIELDS = (
    "t",
    "node",
    "method",
    "input_hash",
    "output_hash",
    "parent_nodes",
    "elapsed_ms",
)


def _record_hash(prev: str, body: dict) -> str:
    data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(prev.encode() + data).hexdigest()


class Provenance:
    """A log of node runs; ``chain`` follows ``parent_nodes``.

    The records are linked by a hash chain: each one holds the hash of the
    one before (``prev_record_hash``) and its own (``record_hash``, the
    SHA-256 of the previous hash and of the fields of the record). A record
    that is changed, removed, inserted or moved breaks the chain from that
    point on, and :meth:`verify` says so. The last hash, :attr:`head`, kept
    elsewhere, also shows that records were cut off the end of the log.

    Choices stated explicitly:
        The record format is a JSONL of
        ``{t, node, method, input_hash, output_hash, parent_nodes,
        elapsed_ms, prev_record_hash, record_hash}``; the hashes of the
        signals are SHA-256 of the values in ``Signal.names`` order. This is a design choice of this module. The
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
        body = {
            "t": float(t),
            "node": str(node),
            "method": str(method),
            "input_hash": _signal_hash(signal_in),
            "output_hash": _signal_hash(output),
            "parent_nodes": list(parents),
            "elapsed_ms": float(elapsed_ms),
        }
        prev = self.head
        self._records.append(
            {**body, "prev_record_hash": prev, "record_hash": _record_hash(prev, body)}
        )

    @property
    def head(self) -> str:
        """The hash of the last record (a fixed string for an empty log)."""
        return self._records[-1]["record_hash"] if self._records else _GENESIS

    def broken_at(self) -> int | None:
        """The index of the first record that breaks the chain, or ``None``.

        A record breaks it if a field is missing or changed, if it does not
        point to the hash of the record before, or if it was inserted or
        moved.
        """
        prev = _GENESIS
        for i, r in enumerate(self._records):
            if any(k not in r for k in _FIELDS):
                return i
            body = {k: r[k] for k in _FIELDS}
            if r.get("prev_record_hash") != prev:
                return i
            if r.get("record_hash") != _record_hash(prev, body):
                return i
            prev = r["record_hash"]
        return None

    def verify(self, head: str | None = None) -> bool:
        """``True`` if the chain is intact.

        Args:
            head: the value of :attr:`head` saved when the log was written. If
                it is given, the log must also end at that record, which shows
                that nothing was cut off the end.
        """
        return self.broken_at() is None and (head is None or head == self.head)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def chain(self, output_hash: str, node: str | None = None) -> list[dict[str, Any]]:
        """The records that led to the latest record with ``output_hash``.

        The hash is of the values only, so two nodes with equal output
        share it: pass ``node`` to take the latest record of that node.
        The chain follows ``parent_nodes``: the parent of a record is the
        most recent earlier record of that node, so equal values at
        different nodes or times are not mixed. The records are in log
        order, which is the topological order of a run.
        """
        last = max(
            (
                i
                for i, r in enumerate(self._records)
                if r["output_hash"] == output_hash
                and (node is None or r["node"] == node)
            ),
            default=None,
        )
        if last is None:
            return []
        seen: set[int] = set()
        stack = [last]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            for parent in self._records[i]["parent_nodes"]:
                j = max(
                    (k for k in range(i) if self._records[k]["node"] == parent),
                    default=None,
                )
                if j is not None:
                    stack.append(j)
        return [self._records[i] for i in sorted(seen)]

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
