# Provenance

A network of estimators gives an answer; provenance tells you where it came from. Every time a node runs, the DAG writes one record. This page runs the chain of [the previous page](dag.md) with a log, follows one output back to its source and saves the log to a file.

## 1. One record per node

`Provenance()` is an empty log. Give it to `ExecutionDAG(..., provenance=log)` and each `run` adds one record per node that ran. A record has `t`, `node`, `method`, `input_hash`, `output_hash`, `parent_nodes`, `elapsed_ms` and the two hashes of the chain, `prev_record_hash` and `record_hash`.

```python
from dense_armor.core.graph import ExecutionDAG, Provenance
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


log = Provenance()
d = ExecutionDAG({"a": Inc(), "b": Inc(), "c": Inc()}, [("a", "b"), ("b", "c")], provenance=log)
d.run({"a": Signal([1.0], names=["x"], units=[""])})
rec = log.records()
print([r["node"] for r in rec], [r["t"] for r in rec])
print(rec[0]["parent_nodes"], rec[2]["parent_nodes"], len(rec[0]["input_hash"]))
```

```text
['a', 'b', 'c'] [0.0, 1.0, 2.0]
[] ['b'] 64
```

The records are in the order of the run. `t` counts the node runs, from 0, one step per node. A hash is the SHA-256 of the values of a signal in the order of its names: 64 hexadecimal characters. Two signals with the same values have the same hash. `elapsed_ms` is 0.0: the DAG does not measure time.

## 2. Following an output back

`chain(output_hash)` gives the records that led to the latest record with that output. It follows `parent_nodes`: the parent of a record is the most recent earlier record of that node.

```python
from dense_armor.core.graph import ExecutionDAG, Provenance
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


log = Provenance()
d = ExecutionDAG({"a": Inc(), "b": Inc(), "c": Inc()}, [("a", "b"), ("b", "c")], provenance=log)
d.run({"a": Signal([1.0], names=["x"], units=[""])})
h = log.records()[1]["output_hash"]
print([r["node"] for r in log.chain(h)])
```

```text
['a', 'b']
```

The output of `b` came from `b`, and `b` came from `a`: `c` is not in the chain because it ran after. If two nodes give the same values the hash is the same for both, so `chain(h, node="b")` picks the latest record of the node `b`.

## 3. Saving and reading the log

`to_jsonl(path)` writes one JSON record per line and `Provenance.from_jsonl(path)` reads them back.

```python
import tempfile
from pathlib import Path
from dense_armor.core.graph import Provenance
from dense_armor.roles import Signal

log = Provenance()
s = Signal([1.0], names=["x"], units=[""])
log.record(0.0, "a", "run", s, s, [])
p = Path(tempfile.mkdtemp()) / "log.jsonl"
log.to_jsonl(p)
print(Provenance.from_jsonl(p).records() == log.records())
```

```text
True
```

## 4. A log that shows tampering

A log is only worth something if you can tell that it was not changed. Each record holds the hash of the record before it (`prev_record_hash`) and its own (`record_hash`), the SHA-256 of the previous hash and of the fields of the record. `verify()` follows the chain, and `broken_at()` gives the index of the first record that does not fit.

```python
from dense_armor.core.graph import ExecutionDAG, Provenance
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


log = Provenance()
d = ExecutionDAG({"a": Inc(), "b": Inc(), "c": Inc()}, [("a", "b"), ("b", "c")], provenance=log)
d.run({"a": Signal([1.0], names=["x"], units=[""])})
saved = log.head
print(log.verify(saved), log.broken_at())
log.records()[1]["output_hash"] = "0" * 64
print(log.verify(saved), log.broken_at())
```

```text
True None
False 1
```

The first line says the chain is intact. After the output of the second record is changed, the chain breaks at that record: from there on, no record can be changed, removed, inserted or moved without being found. `head` is the hash of the last record. Keep it somewhere else when the log is written, and `verify(head)` also finds records that were cut off the end, which the chain alone cannot see.

## Details

The format of the record is a choice of this module: the cited paper (Cheney, Chiticariu and Tan 2009) describes provenance for databases and not a concrete log. A PROV-JSON export or another hash would be a different choice. The hash does not include the names of the channels, only their values. A node that a gate blocked leaves no record, and neither do the nodes after it.
