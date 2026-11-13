# A network of estimators

So far one signal went through one function. Here several estimators are joined by edges, and each hands its output to the next. This page builds a chain of three nodes, then puts a gate in the chain that stops anomalies.

## 1. A chain of estimators

`ExecutionDAG(nodes, edges)` takes a dict `{name: estimator}` and a list of edges `(src, dst)`. `run(signals)` gives each source node its signal in `signals`, runs the nodes in topological order, and returns the output of each node as a `Signal`. A cycle or an edge to an unknown node raises `ValueError`.

```python
from dense_armor.core.graph import ExecutionDAG
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


d = ExecutionDAG({"a": Inc(), "b": Inc(), "c": Inc()}, [("a", "b"), ("b", "c")])
out = d.run({"a": Signal([1.0], names=["x"], units=[""])})
print(d.topological_order(), d.ancestors("c"), d.descendants("a"))
print({k: float(v["x"]) for k, v in out.items()})
```

```text
['a', 'b', 'c'] ['a', 'b'] ['b', 'c']
{'a': 2.0, 'b': 3.0, 'c': 4.0}
```

Each node adds 1 to what it receives. `ancestors` and `descendants` are sorted lists of names. The node is used by what it can do: an anomaly detector (`score_one`) gives a one-value signal with its score, a transformer (`transform_one`) learns the sample and gives the transformed one, and a regressor or classifier (`predict_one`) predicts first and learns after.

A regressor or classifier learns only when you give it the true value of the sample with `run(signals, labels={"name": y})`. Without a label it only predicts.

## 2. A gate that blocks anomalies

`GateNode(detector, threshold)` is a node that returns its input if the detector finds it normal and `None` if it finds it anomalous. The nodes after a blocked gate are skipped and their output is `None`. The gate scores every sample once, and the detector learns only the samples that pass. Without `threshold`, the gate reads the `threshold` attribute of the detector, and a detector with neither is refused.

```python
from dense_armor.core.graph import ExecutionDAG, GateNode
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


class Det:
    threshold = 5.0
    n = 0

    def score_one(self, x, t=None):
        return float(x["x"])

    def learn_one(self, x, t=None):
        self.n += 1
        return self


det = Det()
gate = GateNode(det)
d = ExecutionDAG({"gate": gate, "b": Inc()}, [("gate", "b")])
for v in (1.0, 10.0, 2.0):
    print(v, d.run({"gate": Signal([v], names=["x"], units=[""])})["b"] is None)
print(gate.passed, gate.blocked, det.n)
```

```text
1.0 False
10.0 True
2.0 False
2 1 2
```

The sample 10.0 scores above the threshold 5.0, so the gate returns `None`, node `b` is skipped, and the detector does not learn it: it learned 2 samples out of 3. The gate declares `budget_s` (a `score_one` and a `learn_one` of the detector) and `memory_class` (that of the detector).

## 3. Describe the flow, then build it

`Dataflow` is a way to write the same network as a description. You declare where each signal comes from, ask where a node's signal came from, and then build the `ExecutionDAG`.

```python
from dense_armor.core.graph import Dataflow
from dense_armor.roles import Signal


class Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


d = Dataflow().source("raw", Inc()).then("clean", Inc(), after="raw").then("score", Inc(), after="clean")
print(d.lineage("score"), d.edges())
out = d.build().run({"raw": Signal([1.0], names=["x"], units=[""])})
print({k: float(v["x"]) for k, v in out.items()})
```

```text
['raw', 'clean', 'score'] [('raw', 'clean'), ('clean', 'score')]
{'raw': 2.0, 'clean': 3.0, 'score': 4.0}
```

`source(name, estimator)` declares a node that receives a signal from outside, and `then(name, estimator, after=...)` declares a node that receives the output of another. `lineage(name)` lists the nodes from the source down to `name`, and `build(provenance=...)` returns the `ExecutionDAG`, with the log if you give one. A name used twice, or a node that comes after a name not declared before it, raises `ValueError`.

## Details

Every node has at most one predecessor, so the network is a tree and the run is one pass over the topological order. A node with two predecessors raises `ValueError` when `run` is called. A network with joins is a different design, left out on purpose. A repeated edge is kept once. The log of what each node did is the subject of [the next page](provenance.md).
