"""Tests for Provenance."""

from dense_armor.core.graph import Provenance
from dense_armor.roles import Signal


def test_chain_three_records(tmp_path):
    prov = Provenance()
    sig = Signal([1.0], names=["x"], units=[""])
    out_a = Signal([2.0], names=["y"], units=[""])
    out_b = Signal([3.0], names=["z"], units=[""])
    out_c = Signal([4.0], names=["w"], units=[""])
    prov.record(0.0, "a", "transform", sig, out_a, [])
    prov.record(1.0, "b", "transform", out_a, out_b, ["a"])
    prov.record(2.0, "c", "transform", out_b, out_c, ["b"])
    chain = prov.chain(prov.records()[-1]["output_hash"])
    assert len(chain) == 3
    assert [r["node"] for r in chain] == ["a", "b", "c"]


def test_jsonl_roundtrip(tmp_path):
    prov = Provenance()
    sig = Signal([1.0], names=["x"], units=[""])
    prov.record(0.0, "a", "transform", sig, sig, [])
    p = tmp_path / "prov.jsonl"
    prov.to_jsonl(p)
    back = Provenance.from_jsonl(p)
    assert back.records() == prov.records()


def test_hash_stable_and_distinct():
    s1 = Signal([1.0, 2.0], names=["a", "b"], units=["", ""])
    s2 = Signal([1.0, 2.0], names=["a", "b"], units=["", ""])
    s3 = Signal([1.0, 3.0], names=["a", "b"], units=["", ""])
    h1 = Provenance()
    h1.record(0.0, "a", "transform", s1, s2, [])
    r = h1.records()[0]
    assert len(r["input_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in r["input_hash"])
    h2 = Provenance()
    h2.record(0.0, "a", "transform", s3, s2, [])
    assert h2.records()[0]["input_hash"] != r["input_hash"]


class _Inc:
    """A transformer that adds 1 to every value: the hash changes at each node."""

    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


def test_dag_integration():
    from dense_armor.core.graph import ExecutionDAG, Provenance
    from dense_armor.roles import Signal

    prov = Provenance()
    dag = ExecutionDAG(
        nodes={"a": _Inc(), "b": _Inc(), "c": _Inc()},
        edges=[("a", "b"), ("b", "c")],
        provenance=prov,
    )
    sig = Signal([1.0], names=["x"], units=[""])
    out = dag.run({"a": sig})
    assert out["c"] is not None
    # 3 record, uno per nodo
    recs = prov.records()
    assert len(recs) == 3
    assert [r["node"] for r in recs] == ["a", "b", "c"]
    # chain dell'output di c: 3 record, in ordine
    chain = prov.chain(recs[-1]["output_hash"])
    assert len(chain) == 3
    assert [r["node"] for r in chain] == ["a", "b", "c"]
    # t cresce
    assert recs[0]["t"] < recs[1]["t"] < recs[2]["t"]


def test_chain_with_equal_outputs_at_two_nodes_takes_the_named_node():
    prov = Provenance()
    sig = Signal([1.0], names=["x"], units=[""])
    out = Signal([2.0], names=["y"], units=[""])
    prov.record(0.0, "a", "run", sig, out, [])
    prov.record(1.0, "b", "run", sig, out, [])
    h = prov.records()[0]["output_hash"]
    assert [r["node"] for r in prov.chain(h)] == ["b"]
    assert [r["node"] for r in prov.chain(h, node="a")] == ["a"]
