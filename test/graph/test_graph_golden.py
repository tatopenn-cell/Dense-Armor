"""Tests against the committed golden files."""

import json
from pathlib import Path

import numpy as np
import pytest

from dense_armor.core.graph import (
    Action,
    ActionRejected,
    CausalGraph,
    ExecutionDAG,
    Graph,
    Provenance,
    TypedGraph,
    eigendecomposition,
    fix_sign,
    gft,
    heat_kernel,
    page_rank,
    random_walk_restart,
)
from dense_armor.roles import Signal

GOLD = Path(__file__).parent / "golden"


class _Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


def _path(n: int) -> Graph:
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def _dag10() -> Graph:
    edges = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (4, 6, 1.0),
        (5, 7, 1.0),
        (6, 8, 1.0),
        (7, 9, 1.0),
        (8, 9, 1.0),
    ]
    return Graph(10, edges)


def _schema_from_names(names: dict) -> dict:
    """Ricostruisce lo schema da ``{tipo: {campo: nome_tipo}}``."""
    py = {"str": str, "float": float, "int": int, "list": list, "bool": bool}
    return {
        tname: {fname: py[tn] for fname, tn in fields.items()}
        for tname, fields in names.items()
    }


# ------------------------------------------------------------
# G1
# ------------------------------------------------------------
def test_path10_graph_golden():
    data = np.load(GOLD / "path10_graph.npz")
    g = _path(10)
    w, U = eigendecomposition(g.laplacian)
    U = fix_sign(U)
    assert np.allclose(g.laplacian.toarray(), data["laplacian"], atol=1e-12)
    assert np.allclose(w, data["eigenvalues"], atol=1e-10)
    assert np.allclose(U, data["eigenvectors"], atol=1e-10)


def test_path10_gft_golden():
    data = np.load(GOLD / "path10_graph.npz")
    x = np.load(GOLD / "path10_signal.npy")
    y = np.load(GOLD / "path10_gft.npy")
    assert np.allclose(gft(x, data["eigenvectors"]), y, atol=1e-10)


def test_path10_filtered_golden():
    data = np.load(GOLD / "path10_graph.npz")
    x = np.load(GOLD / "path10_signal.npy")
    y = np.load(GOLD / "path10_filtered.npy")
    H = heat_kernel(data["laplacian"], 1.0)
    assert np.allclose(H @ x, y, atol=1e-10)


# ------------------------------------------------------------
# G2
# ------------------------------------------------------------
def test_dag3_graph_golden():
    data = np.load(GOLD / "dag3_graph.npz", allow_pickle=True)
    dag = ExecutionDAG(
        nodes={"a": _Inc(), "b": _Inc(), "c": _Inc()},
        edges=[("a", "b"), ("b", "c")],
    )
    assert list(dag.topological_order()) == list(data["order"])


def test_dag3_outputs_golden():
    expected = json.loads((GOLD / "dag3_outputs.json").read_text())
    dag = ExecutionDAG(
        nodes={"a": _Inc(), "b": _Inc(), "c": _Inc()},
        edges=[("a", "b"), ("b", "c")],
    )
    out = dag.run({"a": Signal([1.0], names=["x"], units=[""])})
    for k, v in expected.items():
        assert abs(float(out[k]["x"]) - v) < 1e-12


def test_dag3_provenance_golden():
    lines = (GOLD / "dag3_provenance.jsonl").read_text().splitlines()
    assert len(lines) == 3
    prov = Provenance.from_jsonl(GOLD / "dag3_provenance.jsonl")
    chain = prov.chain(prov.records()[-1]["output_hash"])
    assert [r["node"] for r in chain] == ["a", "b", "c"]


# ------------------------------------------------------------
# G3
# ------------------------------------------------------------
def test_briefcase_golden():
    data = json.loads((GOLD / "briefcase.json").read_text())
    g = TypedGraph(schema=_schema_from_names(data["schema"]))
    for name, node in data["nodes"].items():
        g.add_node(name, node["type"], **node["state"])
    a = Action(
        name=data["action"]["name"],
        preconditions=[tuple(p) for p in data["action"]["preconditions"]],
        effects=[tuple(e) for e in data["action"]["effects"]],
    )
    after = a.apply(g)
    expected = json.loads((GOLD / "briefcase_after.json").read_text())
    assert dict(after.graph.nodes["r"]["state"]) == expected["r"]
    assert dict(after.graph.nodes["b"]["state"]) == expected["b"]


def test_briefcase_rejected_golden():
    data = json.loads((GOLD / "briefcase.json").read_text())
    g = TypedGraph(schema=_schema_from_names(data["schema"]))
    for name, node in data["nodes"].items():
        g.add_node(name, node["type"], **node["state"])
    g.update("r", status="moving")
    a = Action(
        name=data["action"]["name"],
        preconditions=[tuple(p) for p in data["action"]["preconditions"]],
        effects=[tuple(e) for e in data["action"]["effects"]],
    )
    with pytest.raises(ActionRejected):
        a.apply(g)
    expected = json.loads((GOLD / "briefcase_rejected.json").read_text())
    assert dict(g.graph.nodes["r"]["state"]) == expected["r"]
    assert dict(g.graph.nodes["b"]["state"]) == expected["b"]


# ------------------------------------------------------------
# G4
# ------------------------------------------------------------
def test_dag10_pagerank_golden():
    expected = json.loads((GOLD / "dag10_pagerank.json").read_text())
    g = _dag10()
    pr = page_rank(g)
    for k, v in expected.items():
        assert abs(pr[int(k)] - v) < 1e-10


def test_dag10_rwr_golden():
    expected = json.loads((GOLD / "dag10_rwr.json").read_text())
    g = _dag10()
    rwr = random_walk_restart(g, seed=0)
    for k, v in expected.items():
        assert abs(rwr[int(k)] - v) < 1e-10


def test_causal_dag3_golden():
    pytest.importorskip("dowhy")
    import numpy as np
    import pandas as pd

    data = json.loads((GOLD / "causal_dag3.json").read_text())
    cg = CausalGraph(nodes=data["nodes"], edges=[tuple(e) for e in data["edges"]])
    n = 2000
    rng = np.random.default_rng(0)
    X = rng.standard_normal(n)
    Z = X + rng.standard_normal(n) * 0.5
    Y = 2.0 * Z + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"X": X, "Z": Z, "Y": Y})
    eff = cg.estimate_effect("Z", 1.0, df, outcome="Y")
    assert abs(eff - data["expected_effect"]) < data["tolerance"]


def test_golden_files_present():
    expected = {
        "path10_graph.npz",
        "path10_signal.npy",
        "path10_gft.npy",
        "path10_filtered.npy",
        "dag3_graph.npz",
        "dag3_provenance.jsonl",
        "dag3_outputs.json",
        "briefcase.json",
        "briefcase_after.json",
        "briefcase_rejected.json",
        "dag10_pagerank.json",
        "dag10_rwr.json",
        "causal_dag3.json",
    }
    present = {f.name for f in GOLD.iterdir() if f.is_file()}
    assert expected <= present, f"mancano: {expected - present}"
