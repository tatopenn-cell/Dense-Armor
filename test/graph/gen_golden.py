"""Generate the golden files for G1-G4.

Run once:
    python test/graph/gen_golden.py

The files are committed. ``test_golden.py`` loads them and checks that
the current code reproduces them. The path of the golden files is
``Path(__file__).parent / "golden"``, never an absolute path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dense_armor.core.graph import (
    Action,
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

GOLDEN = Path(__file__).parent / "golden"
GOLD = GOLDEN  # alias usato dalle funzioni gen_*


def _path(n: int) -> Graph:
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def _dag10() -> Graph:
    """A 10-node graph for PageRank and RWR."""
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


# ------------------------------------------------------------
# G1 — path graph of 10 nodes
# ------------------------------------------------------------
def gen_g1() -> None:
    g = _path(10)
    w, U = eigendecomposition(g.laplacian)
    U = fix_sign(U)
    x = np.sin(np.arange(10) * 0.7)
    np.savez(
        GOLD / "path10_graph.npz",
        laplacian=g.laplacian.toarray(),
        eigenvalues=w,
        eigenvectors=U,
    )
    np.save(GOLD / "path10_signal.npy", x)
    np.save(GOLD / "path10_gft.npy", gft(x, U))
    np.save(GOLD / "path10_filtered.npy", heat_kernel(g.laplacian, 1.0) @ x)


# ------------------------------------------------------------
# G2 — DAG of 3 nodes with provenance
# ------------------------------------------------------------
class _Inc:
    def __init__(self):
        self.n = 0

    def learn_one(self, x, y=None):
        self.n += 1
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


def gen_g2() -> None:
    prov = Provenance()
    dag = ExecutionDAG(
        nodes={"a": _Inc(), "b": _Inc(), "c": _Inc()},
        edges=[("a", "b"), ("b", "c")],
        provenance=prov,
    )
    sig = Signal([1.0], names=["x"], units=[""])
    out = dag.run({"a": sig})
    # ordine topologico
    np.savez(
        GOLD / "dag3_graph.npz",
        nodes=np.array(sorted(dag.nodes.keys())),
        edges=np.array([(u, v) for u, v in dag.edges]),
        order=np.array(dag.topological_order()),
    )
    prov.to_jsonl(GOLD / "dag3_provenance.jsonl")
    # output come JSON
    outputs = {k: float(v["x"]) if v is not None else None for k, v in out.items()}
    (GOLD / "dag3_outputs.json").write_text(json.dumps(outputs))


# ------------------------------------------------------------
# G3 — STRIPS example (robot + briefcase)
# ------------------------------------------------------------
SCHEMA = {
    "robot": {"location": str, "status": str},
    "briefcase": {"location": str},
}


def _briefcase_graph():
    g = TypedGraph(schema=SCHEMA)
    g.add_node("r", "robot", location="loc_a", status="idle")
    g.add_node("b", "briefcase", location="loc_a")
    return g


def gen_g3() -> None:
    g = _briefcase_graph()
    action = Action(
        name="move",
        preconditions=[("r", "status", "==", "idle")],
        effects=[("r", "location", "set", "loc_b")],
    )
    schema_names = {
        tname: {fname: t.__name__ for fname, t in fields.items()}
        for tname, fields in SCHEMA.items()
    }
    (GOLD / "briefcase.json").write_text(
        json.dumps(
            {
                "schema": schema_names,
                "nodes": {
                    "r": {
                        "type": "robot",
                        "state": {"location": "loc_a", "status": "idle"},
                    },
                    "b": {"type": "briefcase", "state": {"location": "loc_a"}},
                },
                "action": {
                    "name": "move",
                    "preconditions": [["r", "status", "==", "idle"]],
                    "effects": [["r", "location", "set", "loc_b"]],
                },
            },
            indent=2,
        )
    )
    after = action.apply(g)
    (GOLD / "briefcase_after.json").write_text(
        json.dumps(
            {
                "r": dict(after.graph.nodes["r"]["state"]),
                "b": dict(after.graph.nodes["b"]["state"]),
            },
            indent=2,
        )
    )
    # rejected: cambia status a "moving", poi l'azione non può partire
    g2 = _briefcase_graph()
    g2.update("r", status="moving")
    rejected = Action(
        name="move",
        preconditions=[("r", "status", "==", "idle")],
        effects=[("r", "location", "set", "loc_b")],
    )
    try:
        rejected.apply(g2)
        raise SystemExit("expected rejection")
    except Exception:
        pass
    (GOLD / "briefcase_rejected.json").write_text(
        json.dumps(
            {
                "r": dict(g2.graph.nodes["r"]["state"]),
                "b": dict(g2.graph.nodes["b"]["state"]),
            },
            indent=2,
        )
    )


# ------------------------------------------------------------
# G4 — PageRank, RWR, causal on 10 nodes (PageRank/RWR) and 3 (causal)
# ------------------------------------------------------------
def gen_g4() -> None:
    g = _dag10()
    pr = page_rank(g)
    (GOLD / "dag10_pagerank.json").write_text(json.dumps(pr, sort_keys=True))
    rwr = random_walk_restart(g, seed=0)
    (GOLD / "dag10_rwr.json").write_text(json.dumps(rwr, sort_keys=True))
    # causal: 3 nodi, effetto 2.0 noto a mano (Z = X + noise, Y = 2 Z + noise)
    (GOLD / "causal_dag3.json").write_text(
        json.dumps(
            {
                "nodes": ["X", "Z", "Y"],
                "edges": [["X", "Z"], ["Z", "Y"]],
                "expected_effect": 2.0,
                "tolerance": 0.1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    GOLDEN.mkdir(parents=True, exist_ok=True)
    gen_g1()
    gen_g2()
    gen_g3()
    gen_g4()
    print(f"golden files written in {GOLDEN}")
    for f in sorted(GOLDEN.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
