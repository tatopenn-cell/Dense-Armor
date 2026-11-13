# Graph Engine — branch `feature/graph-engine`

This branch adds the **Graph Engine** to Dense-Armor: from a single
signal to a system of signals in relation. Four levels, 103 tests
passing, 8 documentation pages.

## What is in it

### G1 — Graph + GFT + spectral filters
- `Graph`: weighted undirected graph, combinatorial and normalized
  Laplacian (checked against `networkx` at 1e-12).
- `eigendecomposition`, `fiedler_vector`, `fix_sign`: Laplacian
  spectrum (checked against closed forms: path, cycle, complete,
  star; residual `||LU - UΛ||` and orthonormality).
- `gft`, `igft`: Graph Fourier Transform (Parseval at 1e-10, checked
  against `pygsp` up to the sign of the eigenvectors).
- `heat_kernel`, `low_pass`, `apply_filter`: spectral filters
  (`heat_kernel` = `expm(-tau L)` at 1e-10, `pygsp Heat` with
  `scale=tau*lmax` at 1e-10).
- `chebyshev_coefficients`, `chebyshev_filter`: Chebyshev recurrence
  with the shifted operator `A = 2L/lmax - I` (error < 1e-10 at
  K=20, decreasing down to machine epsilon).
- `GraphSignal`: a `Signal` whose values live on the graph nodes.

### G2 — DAG + Provenance + Gate
- `ExecutionDAG`: deterministic network of estimators, topological
  order (checked against Kahn written by hand).
- `Provenance`: JSONL log of every pass, chain per `output_hash`.
- `GateNode`: an `AnomalyGate` as a DAG node, blocks the nodes
  downstream.
- `Provenance` + `ExecutionDAG` integration: every `run()` records
  the rows, `chain()` rebuilds the chain.

### G3 — Typed ontology + STRIPS
- `TypedGraph`: typed nodes with a schema, edges with a relation and
  an optional constraint.
- `Action`, `apply_action`: preconditions + effects,
  `ActionRejected` if a precondition fails, graph unchanged on
  rejection.
- `copy.deepcopy` in `apply` (the other nodes are not shared with
  the input graph).

### G4 — PageRank + causality
- `page_rank`, `random_walk_restart`: centrality and propagation
  (checked against `networkx.pagerank` at 1e-12).
- `graph_anomaly_score`: local residual `x - heat_kernel(L, tau) @ x`
  plus PageRank. The anomalous node wins on the test graph.
- `CausalGraph`, `estimate_effect`: DoWhy with
  `backdoor.linear_regression` (effect 1.9982 on a DAG whose true
  effect is 2.0).

## How to test it

```bash
git clone https://github.com/tatopenn-cell/Dense-Armor.git
cd Dense-Armor
git checkout feature/graph-engine
pip install -e ".[dev]"
pip install ruff mypy pygsp dowhy
python -m pytest test/graph/ -q          # 103 passed
python -m ruff check dense_armor/core/graph/ test/graph/
python -m mypy --follow-imports=skip dense_armor/core/graph/*.py
mkdocs build --strict --site-dir /tmp/_site
```

## What was fixed in this session

Nine bugs found and fixed:

1. `test_spectral.py`: the cycle was built without its closing edge
   (it was a path).
2. `test_fourier.py`: `gft` compared without gauge (eigenvector
   sign).
3. `anomaly.py`: `graph_anomaly_score` used `|gft|`, which is not a
   per-node score (the GFT is not localized).
4. `ontology.py`: the parameter `type` shadowed the builtin.
5. `actions.py`: shallow `copy()`, the other nodes shared the state
   dict.
6. `actions.py`: `before` computed and unused (the pre-validation
   was not applied).
7. `test_provenance.py`: `_Pass` nodes gave identical hashes, so
   `chain` could not tell them apart.
8. `gen_golden.py`: `SCHEMA` contained Python types, not JSON
   serializable.
9. `gen_golden.py`: `GOLD` not defined.

### Added on top of G1-G4
- `GraphFourier`, `SpectralFilter`, `ChebyshevFilter`: the transform and
  the filters as objects, with the basis found once.
- `SpectralWavelet`, `chebyshev_wavelet_terms`: graph wavelets from the
  transformed Chebyshev terms (arXiv:2405.13806, Section 3.1).
- `Dataflow`: describe the flow of signals, then build the `ExecutionDAG`.
- `Constraint`, `ConstraintViolation`: named rules on the edges of a
  `TypedGraph`; `validate(raise_on_violation=True)` raises.
- `structural_change`: what changed between two graphs.
- `recover_graph_signal`: time-varying signal recovery on a dynamic graph
  (arXiv:2202.06432, Algorithm 1).
- `learn_graph`, `weights_from_drops`: informed graph learning
  (arXiv:2406.03898, Section III).
- `AdaCGP`: online estimation of the weights of a graph
  (arXiv:2411.01567, Algorithms 1 and 2).

## The documentation

`docs/ontology/` has eight pages (overview, graph and spectrum, filters,
network of estimators, provenance, graph with rules, anomaly,
causality) and three figures made by
`docs/assets/ontology/make_figures.py`. Every code block was run and its
printed output is the one in the page.

## What is NOT in it

- Four of the six cited papers (Cheng, Babaei Giglou, Zan, Wang) are
  bibliographic only: no PDF, no section checked.

## Numbers

| Item | Value |
|---|---|
| Tests | 103 passed, none skipped |
| ruff check / format | clean |
| mypy | Success: no issues found in 13 source files |
| mkdocs build --strict | passes, graph pages in the nav |
| Golden files | 13 (in `test/graph/golden/`) |

## Commits

See `git log --oneline 16f71a0..HEAD`.
