# Ontology

Until now a Dense-Armor estimator looked at one signal: one stream of numbers. A real system is many signals that are **related**: the services of a company and the databases they use, the sensors of a plant and the machines they watch, the parts of a vehicle. An ontology is how Dense-Armor describes such a system.

An ontology says three things:

1. **What the parts are.** Every part has a kind (a service, a database, a sensor) and holds typed fields.
2. **How they are related.** A service depends on a database; a sensor watches a machine.
3. **What must be true.** A service in production must not depend on a test database. The rule is written once, and the library finds every relation that breaks it.

That is what the page [Types, rules and actions](typed.md) teaches, step by step, and it is the page to read first.

## Why graph theory is here

A set of parts and relations is a graph, so the mathematics of graphs gives the ontology its tools. Each of the other pages is one tool, and says what it is for:

| Page | The tool | What you use it for |
| --- | --- | --- |
| [Types, rules and actions](typed.md) | `TypedGraph`, `Constraint`, `Action` | describe the system, check the rules, change it safely |
| [Graph and spectrum](spectral.md) | `Graph`, `gft`, `GraphSignal` | put one number on every part and study how it varies across the relations |
| [Filters](filters.md) | `heat_kernel`, `low_pass`, `chebyshev_filter` | smooth those numbers over the relations |
| [Execution DAG](dag.md) | `ExecutionDAG`, `GateNode` | run estimators in a chain, and stop a chain when a reading is anomalous |
| [Provenance](provenance.md) | `Provenance` | know which part produced which output |
| [Anomaly detection](anomaly.md) | `page_rank`, `random_walk_restart`, `graph_anomaly_score` | find the part that is wrong |
| [Causality](causal.md) | `CausalGraph` | tell what causes what from what merely moves together |

The code is in `dense_armor/core/graph/`. The structure and the spectral functions need NumPy and SciPy. The ontology, the DAG, PageRank and the causal graph use NetworkX, and the effect estimate uses DoWhy and pandas:

```bash
pip install "dense-armor[graph]"
pip install "dense-armor[causal]"
```

Every number shown in these pages was printed by the code above it, and the figures are drawn by `docs/assets/ontology/make_figures.py`.

## Sources and checks

| Module | Source | Used for |
| --- | --- | --- |
| `core`, `spectral`, `fourier` | Niresi, Kuhn, Frusque, Fink (2024), arXiv:2406.03898, Section II.A: `L := D - W` and Eq. (2) | the Laplacian |
| `fourier`, `filters`, `spectral` | Liu, He, Laurent, Di Giovanni, Bronstein, Bresson (2024), arXiv:2405.13806, Sections 2 and 3.1 | the transform `U^T f`, the Chebyshev recurrence |
| `dag`, `ontology`, `causal`, `anomaly` | the papers named in each docstring | background only: not in the paper collection of the task, and no section was checked |

What is checked against another implementation, in `test/graph/`:

| What | Against | Tolerance |
| --- | --- | --- |
| normalised Laplacian | `networkx.normalized_laplacian_matrix` | 1e-12 |
| eigenvalues of path, cycle, complete, star graphs | closed forms | 1e-10 |
| `gft` | PyGSP, up to the sign of each eigenvector | 1e-10 |
| `heat_kernel` | `scipy.linalg.expm` and PyGSP `Heat` | 1e-10 |
| `chebyshev_filter` | the exact filter `apply_filter` | below 1e-10 at order 20 |
| `page_rank` | `networkx.pagerank` | 1e-12 |
| topological order | Kahn's algorithm written by hand | exact |
| the whole pipeline | the golden files in `test/graph/golden/`, made by `gen_golden.py` | exact |
