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
| [Graph and spectrum](spectral.md) | `Graph`, `GraphFourier`, `gft`, `GraphSignal` | put one number on every part and study how it varies across the relations |
| [Filters](filters.md) | `SpectralFilter`, `ChebyshevFilter`, `SpectralWavelet`, `heat_kernel`, `low_pass`, `chebyshev_filter` | smooth those numbers over the relations |
| [Execution DAG](dag.md) | `Dataflow`, `ExecutionDAG`, `GateNode` | run estimators in a chain, and stop a chain when a reading is anomalous |
| [Provenance](provenance.md) | `Provenance` | know which part produced which output |
| [Anomaly detection](anomaly.md) | `page_rank`, `random_walk_restart`, `graph_anomaly_score`, `structural_change` | find the part that is wrong, and see what changed when the graph itself changes |
| [Recovering a signal](recovery.md) | `recover_graph_signal` | estimate the true readings from noisy, missing and wrong ones, on a graph that may move |
| [Learning a graph](learning.md) | `learn_graph`, `weights_from_drops` | build the graph from the readings, with what you already know |
| [Following a graph online](online.md) | `AdaCGP` | update the estimate of the links at every sample |
| [Causality](causal.md) | `CausalGraph` | tell what causes what from what merely moves together |

The code is in `dense_armor/core/graph/`. The structure, the spectral functions, the typed graph with its rules and actions, and the execution DAG need NumPy and SciPy only: they start at once and import no graph package. PageRank, the random walk with restart, `to_networkx()` and the causal graph use NetworkX, and the effect estimate uses DoWhy and pandas:

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
| `recovery` | Yamagata, Naganuma, Ono (2022), arXiv:2202.06432, Section III, Algorithm 1 | the problem (13), the proximity operators (23)-(28), the algorithm |
| `learning` | Niresi, Kuhn, Frusque, Fink (2024), arXiv:2406.03898, Section III | the physics-inspired weights and the problem (9)-(11) |
| `adacgp` | Jenkins and others (2024), arXiv:2411.01567, Sections II-IV | the model (5), Algorithm 1 and 2, the constants of IV.D |
| `wavelet` | Liu and others (2024), arXiv:2405.13806, Sections 2 and 3.1 | the transformed Chebyshev terms (4)-(5) |
| `change` | Jenkins, Variddhisai, El-Medany, Ng, Mandic (2024), arXiv:2411.01567, Section VI | the idea of following the weights and the topology of a graph over time; the two measures are defined in this module |
| `anomaly` (`root_cause_walk`) | Khan, Farea (2026), arXiv:2608.08073, Sections 3.1-3.5, Algorithm 1 | the iterative random walk with restart and its enhanced form |
| `ontology`, `actions` | Vijay, Ethiraj (2025), arXiv:2512.20275, Sections 1-3 | a knowledge graph as a state machine, actions validated on the subgraph they touch |
| `causal` | Sharma, Kiciman (2020), arXiv:2011.04216 | the library the estimator calls and its four steps |
| `provenance`, `dag` | Wang and others (2026), arXiv:2606.04990 (abstract only) | execution provenance as a typed graph of a run |

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
| `Action.apply` | the classical STRIPS rule on sets of facts: the preconditions are in the state, the new state is the state minus the deleted facts plus the added ones | exact |
| `structural_change` | values worked out by hand on a small graph | 1e-12 |
| the whole pipeline | the golden files in `test/graph/golden/`, made by `gen_golden.py` | exact |
