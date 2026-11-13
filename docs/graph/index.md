# How do I work with signals that live on a graph?

A graph is a set of nodes joined by edges. A **graph signal** is one number on each node: the temperature at each sensor, the load on each machine. This section goes from one graph, to a signal on it, to a network of estimators, to a graph with rules, to the question "which node is wrong, and what causes what".

The code is in `dense_armor/core/graph/`. The structure and the spectral functions need NumPy and SciPy. The DAG, the typed graph, PageRank and the causal graph use NetworkX, and the effect estimate uses DoWhy and pandas:

```bash
pip install "dense-armor[graph]"
pip install "dense-armor[causal]"
```

## The path

| Page | What you get | Main names |
| --- | --- | --- |
| [Graph and spectrum](spectral.md) | a weighted graph, its Laplacian, its frequencies, the Fourier transform of a signal on it | `Graph`, `eigendecomposition`, `fiedler_vector`, `gft`, `igft`, `GraphSignal` |
| [Filters](filters.md) | smoothing a graph signal, exactly and with a polynomial that needs no eigenvectors | `heat_kernel`, `low_pass`, `apply_filter`, `chebyshev_coefficients`, `chebyshev_filter` |
| [A network of estimators](dag.md) | estimators joined by edges, run in order, with a gate that blocks anomalies | `ExecutionDAG`, `GateNode` |
| [Provenance](provenance.md) | a log that says which node produced which output | `Provenance` |
| [A graph with rules](ontology.md) | typed nodes, constraints on edges, actions with preconditions | `TypedGraph`, `Action`, `apply_action`, `ActionRejected` |
| [Which node is wrong?](anomaly.md) | PageRank, a random walk, a per-node anomaly score | `page_rank`, `random_walk_restart`, `graph_anomaly_score` |
| [What causes what?](causal.md) | the effect of one variable on another, adjusting for the confounders | `CausalGraph` |

Each page starts from the result of the page before it. Every number shown was printed by the code above it, and the figures are drawn by `docs/assets/graph/make_figures.py`.

## Sources and checks

| Module | Source | Used for |
| --- | --- | --- |
| `core`, `spectral`, `fourier` | Niresi, Kuhn, Frusque, Fink (2024), arXiv:2406.03898, Section II.A, Eq. (1)-(3) | the Laplacian `L = D - W` |
| `fourier`, `filters`, `spectral` | Liu, He, Laurent, Di Giovanni, Bronstein, Bresson (2024), arXiv:2405.13806, Sections 2 and 3.1 | the transform `U^T f`, the Chebyshev recurrence |
| `dag`, `ontology`, `causal`, `anomaly` | the papers named in each docstring | background only: no PDF of them was available, and no section was checked |

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

## Details

The section was built in four steps: graph and spectrum, DAG and provenance, typed graph and actions, PageRank and causality. A first review found nine mistakes, listed in `REVISIONE.md`, among them an anomaly score built on the Fourier magnitude, which is a frequency and not a per-node quantity, and a copy of an action that shared the state of the other nodes. A second review changed the points marked "Details" in [the DAG](dag.md), [provenance](provenance.md), [the typed graph](ontology.md) and [the anomaly score](anomaly.md).
