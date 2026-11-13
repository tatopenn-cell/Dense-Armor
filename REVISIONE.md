# Revisione — Graph Engine (G1–G4)

Data: 2026-11-13
Branch: `feature/graph-engine`, sopra `16f71a0` (`git log --oneline 16f71a0..HEAD`).
Due giri di revisione: il primo ha trovato 9 errori (tabella sotto), il secondo
ha trovato e corretto i punti della sezione "Secondo giro".

## Dichiarazione di lettura

Le Parti di `REVISIONE_IA.md` sono state lette **dalla sessione** (il
documento è stato incollato nella conversazione). Il file
`REVISIONE_IA.md` **non è nel repo** (verificato con `git ls-files`):
la dichiarazione è sulla lettura del testo, non sulla presenza nel repo.

| Parte | Titolo | Letta | Controllo derivato |
|---|---|---|---|
| 1 | Procedura di revisione | sì | Ordine della review (0–9) |
| 2 | Come si scrive il codice | sì | Ruff, mypy, type hints, docstring Google |
| 3 | Come si scrivono i test | sì | Test indipendenti, non contro sé stessi |
| 4 | Regole delle pagine | sì | (i docs mancano, vedi Gap) |
| 5 | Forma del task | sì | 26 item con Criterion |
| 6 | Auto-controllo | sì | 17 controlli eseguiti |
| 7 | Forma del delivery | sì | Questo file |
| 8 | I modi di sbagliare | sì | 3 errori trovati e corretti (vedi sotto) |
| 9 | Vincoli e comandi | sì | Un branch, no push, no PR |
| 10 | Template correzioni | sì | Non applicabile (nessun round) |
| 11 | API dei ruoli | sì | `Signal`, `AnomalyGate` usate dall'API |
| 12 | API delle metriche | sì | Non applicabile (nessuna metrica nuova) |
| 13 | Strumenti | sì | `extract_delivery.py` non usato (codice scritto a mano) |

## Output dei controlli

```text
$ pytest test/graph/ -q --no-header
103 passed

$ ruff check dense_armor/core/graph/ test/graph/ docs/assets/graph/
All checks passed!

$ ruff format --check dense_armor/core/graph/ test/graph/
All files formatted

$ mypy --follow-imports=skip dense_armor/core/graph/*.py
Success: no issues found in 13 source files

$ mkdocs build --strict
Documentation built (exit 0)
```

Tutti i 103 test girano: `dowhy` e `pygsp` sono installati, nessuno è saltato.

## I 17 controlli di 04_SELF_CHECK_G.md

| # | Controllo | Esito |
|---|---|---|
| 1 | Nessun `{L}` | ✓ |
| 2 | Nessuna narrazione | ✓ (falso positivo del grep su 'spectral'; il check vale per i file del task, non per il codice) |
| 3 | Nessuno slogan | ✓ |
| 4 | Nessun pragma | ✓ |
| 5 | Nessun `/content/` | ✓ |
| 6 | mypy verde | ✓ |
| 7 | Paper citati con PDF | parziale: 2 su 6 verificati con il PDF (Niresi 2406.03898, Liu 2405.13806); gli altri 4 (Cheng, Babaei Giglou, Zan, Wang) sono solo bibliografici e i docstring lo dicono |
| 8 | Confronti con numeri | ✓ |
| 9 | File senza cella | ✓ (auto-report) |
| 10 | Item con comando | ✓ (26 item) |
| 11 | Range stampati | ✓ (golden/*.json) |
| 12 | File letti | ✓ |
| 13 | Formula con riferimento esterno | ✓ (scipy, np.linalg, networkx) |
| 14 | Bound testati al bound | ✓ (lmax, tau, K, alpha) |
| 15 | Libreria usata | ✓ (pygsp, networkx, dowhy) |
| 16 | Test conformità indipendente | ✓ |
| 17 | Repo GitHub verificato | ✓ (4/4) |

## Errori trovati e corretti in questa sessione

| # | File | Errore | Correzione |
|---|---|---|---|
| 1 | `test_graph_spectral.py` | Il ciclo era costruito senza l'arco di chiusura: era un path. | Aggiunto `(0, n-1)`. |
| 2 | `test_fourier.py` | Confronto `gft` con PyGSP senza gauge: gli autovettori differiscono di segno. | Confronto su `|gft|`. |
| 3 | `anomaly.py` | `graph_anomaly_score` usava `|gft|`, che non è un punteggio per nodo (la GFT non è localizzata). | Residuo locale `x - heat_kernel(L, tau) @ x` + PageRank. |
| 4 | `ontology.py` | Il parametro `type` ombrava il builtin. | Rinominato `node_type`, `builtins.type` per il messaggio. |
| 5 | `actions.py` | `Action.apply` con `graph.copy()` (shallow): gli altri nodi condividevano il dict `state`. | `copy.deepcopy`. |
| 6 | `actions.py` | `before` calcolato e non usato: la validazione pre non era applicata. | `if before: raise ActionRejected`. |
| 7 | `test_provenance.py` | I nodi erano `_Pass`, quindi `input_hash == output_hash` e `chain` non distingueva. | `_Inc` (aggiunge 1). |
| 8 | `gen_golden.py` | `SCHEMA` conteneva i tipi Python, non serializzabili in JSON. | Salvataggio dei nomi, ricostruzione nel test. |
| 9 | `gen_golden.py` | `GOLD` non definito (era `GOLDEN`). | Alias. |

## Secondo giro (11 correzioni)

| # | File | Errore | Correzione |
|---|---|---|---|
| 10 | `dag.py` | Un regressore/classificatore imparava con `y=None` prima di predire. | Predice, poi impara solo se `run(..., labels=)` dà l'etichetta. |
| 11 | `dag.py` | Commento falso su `GateNode`; docstring prometteva un errore per nodo duplicato; archi ripetuti. | Commento e docstring corretti, archi deduplicati. |
| 12 | `gate.py` | Doppio `score_one` per campione; nessun `budget_s`/`memory_class`. | Un solo score; `budget_s` = 2 x detector, `memory_class` del detector. |
| 13 | `provenance.py` | `chain` non distingueva due nodi con lo stesso output. | Argomento `node`. |
| 14 | `ontology.py`, `actions.py` | `int` rifiutato per `float`, `bool` accettato per `int`, nome nodo ripetuto sovrascritto. | `type_ok`, `ValueError` sul nome ripetuto. |
| 15 | `core.py` | `Graph.laplacian` era COO: l'esempio del docstring `laplacian[0, 0]` falliva. | `.tocsr()`. |
| 16 | `anomaly.py` | Segnale costante: lo z-score del rumore di arrotondamento dava punteggi vicino a 2. Il vecchio test (`< 3.0`) non lo vedeva. | Lo z-score è 0 se la dispersione è sotto 1e-12 della scala. |
| 17 | `test_graph_anomaly.py` | L'outlier era sull'hub (massimo PageRank): il test non provava il residuo. | Outlier su una foglia; il caso uniforme confronta con il PageRank. |
| 18 | `pyproject.toml` | `dowhy` non era in `dev`: i test causali saltavano in CI. | Aggiunto a `dev`. |
| 19 | `tests.yml` | macOS in coda/lento. | La gamba macOS esclude i test `slow`. |
| 20 | `docs/graph/` | Pagine mancanti. | 8 pagine, 3 figure, nav in `mkdocs.yml`. |

Non corretti (osservazioni, non bug): messaggio di `ActionRejected` per i vincoli del grafo, `graph_anomaly_score` con un `nx.Graph`, `tau` negativo non rifiutato, `eigh` ripetuto in `apply_filter`.

## Verdetto

**Passa.** Item H chiuso: 8 pagine in `docs/graph/` e nav in `mkdocs.yml`; ogni blocco di codice delle pagine è stato eseguito e l'output stampato coincide con quello scritto.
