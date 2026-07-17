# Changelog

Formato basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/).

## [1.0.6]

Risolti i 3 punti lasciati aperti come Known Issues in 1.0.5.

### Changed
- **Logging** (comportamento cambiato, non solo interno): le ~14 chiamate
  `print()` che stampavano lo stato di `Orca.protect_and_forward` ad ogni
  inferenza (`[ORCA] Attivazione...`, `orca/utility/iodat.py`,
  `core/compiler.py` save/load pipeline) sono ora `logging.getLogger(__name__)`.
  **Di default sono silenziose** (nessun handler configurato dalla libreria,
  convenzione standard) — per vederle di nuovo: `logging.basicConfig(level=logging.INFO)`
  prima di usare la libreria. Le ~35 `print()` rimaste (in `armatura.py`
  metodo `referto()` e CLI `main()`, `utility/anwav.py`, `utility/diagnostic.py`)
  **non sono state toccate**: sono referti/output voluti quando l'utente
  chiama esplicitamente quelle funzioni, non rumore di background —
  convertirle avrebbe reso silenzioso per default uno strumento il cui
  scopo e' stampare un referto.

### Fixed
- **Copertura docstring/type hint**: dal 53.9%/58.8% (55/60 su 102
  funzioni) al 100% — tutte le funzioni/metodi pubblici e privati hanno
  ora una docstring one-line e annotazioni di tipo su argomenti/ritorno.
- **`except Exception:` generico**: ristretto a eccezioni specifiche dove
  identificabili — `core/memory.py` (`subprocess`/parsing di `nvidia-smi`
  a `(CalledProcessError, TimeoutExpired, FileNotFoundError, OSError,
  ValueError)`; query VRAM `jax.devices()` a `(RuntimeError, AttributeError)`),
  `core/noise.py` (`jax.default_backend()` a `RuntimeError`). Lasciati
  broad-by-design, ma documentati con un commento: `memory.py` (pulizia
  cache JIT best-effort, ora anche loggata a livello debug invece di
  `pass` silenzioso) e `utility/resonance_search.py::smoke_test` (per
  definizione uno smoke test deve catturare qualunque fallimento).

## [1.0.5]

### Fixed
- `pyproject.toml`: campo `license` come tabella TOML deprecata da
  setuptools (avviso ad ogni build, sarebbe diventato errore bloccante dal
  18 febbraio 2027) sostituito con stringa SPDX standard `"BUSL-1.1"` +
  `license-files = ["LICENSE.md"]`. Richiede `setuptools>=77` (bumpato in
  `build-system.requires`).
- Residuo del vecchio nome del progetto ("sentinel02") nei commenti di
  intestazione di `dense_armor/core/damping_operator.py` e
  `dense_armor/utility/metro.py` — aggiornati al percorso corrente
  (`shield_/...`), nessun impatto funzionale.

### Known Issues (non risolti in questa versione)
- **Logging**: 49 chiamate a `print()` sparse nel codice per i messaggi
  `[ORCA] ...`. Esiste gia' `dense_armor/core/logger.py` ma non e'
  importato da nessun altro modulo — chi integra la libreria non puo'
  abbassare la verbosita', reindirizzare su file o silenziare i log senza
  modificare il sorgente.
- **Copertura docstring/type hint**: su 102 funzioni/metodi totali, 55
  hanno una docstring (53.9%) e 60 hanno almeno un type hint tra
  argomenti/valore di ritorno (58.8%).
- **`except Exception:` generico**: 5 punti che catturano tutto
  silenziosamente — `core/memory.py` (righe 56, 72, 96), `core/noise.py`
  (riga 43), `utility/resonance_search.py` (riga 73). Da rivedere caso per
  caso: quali eccezioni sono davvero previste (es. file mancante) e quali
  nascondono bug che dovrebbero propagarsi.

## [1.0.4]

### Changed
- Nessuna modifica al codice rispetto a 1.0.3 — solo bump di versione,
  perche' 1.0.3 risultava gia' occupata su PyPI da un caricamento
  precedente.

## [1.0.3]

### Fixed
- `filter_data_stream` (motore di `AdaptiveSignalStabilizer`) ricompilava
  l'intero kernel JIT ad ogni chiamata invece di riusare quello
  precompilato — kernel 1D spostato in `__init__`, soglie/gain passati
  come argomenti jit invece che letti da `self.*` dentro la funzione.
- `jnp.insert` chiamato fuori da `jax.jit` subito dopo lo scan (stesso
  problema, punto diverso) — sostituito con `np.insert` post-conversione.
- `Orca._execute_4_phase_input_shield` / `_execute_4_phase_output_shield`
  eseguivano la loro pipeline elementwise (`jnp.where`/`jnp.isnan`/
  aritmetica) in modalita' eager, fuori da qualunque `jax.jit` — estratta
  in due kernel dedicati precompilati una sola volta in `__init__`.
  Eliminate anche chiamate ridondanti duplicate a
  `filter_batch_scenarios`/`compute_damping_gating`.
- Risultato: 3 chiamate a `protect_and_forward` passano da ~11s a ~0.03s
  (zero ricompilazioni XLA residue a regime, verificato con `cProfile`).
- CI: workflow lanciava ancora `pytest tests/` dopo la rinomina della
  cartella in `test/` — corretto in `.github/workflows/tests.yml` e
  `README.md`.

### Changed
- Cartella dei test rinominata da `tests/` a `test/`.

## [1.0.1]

### Fixed
- **Bug critico**: `evaluate_abc_discrepancy` (radicale ABCollatz)
  calcolava il radicale su `a*b*c` invece che sul valore generato dalla
  traiettoria di Collatz — la traiettoria era di fatto ignorata,
  producendo lo stesso risultato indipendentemente dal valore Collatz
  reale (verificato: b=7 e b=999999 davano discrepanze identiche prima
  del fix).
- Dipendenza core mancante `psutil` aggiunta a `pyproject.toml` (trovata
  testando l'installazione del wheel in un venv pulito).

### Added
- Suite di test pytest (`test/`) e CI GitHub Actions
  (`.github/workflows/tests.yml`), matrice Python 3.10/3.11/3.12 su
  ubuntu/windows + controllo build/metadata.
- Istruzioni per lanciare la suite di test in locale nel README.

## [1.0.0]

Prima pubblicazione.
