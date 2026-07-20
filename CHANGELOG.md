# Changelog

Formato basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/).

## [1.1.1]

### Fixed
- **NaN sanato con la media globale della serie invece di una stima locale**
  (`core/hybrid_engine.py`, `armatura.py`): trovato testando l'esempio esatto
  del README con un'installazione pulita da PyPI subito dopo la 1.1.0. Un
  `NaN` veniva sostituito con `np.nanmean` calcolata su TUTTA la serie —
  se uno spike enorme era presente ovunque nella serie (anche lontanissimo),
  contaminava il sostituto di ogni NaN, non solo di quelli vicini allo
  spike. Su serie brevi il sostituto risultava assurdo (2000.81 invece di
  ~1.28 sull'esempio del README, n=6); su serie più lunghe l'effetto si
  attenuava ma non spariva (51.24 invece di ~1.0 su n=200) — non era un
  caso limite di serie corte, era la logica stessa. Se il NaN cadeva anche
  entro il raggio locale di uno spike (fino a 20 passi), la volatilità
  locale gonfiata dallo spike impediva pure al trigger di ricorreggerlo
  verso la baseline. Fix: nuova funzione `_local_nan_fill` — ogni NaN è
  sostituito con la mediana dei vicini FINITI in una finestra locale
  (mediana, non media: robusta a uno spike che capiti nella stessa
  finestra), non con una statistica sull'intera serie. La marcatura come
  anomalia (`anomalie`) era già sempre corretta anche col bug — solo il
  valore restituito in `pulito` per quel punto era sbagliato.

## [1.1.0]

### Changed
- **Motore di `Armatura` sostituito**: `Armatura.analizza()` non usa più
  `AdaptiveSignalStabilizer` (Stadio 1) + `ABCollatz` (Stadio 2). Il gate
  ABCollatz era già documentato in [1.0.10] come matematicamente non
  discriminante (il radicale di una traiettoria di Collatz non ha relazione
  monotona con l'intensità del rumore — verificato con sweep numerico, e una
  sigmoide monotona alternativa, pur costruita e testata, peggiorava
  sistematicamente su 7 scenari reali). Sostituito con un motore a trigger
  binario (`dense_armor/core/hybrid_engine.py`), portato e adattato dalla
  logica phi_ab/vettore-dinamico/trigger già verificata in Dense-Evolution
  (`dense_evolution/healing.py`, oggi in produzione su PyPI dense-evolution
  >= 8.1.9) e in `ia_utils.vector_healing.enhanced_dense_healing_hybrid`.
  Tre adattamenti necessari per il nuovo dominio (serie scalari a scala
  libera, non vettori di embedding normalizzati), tutti verificati con test
  manuali mirati prima di essere accettati: (1) la baseline della finestra
  locale usa i valori già guariti (`out`), non quelli grezzi, altrimenti un
  outlier passato continua a spostare la baseline per `radius` passi dopo
  di sé; (2) l'IPG (gradiente istantaneo) resta sui valori grezzi, non su
  quelli guariti, altrimenti un punto respinto non lascia mai più traccia
  e nessun gradino reale può essere riconosciuto in seguito; (3) la
  distanza tra stati è normalizzata sulla volatilità locale del segnale
  (deviazione standard delle differenze successive nella finestra), non su
  una costante fissa né sulla grandezza assoluta dei valori — altrimenti
  o qualunque cambiamento sopra ~1.4 unità restava permanentemente
  appiattito, o (nel tentativo intermedio) anche uno spike isolato passava
  intatto perché "proporzionalmente coerente" con la propria grandezza.
  Contratto pubblico di `Armatura` invariato (`analizza`, `referto`,
  `referto_json`, `deriva`, stessa firma del costruttore); `K` è ora
  binario (0.0/1.0) invece di una sigmoide continua, coerente con la
  natura binaria del trigger sottostante.
- **`Orca` non è stata toccata in questa release**: continua a usare
  `AdaptiveSignalStabilizer` + `ABCollatz` + `apply_damping_blend`. I
  numeri di robustezza adversarial nel README (PGD/Carlini-Wagner/Fourier)
  restano validi e non ricalcolati, perché misurati su quel motore.

## [1.0.11]

### Fixed
- **`K_anomalous` in `apply_damping_blend` (`dense_armor/core/damping_operator.py`)**:
  la formula `c_anom/(c_anom+diff)` dava K_anomalous massimo a differenza
  quasi nulla (segnale già pulito, dovrebbe damparsi poco) e lo faceva
  decrescere fino al pavimento `_ALPHA=0.25` al crescere della differenza —
  l'opposto di quanto dichiarato nella docstring della funzione. Invertito
  il rapporto (`diff/(c_anom+diff)`). Testato su 30 seed × 4 livelli di
  rumore: il fix batte l'originale 30/30 a rumore medio/alto/molto alto
  (RMSE fino a -0.137), ma peggiora sistematicamente a rumore molto basso
  (limite noto, documentato e coperto da test di regressione dedicato).

### Added
- **`dense_armor/utility/healing.py` — `healing_filter`**: nuovo modulo,
  a sé stante (non integrato in `armatura.py`/`orca.py`), porting
  concettuale da `Dense-Evolution/dense_evolution/healing.py`. A differenza
  di ABCollatz e del damping Stadio 1 (giudicano un punto dal proprio
  residuo istantaneo), classifica ogni punto guardando quanti vicini
  condividono la stessa deviazione dalla baseline locale — deviazione
  isolata = rumore (sostituita con la mediana di una finestra più ampia),
  deviazione sostenuta dai vicini = cambiamento vero (lasciata passare).
  Batte una mediana mobile a raggio 2 su segnali con transizioni vere +
  spike (40/40 e 30/30 su più varianti testate); perde su denoising puro
  di segnali stazionari a basso/medio rumore (limite noto, non è il suo
  caso d'uso primario).

## [1.0.10]

### Fixed
- **Gate ABCollatz dello Stadio 2 (`compute_damping_gating`)**: era
  matematicamente saturo a ~0.85 sempre, indipendentemente dal rumore
  vero — causa: il radicale di un intero (derivato dalla traiettoria di
  Collatz) non ha nessuna relazione monotona con la sua grandezza
  (verificato con sweep numerico: rumore crescente 0->1000 dava
  discrepanze 0, 300, 220, 600, -8, 13920, 0, senza andamento). Non era
  un problema di scala/compressione: l'artefatto restava identico su
  dati grezzi.
- Costruita e testata anche una sigmoide monotona e scala-invariante sul
  rumore relativo, genuinamente discriminante — ma su 7 scenari reali
  (`test/testKalman.py`), sia in modalità cieca sia con riferimento vero
  esplicito, peggiora sistematicamente l'RMSE rispetto al fallback
  costante 0.85 (anche alzando il pavimento minimo fino a 0.84). Nel
  design attuale di Orca il riferimento (Stadio 1) è già una stima
  affidabile: correggere sempre con forza verso di esso batte qualunque
  discriminazione basata sul rumore locale. Ripristinato il fallback
  costante, ora dichiarato esplicitamente invece di emergere per
  accidente da una formula rotta.

Nessun cambio di comportamento a runtime per chi già usa il pacchetto
(RMSE e tempi verificati pressoché identici prima/dopo su tutti gli
scenari di test) — il fix è di correttezza/manutenibilità del codice, non
di funzionalità.

## [1.0.9]

Indagine approfondita partita dalla verifica di uno script esterno che
testava (male, con una reimplementazione a mano non fedele) il fix 1.0.8 —
ha portato a scoprire e risolvere 4 bug distinti in cascata nello scudo
entrata, ciascuno mascherato dal precedente.

### Fixed
- **Compressione log10 (`Orca._execute_4_phase_input_shield`)**: due fattori
  di scala indipendenti (uno da pulito, uno da corrotto) collassavano
  qualunque valore alla stessa magnitudine compressa, distruggendo ogni
  differenza relativa prima ancora che Stadio 1/Stadio 2 la vedessero
  (verificato: anche un pass-through totale senza nessuna vera protezione
  ricostruiva il pulito esatto, allo stesso modo). Ora un solo fattore
  condiviso, derivato dal pulito, applicato a entrambi.
- **Stadio 2 ingannato dal segnale già ammortizzato**: `compute_damping_gating`
  valutava `f1` (output dello Stadio 1) invece del segnale originale,
  sotto-stimando l'anomalia se già parzialmente corretta a monte.
- **Contaminazione post-shock dello Stadio 1**: il motore ricorsivo
  (`AdaptiveSignalStabilizer`) lasciava sempre passare almeno il 25%
  (`k_anom_min`) di un'anomalia enorme nel proprio stato interno, che poi
  decadeva lentamente contaminando per diversi passi anche campioni
  successivi perfettamente normali.
- **Costante `c_anom` fissa non in scala con i dati compressi**: era
  comparabile in grandezza al rumore in spazio compresso, impedendo alla
  soppressione naturale delle anomalie di funzionare anche con lo State
  Flush attivo.

### Added
- **Hard-clamp deterministico**: `raw_noise` (dati grezzi originali, prima
  di qualunque compressione) > 0.05 forza il gate finale a 0.99 (non più
  0.85, per non ereditare il pavimento pensato per i disturbi ordinari).
- **State Flush**: lo stesso segnale hard-clamp, già autorevole, passato
  anche allo Stadio 1 (nuovo parametro opzionale `hard_clamp_mask` su
  `filter_batch_scenarios`/`_process_single_scenario`/`_step_kernel`) —
  azzera il pavimento di guadagno minimo solo per il passo flaggato.
- **`c_anom` scala-adattiva**: proporzionale alla magnitudine locale
  corrente (`prev_filtered`, già nello stato ricorsivo) invece di una
  costante fissa assoluta — si adatta da sola sia a dati grezzi
  (`Armatura`, `filter_data_stream` diretto) sia a dati compressi (`Orca`),
  senza imporre l'assunzione di scala di un solo chiamante nella classe
  generica `AdaptiveSignalStabilizer`.

Verificato con numeri reali: un outlier da 9999 in una serie di valori
~1.3 è ora protetto a 3.61 (indistinguibile dagli altri campioni ~3.4-3.62
dopo il modello), tutti i vicini tornano normali, margine d'errore corretto.
Tutti i parametri nuovi sono opzionali con default che preservano il
comportamento esistente per chi non li usa.

## [1.0.8]

### Fixed
- `Orca._execute_4_phase_input_shield`: lo Stadio 2 (gating ABC/Collatz)
  valutava `f1`, l'output già ammortizzato dallo Stadio 1
  (`AdaptiveSignalStabilizer`), invece del segnale originale — se lo
  Stadio 1 riduceva parzialmente un outlier enorme, lo Stadio 2 poteva
  sotto-stimare quanto fosse anomalo l'input reale. Ora valuta il segnale
  compresso pre-Damping.

### Added
- Sbarramento deterministico (hard-clamping): la soglia di rumore critico
  (0.05) è calcolata sui dati **grezzi originali**, prima di qualunque
  compressione log10 (che rinormalizzerebbe ogni valore individualmente,
  facendo perdere l'intensità reale del rumore). Se superata, il gate
  finale viene forzato alla blindatura massima (0.85), bypassando il
  calcolo ABC/Collatz solo per le macro-anomalie; sotto soglia la
  pipeline sinergica originale resta invariata. Verificato: il leak
  residuo su una macro-anomalia crolla a zero. `@jax.jit`/`jax.vmap`
  intatti, nessuna nuova chiamata eager fuori dal kernel precompilato.

## [1.0.7]

### Fixed
- `evaluate_abc_discrepancy` arrotondava sempre `b` prima di calcolare il
  radicale, anche quando chiamata da `compute_damping_gating_smooth` — che
  genera apposta valori `collatz_wave` non arrotondati tramite
  `execute_collatz_step_smooth`. La variante smooth collassava così
  silenziosamente sulla stessa matematica di quella discreta, un livello
  più in basso di dove il docstring già avvertiva del rischio. Aggiunto
  `smooth_mode: bool = False` (gestito con `jnp.where`, compatibile con
  `jax.jit`/`vmap`); `compute_damping_gating_smooth` ora passa
  `smooth_mode=True`. La variante discreta di default non è cambiata.
  Verificato: prima del fix le due varianti davano risultati identici
  anche su input non interi; ora divergono, come previsto.

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
- **Test JIT flaky sotto pressione di RAM**: `Orca._gc_se_ram_bassa()`
  chiamava `jax.clear_caches()` (svuota la cache di compilazione JIT di
  TUTTO il processo) alla stessa soglia morbida di `gc.collect()` — se la
  RAM libera scendeva anche solo temporaneamente sotto quel margine
  preventivo, la precompilazione fatta in `__init__` veniva vanificata e
  ogni chiamata successiva ricompilava XLA da zero. Ora `jax.clear_caches()`
  scatta solo al limite duro (`min_free_ram`), non al margine preventivo.
  Il test di regressione (`test_orca_protect_and_forward_usa_la_cache_jit_
  non_ricompila_ogni_volta`) ora finge anche RAM abbondante via mock,
  rendendolo indipendente dallo stato reale della macchina/CI.
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
