# Changelog

Formato basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/).

## [Unreleased]

### Added
- **`utility/streaming.py`** (`StreamingDeviationDetector`,
  `MultiChannelStreamingDeviationDetector`, `classify_segments_
  multichannel`): porta a latenza zero solo la meta' causale di
  `classify_segments` (arbiter.py) -- il flag di deviazione per-punto,
  non l'etichetta finale spike/regime, che guarda `radius` punti avanti
  alla fine di una sequenza deviante e resta una domanda batch/offline
  per design, non una svista. Buffer semplice (deque, O(span)), non una
  struttura a due heap -- misurato a ~18.6kHz sostenibile per le
  finestre gia' usate in tutto il progetto (10-100 punti), oltre 180x
  il tasso reale di un anello di controllo robotico (30-100Hz). Il
  supporto multi-canale rimuove il bisogno di un ciclo manuale
  per-canale (giunti di un braccio robotico, assi di un IMU), ogni
  canale con la propria finestra di riferimento indipendente. Promosso
  da Dense-Evolution-Discovery (Esperimenti 48-49) dopo validazione su
  due domini fisici reali indipendenti (braccio robotico teleoperato
  SO-101, IMU umano reale UCI HAR) -- stessa disciplina di promozione
  gia' usata per `stable_frame_filter.py`. Corrispondenza bit-per-bit
  con `classify_segments` verificata direttamente, non assunta.

## [1.1.13]

### Fixed
- **`utility/cusum.py` (`cusum_detector`)**: il default `h=5.0` ("classico
  da manuale", mai validato contro una lunghezza di serie pratica) dava
  un tasso di falso allarme a livello di STREAM del 100% (`adaptive`) /
  85% (`fixed`) su una serie stabile di 1000 campioni -- invisibile al
  test esistente, che controllava solo il tasso di falso allarme per
  PUNTO (molto più basso e fuorviante). Trovato come sottoprodotto
  diretto di un porting parallelo dello stesso algoritmo per
  online-ml/river, dove lo stesso identico problema è emerso per primo.
  Nuovo default `h=20.0`: 3.5% (`adaptive`) / 15.5% (`fixed`) sulla
  stessa serie di 1000 campioni. `fixed` resta strutturalmente più
  esposto (il suo riferimento non si aggiorna mai, quindi una stima di
  warmup sfortunata non si autocorregge) -- non un difetto di taratura,
  conseguenza diretta di cosa `fixed` è pensato per fare. Nuovo test di
  regressione (`test_default_h_keeps_stream_level_false_alarm_rate_low_
  on_a_practical_length`) per non ricadere silenziosamente nello stesso
  problema. Nessun benchmark preregistrato è stato toccato: tutti e
  quattro (`test_benchmark_v0_runtime_behavioral_drift.py`,
  `test_benchmark_v2_1_ablation.py`, `test_benchmark_v2_1_log_
  onesided.py`, `test_benchmark_v2_agent_runtime.py`) passano `h=5.0`
  esplicitamente nel proprio `CUSUM_KW` congelato, indipendentemente dal
  default della funzione.

### Added
- **`utility/stable_frame_filter.py` (`velocity_gated_stable_mask`)**: filtro
  che restringe l'analisi di un segnale ai punti dove un segnale di
  riferimento COMPANION indica uno stato stabile/lento -- evita che un
  detector interpreti come anomalia un transitorio spiegato da un
  confondimento noto e indipendente (es. il comando/leader di un braccio
  robotico che si muove veloce, causando un lag di tracking normale).
  Parametro `already_rate` per riferimenti gia' di tipo velocita' (es.
  magnitudine giroscopica) invece che posizione -- trovato necessario da
  un fallimento reale su un secondo caso, non aggiunto preventivamente.
  Promosso da Dense-Evolution-Discovery dopo validazione su due domini
  fisici reali indipendenti (braccio robotico teleoperato SO-101, IMU
  umano reale) -- stessa disciplina di promozione gia' usata per
  `one_sided_upper_filter`.

## [1.1.12]

### Added
- **`utility/cusum.py` (`cusum_detector`)**: canale complementare a
  `classify_segments` per il drift lento/sostenuto -- CUSUM a due code
  (Page 1954), modalita' `adaptive` (default, finestra causale scorrevole,
  dichiarata onestamente come variante non identica allo schema originale
  a riferimento fisso) e `fixed` (lo schema classico). Gestione NaN/Inf
  esplicita. Nato da un gap reale misurato in
  `test/test_benchmark_v0_runtime_behavioral_drift.py`:
  `classify_segments` rileva un glitch transitorio al 100% (latenza 0) ma
  solo il 9-27% di un drift graduale nella finestra di transizione.
- **`utility/one_sided.py` (`one_sided_upper_filter`)**: filtro a una
  coda componibile con `classify_segments`/`cusum_detector` -- per segnali
  dove solo un AUMENTO e' significativo (latenza, error rate). Riduce i
  falsi positivi misurati su telemetria reale di un agente Qwen2 1.8B
  (via Ollama) dal 22.5% al 12.5% (`classify_segments`) e dal 17.5% al
  10.0% (`cusum_detector`), e il falso rifiuto su un cambio di
  comportamento legittimo dal 24.0% al 4.0% su entrambi -- al costo reale
  di una sensibilita' ridotta al drift persistente (non nascosto: vedi
  `test/test_benchmark_v2_1_ablation.py`). Un log-transform provato in
  parallelo non ha portato benefici misurabili ed e' stato scartato.

### Investigated (percorso di validazione, non solo il risultato finale)
- Benchmark sintetico congelato (`test/test_benchmark_v0_runtime_
  behavioral_drift.py`): protocollo preregistrato, 4 condizioni (baseline,
  drift, glitch, test negativo), mai ritoccato dopo aver visto i
  risultati.
- Benchmark su agente reale (`test/agent_v2/`, `test/test_benchmark_v2_*`):
  prima validazione non sintetica -- loop minimale con Qwen2 1.8B via
  Ollama (tool-calling manuale, il modello non restituisce `tool_calls`
  strutturati nonostante il tag "tools"), 4 scenari con ground truth
  esatta. Trovato: la latenza reale ha una coda molto piu' pesante del
  rumore gaussiano sintetico, da cui il filtro one-sided sopra.

## [1.1.11]

### Investigated (nessun candidato promosso)
- **`compute_damping_gating_smooth` (Collatz continuo), mai benchmarkato,
  ora chiuso**: il suo stesso docstring rimandava a
  `test/test_collatz_smooth_experiment.py` per il confronto misurato --
  file mai esistito. Scritto ora, gira `Orca` per intero sui 7 scenari di
  `test/testKalman.py` una volta con il gate costante (default) e una con
  la variante smooth. Risultato: smooth vince 0/7, pareggia 6/7 (RMSE
  identico entro 1e-4), perde su "Stealth sub-soglia" (RMSE 0.0029 vs
  0.0012, oltre il doppio). Non promosso -- docstring aggiornato con il
  risultato reale al posto del rimando a un file inesistente.
- **`hybrid_shield` (phi_ab, gia' vendorizzato per Armatura) come possibile
  gate per Orca**: non e' uno scambio diretto (motore binario a ciclo su
  tutta la serie, non un gate continuo per elemento), quindi confrontato
  come pipeline completa (`test/test_hybrid_shield_vs_orca_experiment.py`)
  sugli stessi 7 scenari. Risultato misto, non un vincitore netto:
  vince 4/7 su impulsi isolati/collasso a zero/buco NaN (in 3 casi RMSE
  ~0.0000, quasi perfetto), ma perde nettamente su segnale sinusoidale
  strutturato e rumore Cauchy pervasivo (RMSE 2-3x peggiore di Orca) --
  ottimo per anomalie isolate, pessimo quando il cambiamento e' diffuso e
  genuino, non un'anomalia da respingere.
- **`pressure_valve` (JSD-adattivo, gia' in `robust_filters.py`, mai
  collegato a nessuna pipeline) come terzo candidato**: stesso confronto
  a 7 scenari (`test/test_pressure_valve_vs_orca_experiment.py`). Vince
  1/7 ma in modo netto: RMSE 0.0000 esatto sulla rottura di livello
  permanente (l'unico dei tre a riconoscere un vero cambio di regime),
  ma fallisce clamorosamente sugli impulsi isolati (RMSE 86.6 su
  "impulsi alternati", peggio di non fare nulla). Causa reale trovata
  (non un'ipotesi): con 3 picchi ravvicinati in una finestra di 21 punti,
  mediana/IQR e mediana/MAD restano ESATTAMENTE 0 (18/20 punti identici,
  matematicamente corretto per stimatori con breakdown ~25-50%), quindi
  vengono scartati e resta solo media/std -- non robusta, contaminata
  dagli stessi picchi che dovrebbe rilevare. Fix parziale applicato: la
  finestra locale ora esclude il punto stesso (come gia' faceva solo il
  sotto-passo sigma-clipping), corretto in linea di principio ma
  verificato non sufficiente quando PIU' outlier restano comunque vicini
  tra loro nella stessa finestra (nessun cambiamento misurabile sui 7
  scenari). Un M-estimator di Huber (IRLS, vedi Sung et al. 2019,
  arXiv:1912.04982, ricerca bibliografica reale via RAG locale) e' stato
  provato come alternativa piu' sofisticata e converge ANCH'ESSO a scala
  0, per lo stesso motivo di fondo (la sua scala e' a sua volta una MAD
  dei residui). Conclusione onesta: non e' un problema di scelta
  dell'estimatore, e' che dividere per una scala locale legittimamente
  zero (baseline quasi piatta) e' indefinito per costruzione, qualunque
  stimatore si usi. Non risolto -- vedi il "LIMITE NOTO" nel docstring di
  `pressure_valve` per la via di fix non ancora tentata (scala di
  fallback dalla finestra di riferimento piu' ampia).
- **Conclusione dei tre esperimenti**: nessuno dei tre candidati batte in
  modo netto il gate costante 0.85 di Orca su tutti gli scenari --
  ognuno (incluso il default) ha una nicchia precisa (impulsi isolati,
  segnale diffuso/rumore pervasivo, cambio di regime permanente). Il
  gate costante resta quello di default; nessuna promozione fatta.

### Fixed
- **`core/vector.py`, `run_parallel_scenarios` ricompilava ad ogni chiamata**: `jax.jit(jax.vmap(...))`
  veniva ricostruito DENTRO il metodo ad ogni invocazione -- una nuova closure Python ad ogni
  chiamata, quindi un nuovo oggetto `jax.jit` la cui cache di compilazione non poteva mai fare
  hit con la precedente. Il "warm run" restava bloccato a ~130ms invece di scendere ai pochi ms
  attesi (la diagnosi originale, un report Colab, puntava al cast `np.array` come colpevole --
  verificato e smentito: il vero costo era la ricompilazione ad ogni chiamata). Fix: motore
  jit+vmap spostato a livello di modulo, costruito una sola volta; `base_state` passato come
  argomento tracciato invece che chiuso in closure, cosi' valori diversi non forzano una nuova
  compilazione. Verificato: da un plateau di ~140ms/chiamata a ~2.4ms/chiamata (~55x) su chiamate
  ripetute al metodo pubblico.
- **`utility/orca.py`, riferimento pulito che attraversa lo zero esplodeva la compressione log10**:
  un valore pulito quasi-zero (anche solo un residuo di floating point, mai esattamente `0.0` --
  es. `sin(4π) ≈ -4.9e-16`) faceva esplodere il fattore di scala condiviso `fact_shared`. Siccome
  lo stesso fattore si applica anche al corrotto (che NON e' vicino a zero), il suo controvalore
  compresso schizzava a ordini di grandezza sopra i punti vicini (misurato: da `~1e-4` tipico a
  `~2e9` in un caso), sballando la memoria causale di `AdaptiveSignalStabilizer` per l'intero
  batch, non solo per quel punto. Fix a due livelli: (1) l'esponente non viene piu' amplificato
  oltre la scala target (`val_e`); (2) il valore compresso del corrotto resta comunque limitato a
  un multiplo ampio (20x) della scala target, indipendentemente dalla causa esatta del salto.
  Verificato: MSE su un riferimento sinusoidale che attraversa lo zero da ~9 (peggio della
  modalita' cieca sullo stesso segnale) a ~0.02.

### Changed
- **`core/engine.py`, naming e precisione non armonizzati col resto della libreria**: le variabili
  interne di `AdaptiveSignalStabilizer` (`_step_kernel` e la variante storica `_legacy_step`) si
  chiamavano `phi_ab`/`v_dyn`/`trigger` -- stessi nomi di `core/hybrid_engine.py` (motore di
  Armatura) per una formula matematicamente diversa. Rinominate in modo non ambiguo
  (`coherence_signal`/`coherence_mix` e `coherence_legacy`/`dynamic_shift_legacy`/
  `is_anomaly_legacy`). Il file usava anche `float32` esplicito ovunque, mentre il resto della
  libreria richiede `jax_enable_x64` e gira in `float64` -- i commenti giustificavano il
  `float32` con una "compatibilita' di esportazione binaria nativa" che non esiste da nessuna
  parte nel repo (verificato con una ricerca sull'intero codebase). Convertito tutto a `float64`,
  rimossi i commenti fuorvianti. Suite completa (189 test, inclusi i 5 file adversarial
  `test_bound*.py`) invariata prima e dopo entrambe le modifiche.

### Added
- **`Orca` usa ora `UniversalMemoryGuard` invece di un check RAM ad-hoc**: `_gc_se_ram_bassa()`
  duplicava a mano un check `psutil` gia' fatto meglio da `core/memory.py` (che copre anche la
  VRAM, mai controllata prima). Cambio di comportamento onesto: sotto la soglia critica ora
  solleva `MemoryPressureError` invece di continuare in silenzio -- fail-fast con un errore
  leggibile invece di un OOM qualche chunk piu' avanti.
- **`Orca` ricorda riferimenti puliti passati e li richiama via `apply_fast_resonance`**: in
  modalita' cieca (`x_reference=None`), prima di ricadere sulla stima locale (`_blind_reference`),
  Orca cerca nella propria memoria (per shape, fino a `reference_memory_size=32` per default) un
  riferimento gia' visto simile all'input corrotto corrente (soglia `reference_recall_min_score=
  0.90`, calibrata empiricamente: segnali correlati restano sopra 0.95 anche con rumore
  moderato, segnali scorrelati non superano ~0.75). Banca vuota o nessun match abbastanza simile
  -> comportamento identico a prima.
- **`chunk_threshold` di `Orca` auto-tarato su `AIHardwareProfiler`**: default cambiato da un
  intero fisso (`1_000_000`, uguale per qualunque host) a `None`, che auto-deriva il valore
  scalando proporzionalmente a `max_tensor_dim` dell'host corrente (RAM/GPU-TPU). Su un host
  "base" (RAM<12GB, nessuna GPU/TPU) il risultato e' identico al vecchio default fisso -- nessuna
  regressione silenziosa sul caso comune. Un valore esplicito bypassa l'auto-tuning senza nemmeno
  istanziare il profiler.
- **`Orca(use_arbiter=True)`, opzione nuova, default `False`**: instrada ogni
  punto verso il correttore giusto invece di forzarne uno solo su tutta la
  riga -- nato dai tre esperimenti sopra (nessun gate unico batteva gli
  altri due ovunque). Nuovo modulo `utility/arbiter.py`: classifica ogni
  punto come 'clean'/'spike'/'regime' confrontandolo con una finestra di
  riferimento larga e CAUSALE (solo punti prima di i, mai dopo -- stessa
  scala della "molla" JSD di `pressure_valve`, non la finestra locale
  stretta che si e' vista collassare sopra), poi la lunghezza e la
  coerenza interna della sequenza di punti devianti decide impulso
  isolato vs cambio di regime sostenuto. 'spike' -> rigetto duro (mediana
  della finestra larga); 'regime' -> valore grezzo (fiducia, stessa
  filosofia del gate costante); 'clean' -> quello che lo scudo standard a
  4 fasi ha gia' prodotto (il suo smorzamento morbido, non un
  pass-through, altrimenti un segnale continuo non anomalo come
  un'oscillazione strutturata verrebbe lasciato passare intatto).
  Verificato su tutti i 7 scenari di `test/testKalman.py`
  (`test/test_arbiter_orca_integration.py`): mai peggiore del default
  (`use_arbiter=False`, garanzia testata), migliore su 5/7 (impulsi
  isolati, sub-soglia, Cauchy, rottura di livello, buco NaN), pari sul
  collasso a zero, identico sul segnale strutturato (ricade correttamente
  sullo scudo standard, com'e' giusto: e' smorzamento continuo, non
  un'anomalia discreta). Popola anche `Orca.etichette_arbitro`/
  `incertezza_arbitro` (l'incertezza della classificazione, non la
  dimensione della correzione -- segnale complementare a
  `margine_ingresso`, non ridondante) e
  `Orca.tipi_corruzione_visti(slice_shape)`, un Counter dei tipi di
  corruzione gia' osservati per quella shape quando `x_reference` era
  noto -- memoria separata dalla banca dei riferimenti puliti gia'
  esistente, che ricordava solo la FORMA, non il TIPO di corruzione.
  CRONOLOGIA di un limite trovato e risolto nella stessa sessione: la
  prima versione usava una finestra di riferimento SIMMETRICA, che a
  cavallo di una rottura di livello permanente mescolava vecchio e nuovo
  livello nella propria scala, non rilevando affatto la transizione
  (deviazione sempre 0) -- un primo confronto isolato aveva scambiato
  questo per un successo (RMSE 0.0), ma era un artefatto: i dati grezzi
  di quello scenario di test coincidono GIA' col target, quindi "passa
  tutto grezzo" da' RMSE 0 indipendentemente dall'etichetta. Attraverso
  Orca la mancata rilevazione si vedeva davvero: nessun miglioramento
  (9.4007 identico con o senza l'arbitro). Passando alla finestra
  causale, la rilevazione e' diventata reale (indici 60-74 etichettati
  'regime', esattamente la transizione vera a 60) e l'RMSE reale scende
  a 6.0449 -- vedi `utility/arbiter.py` per il dettaglio completo.
- **`dense_armor.mcp_server`, nuovo (extra `[mcp]`)**: server MCP con 5
  tool (`dense_armor_health`, `dense_armor_clean_signal`,
  `dense_armor_detect_anomalies`, `dense_armor_robust_filter`,
  `dense_armor_heal_series`) cosi' un agente puo' ripulire una serie
  senza scrivere Python. Diretto e in-process, non un proxy HTTP verso
  un kernel separato come l'adattatore di Dense-Evolution -- Dense-Armor
  non ha una web UI da condividere. Vive apposta sotto
  `dense_armor.mcp_server`, non un semplice `mcp_server`: con
  Dense-Evolution installato nello stesso ambiente (il suo adattatore si
  chiama esattamente `mcp_server`), un nome non annidato e' una
  collisione VERA, osservata direttamente (`from mcp_server.server
  import mcp` risolveva ai 25 tool di Dense-Evolution invece dei 5 di
  qui, a seconda dell'ordine di installazione), non un'ipotesi --
  scoperta e corretta prima che questo finisse pubblicato. Nuovo script
  console `dense-armor-mcp`, separato da `dense-armor` (la CLI di
  Armatura, invariata). Un bug reale trovato e corretto nello sviluppo:
  lo schema Pydantic di `values` dichiarava supporto NaN nel docstring
  ma rifiutava silenziosamente `null` JSON dentro la lista (`List[float]`
  invece di `List[Optional[float]]`). Verificato end-to-end via
  `mcp.call_tool()` reale (non solo le funzioni Python dirette) in
  `test/test_mcp_server.py`, 222/222 test verdi in totale.

## [1.1.9]

### Fixed
- **`core/compiler.py` ricompilava ad ogni chiamata**: `run_dynamic_pipeline`,
  `run_pipeline_with_chunking` e `compute_gradients` definivano il proprio kernel `@jax.jit`
  DENTRO il metodo -- un nuovo oggetto `jax.jit` (quindi una cache di compilazione vuota) ad
  ogni singola chiamata, mai riuso della compilazione XLA anche a parita' di shape tra una
  chiamata e la successiva. Stesso bug gia' trovato e risolto nel motore di
  `AdaptiveSignalStabilizer` (vedi [1.0.3]), mai corretto qui. Trovato scrivendo un test che
  doveva dimostrare l'utilita' reale di `PipelineProfiler` (warm-up separato dal regime): con
  il bug, warm-up e regime erano praticamente identici. Fix: i tre kernel spostati a livello di
  modulo, compilati una sola volta. Verificato: 375ms -> 0.2ms a regime sulla stessa pipeline
  (>1700x), test di regressione aggiunto.
- **`utility/anwav.py` andava in crash su console Windows non-UTF-8**: i verdetti su volume
  "troppo spinto"/"troppo silenzioso"/picco a rischio usavano emoji (`❌`, `⚠️`) nei `print()` --
  su cp1252 (l'encoding di default della console Windows) sollevavano `UnicodeEncodeError`,
  quindi qualunque uso reale della funzione fuori da pytest (che forza UTF-8 in
  `test/conftest.py`) su un file rumoroso o silenzioso andava in crash. Rimosse le emoji dai
  messaggi.
- **Link rotto in questo stesso CHANGELOG bloccava il deploy del sito in silenzio**: la voce
  1.1.8 linkava `[Toolkit](api/toolkit.md)`, un percorso relativo valido solo dentro `docs/` --
  ma `CHANGELOG.md` vive nella root del repo e viene incluso cosi' com'e' in
  `docs/changelog.md`. `mkdocs build --strict` rifiutava giustamente la build, e i due push
  precedenti non erano mai arrivati sul sito pubblico (la sezione toolkit di
  `getting-started.md` e gli esempi in `docs/api/toolkit.md` erano gia' su GitHub ma non ancora
  online). Corretto con l'URL completo del sito; verificato dal vivo che il sito pubblico ora
  li mostra davvero.

### Changed
- **`docs/api/toolkit.md` non vende piu' ogni modulo allo stesso modo**: aggiunte note oneste
  dove l'utilita' e' effettivamente sottile (`TensorVault` sono matrici scrivibili in una riga,
  l'euristica RAM di `AIHardwareProfiler` e' arbitraria, `BitwisePermutationEngine` fa un solo
  swap controllato per chiamata non una permutazione generale, l'update di
  `ParametricScenarioSimulator` e' una EMA fissa non un modello configurabile, i formatter di
  logging sono wrapper sottili, `StochasticAdversarialNoise` si sovrappone alla suite
  adversarial reale invece di aggiungerci qualcosa) e rafforzate quelle con prove reali
  (`PipelineProfiler` ha scoperto il bug sopra; i preset producono filtraggio misurabilmente
  diverso; `kappa` in `apply_fast_resonance` modula davvero il punteggio, non e' cosine
  similarity travestita).

### Added
- **Test che dimostrano l'utilita' dichiarata dei moduli, non solo che "non crashano"**: preset
  diversi (`SIGNAL_STABILIZER_PRESETS`) producono un filtraggio misurabilmente diverso quando
  collegati a `AdaptiveSignalStabilizer`; `kappa` in `apply_fast_resonance` modula davvero il
  punteggio; il warm-up di `PipelineProfiler` e' davvero piu' lento del regime (il test che ha
  scoperto il bug di ricompilazione sopra).
- **`docs/api/toolkit.md`**: ogni modulo del toolkit ha ora un esempio di codice reale
  verificato (non solo la firma auto-generata dai docstring), preceduto da cosa fa e a cosa
  serve.

## [1.1.8]

### Fixed
- **`docs/getting-started.md` non spiegava come usare il toolkit**: la pagina aveva un
  quickstart per Orca, Armatura e robust_filters, ma per il toolkit (14 moduli, aggiunti in
  1.1.6-1.1.7) solo un link nudo nella home del sito, nessun esempio eseguibile. Aggiunta una
  sezione "Standalone toolkit" con due esempi reali verificati (`DynamicAICodegen`,
  `UniversalMemoryGuard`) e link a [Toolkit](https://tatopenn-cell.github.io/Dense-Armor/api/toolkit/)
  per il resto.

## [1.1.7]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca` in questa versione. I 14 moduli
del toolkit aggiunti a 1.1.6 sono ora davvero pubblici (documentati sul sito, `show_source:
true`), quindi tutto quello che c'era di poco professionale nel codice sorgente -- non solo nel
README -- ora si vede.

### Added
- **Sito**: nuova pagina `docs/api/toolkit.md` (mkdocstrings, pesca dai docstring reali) per i
  14 moduli aggiunti in 1.1.6 -- prima invisibili sul sito, l'API Reference copriva solo i 5
  moduli dello scudo. Linkata da `mkdocs.yml`, `docs/index.md`, `docs/api/index.md`.
- **README**: la sezione `$ toolkit --standalone` (prima solo un blocco di import con commenti)
  riscritta con prosa reale, raggruppata per categoria (pipeline/chunking, hardware/profiling,
  tensori/configurazione, logging/provenance, audio/I/O dati, ricerca per similarità).

### Changed
- **Rimosso "Sentinel"/branding residuo dal codice sorgente** dei 14 moduli toolkit e dei 5
  moduli originali dello scudo (`armatura.py`, `core/damping_operator.py`, `utility/metro.py`,
  `utility/orca.py` -- mancati nel passaggio di de-branding di 1.1.5, che si era fermato al
  testo rivolto all'esterno README/pyproject/sito, senza scendere nei singoli docstring con
  `show_source: true`): banner "SENTINEL ENTERPRISE... BEAST MODE" e claim falsi di "logica
  quantistica" in `core/chunk.py`, blocco "Fix applicati/BILANCIAMENTO AUREO" (contenuto da
  changelog finito nel docstring) in `core/compiler.py`, riferimenti a file interni inesistenti
  (`simulator.py`, `NoiseModel`, `test21.py`/`main.py`) in `core/vector.py`/`core/noise.py`/
  `core/chunk.py`, intestazioni "Sentinel Metrology Framework" e vecchio percorso interno
  `shield_/...` nei 4 moduli originali.
- **Simboli pubblici rinominati** (uniche API realmente rotte da questa release, erano appena
  diventate pubbliche in 1.1.6): `execute_pipeline_beast_mode` -> `execute_pipeline_chunked`
  (`core/chunk.py`); `SENTINEL_PRESETS` -> `SIGNAL_STABILIZER_PRESETS` (`core/preset.py`);
  `get_enterprise_logger` -> `get_json_file_logger`, nome logger/file di default
  `sentinel*`/`sentinel_dashboard.log` -> `dense_armor*`/`dense_armor.log`, campo JSON
  `"framework": "Sentinel-TensorFlowEngine"` -> `"dense-armor"` (`core/logger.py`);
  `AIEngineVisualizer.ENGINE_SIGNATURE`/intestazione del trend report non dicono più
  `TensorFlowEngine-Sentinel` (`core/visualizer.py`).
- **`utility/orca.py`**: rimossa una `sys.path.insert(...)` morta (residuo di quando il modulo
  girava come script standalone fuori dal pacchetto -- gli import relativi non ne hanno
  bisogno, e nient'altro nel file usava `sys`/`os` dopo quella riga) e i relativi import
  inutilizzati.
- **`armatura.py`**: docstring e messaggi di uso CLI aggiornati da `python armatura.py`/`from
  armatura import Armatura` (invocazione da script standalone) a `python -m dense_armor`/`from
  dense_armor import Armatura` (pacchetto installato).

## [1.1.6]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca` in questa versione. Partita
dalla richiesta di portare la coverage Codecov al 100%: 14 file in `core/`/`utility/`
risultavano a 0% di coverage. Prima di cancellarli come "codice morto" (la prima ipotesi),
trovato `dense_armor/README.md` (riferimento interno) che li documenta tutti come componenti
reali e intenzionali — indagine invertita, modulo per modulo, invece di procedere con la
cancellazione.

### Fixed
- **`core/__init__.py` vuoto**: il vero file di inizializzazione del pacchetto (import/export
  di 9 classi: `AIHardwareProfiler`, `StochasticAdversarialNoise`, `UniversalMemoryGuard`,
  `MemoryPressureError`, `ParametricScenarioSimulator`, `BitwisePermutationEngine`,
  `DynamicAICodegen`, `CMD_MAP`, `TensorVault`, `AIEngineVisualizer`, `PipelineProfiler`)
  esisteva ma era salvato come `core/init.py` (nome sbagliato, mai eseguito da Python) — il
  file col nome giusto era vuoto. Nessuna di queste classi era importabile da
  `dense_armor.core.*` nonostante il codice reale esistesse. Contenuto spostato nel file col
  nome corretto, `init.py` eliminato.
- **`core/preset.py`**: l'intero dizionario `SENTINEL_PRESETS` (4 profili calibrati) era
  definito due volte verbatim nello stesso file. Rimossa la duplicazione.
- **`utility/iodat.py`**: `lodat()` ritornava in silenzio un tensore di dati casuali finti se
  il file richiesto non esisteva (solo un log di warning, nessun errore) — ora solleva
  `FileNotFoundError`.

### Added
- **Test reali per tutti e 14 i moduli** precedentemente a 0% di coverage: `chunk.py`,
  `compiler.py`, `memory.py`, `preset.py`, `logger.py`, `tensor.py`, `vector.py`, `noise.py`,
  `profiler.py`, `visualizer.py` (`core/`); `anwav.py`, `diagnostic.py`, `iodat.py`
  (`utility/`, con file WAV/HDF5/NetCDF reali generati nei test, non mock) — più le righe
  ancora scoperte in `resonance_search.py`. Coverage totale del pacchetto: 54% -> 91%.
- **README**: nuova sezione `$ toolkit --standalone` che documenta questi moduli come seconda
  parte del pacchetto, indipendente dallo scudo anomalie di Armatura/Orca.

## [1.1.5]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca`/`robust_filters` in questa
versione -- solo infrastruttura di test, documentazione e metadati.

### Added
- **Suite di test adversarial reale** (`test/test_boundA-E.py`): i test dietro la tabella
  "robustezza adversarial" del README (PGD/BIM/MI-FGSM, affine/elastica, Carlini-Wagner L2/Linf,
  DeepFool, Fourier a banda estesa) esistevano solo in un progetto precursore ("sentinel3", mai
  parte di questo repository) da cui `AdaptiveSignalStabilizer` era stato estratto. Riportati
  qui con una modifica sostanziale: l'import punta ora a
  `dense_armor.core.engine.AdaptiveSignalStabilizer` (il pacchetto reale) invece del modulo
  locale standalone del progetto precursore -- stessa classe, stessa API, ri-eseguiti e
  confermati riprodurre esattamente i numeri gia' nel README (PGD 0.0130, BIM 0.0625, MI-FGSM
  0.0784, C&W L2 78.96%, C&W Linf 64.39%, DeepFool 78.79%, Fourier 99.77%+): la tabella e' ora
  una misura reale e riproducibile, non piu' un dato non tracciato.
- **Sito di documentazione** (GitHub Pages, <https://tatopenn-cell.github.io/Dense-Armor/>):
  home, guida rapida, riferimento API generato dai docstring reali dei 5 moduli pubblici
  (`Armatura`, `Orca`, `hybrid_engine`, `engine`, `robust_filters`), changelog/licenza
  sorgentati da `CHANGELOG.md`/`LICENSE.md` -- stesso impianto mkdocs-material gia' in uso su
  Dense-Evolution.

### Fixed
- **Due assert instabili in CI**: `test_boundB.py`/`test_boundD.py` avevano un
  `assert t_elapsed >= 5.0` che non verificava la correttezza della difesa, solo che il calcolo
  fosse durato "abbastanza" sulla macchina di sviluppo originale -- falliva su ogni runner CI
  piu' veloce (verificato: 4.41s invece dei 5.0s richiesti su GitHub Actions, riprodotto su
  ubuntu/windows-latest, ogni versione Python). Rimosso: le assert reali (nessuna divergenza
  numerica) sono l'unica verifica che conta.

### Changed
- **Rimosso "Sentinel"** (nome interno originale del progetto) dal testo rivolto
  all'esterno: descrizione PyPI (`pyproject.toml`), README, home della documentazione,
  docstring di `__init__.py`, e l'etichetta nella definizione di Software in `LICENSE.md`
  (solo l'etichetta, nessuna modifica ai componenti elencati/coperti dalla licenza). Il
  `CHANGELOG.md` (storia reale del progetto) e i commenti interni nel codice restano
  invariati.

## [1.1.4]

### Added
- **`pressure_valve`**: orchestratore automatico dei quattro rilevatori
  aggiunti in 1.1.3. Non un voto (quanti metodi segnalano un punto) ma la
  combinazione classica a minima varianza (stimatore BLUE): ogni metodo
  produce una coppia (centro, scala) locale, i pesi della combinazione
  sono derivati con un moltiplicatore di Lagrange minimizzando la varianza
  della combinazione pesata sotto il vincolo che i pesi sommino a 1 --
  w_k proporzionale a 1/scala_k^2, un metodo la cui incertezza si gonfia
  (es. Chauvenet quando la finestra contiene gia' un outlier) viene
  automaticamente pesato meno, senza scartarlo esplicitamente. Decisione
  finale sempre binaria (marcato/sostituito con la mediana locale, o
  intatto). Soglia di default (`soglia_pressione=8.0`) calibrata
  empiricamente -- 1/300 falsi positivi su rumore gaussiano puro,
  rilevamento corretto di un outlier chiaro e di due outlier ravvicinati.
- **Soglia dinamica via JSD ("la molla")**: `soglia_pressione` non e' piu'
  fissa -- si confronta la finestra locale con una piu' ampia (Jensen-
  Shannon divergence) e si allarga proporzionalmente quando le due
  distribuzioni divergono (una vera transizione di regime in corso), senza
  mai smettere di decidere in binario. La soglia puo' solo allargarsi
  (JSD >= 0), mai restringersi sotto il valore base.
  - **Bug reale trovato e corretto durante la calibrazione**: la JSD
    "ingenua" (bin fissi, epsilon quasi-zero) risultava PIU' alta su
    rumore stazionario puro che vicino a una vera transizione -- il
    contrario esatto dello scopo del meccanismo, un artefatto da pochi
    campioni per bin. Corretto con binning adattivo (~6 campioni/bin) e
    smoothing di Laplace vero (pseudo-conteggio, non solo un epsilon per
    evitare log(0)) -- riverificato: JSD media vicino a una vera
    transizione 1.0->5.0 ora 2-3x piu' alta che su rumore stazionario alla
    stessa ampiezza, la direzione corretta.

## [1.1.3]

### Added
- **`dense_armor/utility/robust_filters.py`**: quattro rilevatori di anomalie
  "a basso costo" -- Chauvenet's criterion (1863), Tukey's fences/IQR, Hampel
  filter, sigma-clipping iterativo. Statistiche classiche, note e usate da
  decenni in telemetria/astronomia, nessun modello dinamico/stato -- solo
  aritmetica su una finestra locale centrata (pensate per pulizia offline/
  batch, non il ciclo causale real-time di `core/hybrid_engine.py`; adatte
  anche a un futuro porting embedded/Arduino, dove non ci si potra' appoggiare
  a dipendenze come numpy).
- **`pressure_valve`**: orchestratore automatico dei quattro metodi sopra.
  Non un voto (quanti metodi segnalano un punto) ma la combinazione classica
  a minima varianza (stimatore BLUE): ogni metodo produce una coppia
  (centro, scala) locale, e i pesi della combinazione sono derivati con un
  moltiplicatore di Lagrange minimizzando la varianza della combinazione
  pesata sotto il vincolo che i pesi sommino a 1 -- il risultato,
  w_k proporzionale a 1/scala_k^2, pesa automaticamente meno un metodo la
  cui incertezza si gonfia (es. Chauvenet quando la finestra contiene gia'
  un outlier), senza doverlo scartare esplicitamente. Decisione finale
  sempre binaria (marcato/sostituito con la mediana locale, o intatto) --
  nessuno stato intermedio, coerente col resto dell'ecosistema. Soglia di
  default (`soglia_pressione=8.0`) calibrata empiricamente (non "3 sigma":
  la scala combinata e' sempre piu' stretta di ogni scala singola) --
  verificato 1/300 falsi positivi su rumore gaussiano puro, rilevamento
  corretto di un outlier chiaro e di due outlier ravvicinati (dove
  Chauvenet da solo soffre), nessun tocco su un gradino genuino sostenuto.

## [1.1.2]

### Fixed
- **`hybrid_shield` restava bloccato su un equilibrio sbagliato dopo un
  cambiamento di livello genuino e sostenuto** (`core/hybrid_engine.py`):
  trovato testando `Armatura(livello_ia=0.0)` dal vivo su uno scenario base
  (salto da 1.0 a 5.0, poi 30 punti stabili al nuovo livello). La finestra
  di baseline locale era presa da `out` (già guarito) invece che da
  `processed` (grezzo) — una scelta deliberata per proteggere da uno spike
  isolato passato che sposta la baseline, ma con un effetto collaterale
  peggiore su un gradino vero: i primi 1-2 punti dopo la transizione
  vengono inevitabilmente respinti (nessun rilevatore causale distingue un
  gradino vero da uno spike al primo campione), quei punti respinti
  finiscono in `out`, e la finestra dei punti successivi li rimedia dentro
  il proprio calcolo — producendo una baseline "a metà strada" che respinge
  ANCHE i punti successivi genuini, in un loop che si autoalimenta. Il
  segnale restava bloccato per sempre su un valore intermedio sbagliato
  (~4.0 invece di 5.0), non solo per una manciata di passi transitori.
  Fix: la finestra torna a usare `processed` (grezzo), riallineata a
  `ia_utils.vector_healing.enhanced_dense_healing_hybrid` (il riferimento
  che questo modulo dichiara di generalizzare) — il valore di fallback è
  ora la MEDIANA della finestra grezza, non la media: la mediana ignora già
  da sola un singolo spike isolato nella finestra, quindi la protezione
  originale resta valida senza bisogno del trucco `out` (riverificato
  direttamente). Il test di regressione esistente
  (`test_gradino_sostenuto_non_viene_appiattito_ma_lo_spike_isolato_si`)
  non lo intercettava (tolleranza/finestra di controllo troppo larghe,
  passava anche col bug) — aggiunto un nuovo test più severo che fallisce
  sul codice vecchio e passa su quello corretto.

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
