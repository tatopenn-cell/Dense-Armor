```
    ██████╗ ███████╗███╗   ██╗███████╗███████╗       █████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗
    ██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝      ██╔══██╗██╔══██╗████╗ ████║██╔═══██╗██╔══██╗
    ██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗  █████╗███████║██████╔╝██╔████╔██║██║   ██║██████╔╝
    ██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ╚════╝██╔══██║██╔══██╗██║╚██╔╝██║██║   ██║██╔══██╗
    ██████╔╝███████╗██║ ╚████║███████║███████╗      ██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║
    ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝
```

<p align="center">
  <img alt="tests" src="https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/tatopenn-cell/Dense-Armor"><img alt="codecov" src="https://codecov.io/gh/tatopenn-cell/Dense-Armor/branch/master/graph/badge.svg"></a>
  <img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg">
  <img alt="license" src="https://img.shields.io/badge/license-BSL_1.1-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="training" src="https://img.shields.io/badge/training%20required-no-brightgreen.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <img alt="detectors" src="https://img.shields.io/badge/anomaly%20detectors-4%2B1-blueviolet.svg">
  <img alt="combination" src="https://img.shields.io/badge/combination-Lagrange%20(BLUE)-9cf.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<p align="center"><strong>Runtime shield per input/output di modelli IA. Nessun riaddestramento. Nessuna magia — solo damping adattivo verificato con test reali.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Documentazione completa, riferimento API, guida rapida →</strong></a></p>

---

## `$ cosa fa`

Un sensore che manda letture perse (`NaN`) o spara un valore assurdo (`1e6` invece di `1.2`) rompe silenziosamente qualunque pipeline a valle. Dense-Armor si mette in mezzo, tra il dato grezzo e il modello che lo consuma:

```
  dato corrotto ──► [ SCUDO INGRESSO ] ──► modello IA ──► [ SCUDO USCITA ] ──► output pulito
                     purifica vs               │             purifica vs
                     riferimento                │             risposta-al-riferimento
                     (o stima cieca robusta)    │             (o auto-consistenza)
```

- **Ingresso**: ripulisce il dato corrotto verso un riferimento pulito, se lo hai — o verso una stima robusta ricavata dal dato stesso, se non lo hai.
- **Uscita**: verifica che la risposta del modello non sia a sua volta corrotta, confrontandola con la risposta che il modello darebbe al riferimento pulito.
- **Margine d'errore**: per ogni valore corretto, restituisce quanto è stato spostato per ripulirlo. Correzione piccola → fidati. Correzione grande → tratta con cautela.

Lascia i pesi intatti. Gira a runtime. Funziona da 1D fino a 11D, testato (vedi `test/test_orca2.py`).

---

## `$ install`

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[quantum]"      # + Dense-Evolution (simulatore NISQ)
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
```

```python
# obbligatorio prima di ogni import — richiesto dal gating a 64-bit
import jax
jax.config.update("jax_enable_x64", True)
```

```powershell
# equivalente da PowerShell, prima del lancio
$env:JAX_ENABLE_X64="True"
```

Per lanciare la suite di test in locale (clonando il repo, non serve se hai solo installato da pip):

```bash
pip install -e ".[dev]"
pytest test/ -v
```

---

## `$ quickstart`

```powershell
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

```
> anomalia @ indice 2 (picco 9999)
> anomalia @ indice 4 (NaN)
> tutto il resto: intatto
```

Collegato a un modello vero:

```python
from dense_armor.utility.orca import Orca

orca = Orca()
output_protetto = orca.protect_and_forward(
    mio_modello,                              # callable: x -> output (JAX/NumPy)
    dato_corrotto,                            # tensore dal sensore/pipeline
    x_reference=dato_pulito_di_riferimento,   # opzionale ma consigliato
)

orca.margine_ingresso_medio, orca.margine_uscita_medio   # quanto fidarsi
```

Serie 1D (loss di training, metriche, token stream):

```python
from dense_armor import Armatura

a = Armatura(livello_ia=0.0)   # 0 = filtra attivamente · 1 = solo segnala
pulito, K, anomalie = a.analizza(serie)
```

---

## `$ internals`

Due motori distinti, a seconda di dove entri:

```
Armatura (serie 1D: loss, metriche, sensori)   Orca (scudo entrata+uscita per un modello IA intero)
─────────────────────────────────────────      ─────────────────────────────────────────────────────
core/hybrid_engine.py                           STADIO 1  core/engine.py
motore a trigger binario (phi_ab/vettore         stabilizzatore adattivo, soglia dinamica su
dinamico), verificato in Dense-Evolution e       volatilità recente
adattato a segnali scalari a scala libera        STADIO 2  utility/collatz.py
                                                  gating basato su Collatz, decide QUANTO
                                                  smorzare verso il riferimento pulito
```

`Armatura.analizza()` decide punto per punto, senza gradi intermedi: un valore o è un cambiamento genuino (passa) o è rumore/spike isolato (sostituito con la baseline locale — la finestra recente se non hai un riferimento, il tuo riferimento esplicito se lo passi).

`Orca.protect_and_forward()` (scudo completo per un modello) usa ancora i due stadi originali. Senza riferimento pulito (modalità cieca), prima di ricadere sulla stima locale Orca cerca nella propria **memoria di riferimenti puliti passati** (per shape, via `apply_fast_resonance` — la stessa risonanza descritta più sotto nel toolkit) un riferimento simile all'input corrotto corrente; se non lo trova, rigetto degli outlier gravi via mediana locale, poi Stadio 1 in versione causale — usa tutta la storia della serie, non solo i vicini immediati, per stimare cosa "dovrebbe" essere quel punto. Ogni volta che un riferimento pulito viene passato esplicitamente, resta in memoria per le chiamate cieche successive (`reference_memory_size`, default 32 per shape).

Prima di ogni chunk pesante, Orca verifica anche RAM/VRAM disponibili tramite `UniversalMemoryGuard` (vedi toolkit sotto): sotto la soglia critica solleva `MemoryPressureError` invece di rischiare un OOM silenzioso qualche chunk più avanti. La dimensione stessa dei chunk (`chunk_threshold`) è auto-derivata da `AIHardwareProfiler` — scala con RAM/backend dell'host invece di un valore fisso uguale per qualunque macchina — a meno che tu non ne passi uno esplicito al costruttore.

Nota tecnica: `AdaptiveSignalStabilizer.filter_batch_scenarios` (usato ad es. dalla suite di test adversarial) accetta solo 2D/3D/4D. Lo scudo entrata di `Orca` e' un percorso diverso, senza quel limite.

---

## `$ orca --use_arbiter`

`Orca.protect_and_forward(..., use_arbiter=True)` — opzionale, default `False` — instrada ogni punto verso il correttore giusto invece di forzarne uno solo su tutta la riga:

```python
orca = Orca()
output_protetto = orca.protect_and_forward(mio_modello, dato_corrotto, use_arbiter=True)

orca.etichette_arbitro       # array 'clean'/'spike'/'regime', uno per punto
orca.incertezza_arbitro_media   # 0..1: quanto la classificazione stessa e' ambigua
orca.tipi_corruzione_visti(dato_corrotto.shape)   # Counter, popolato quando x_reference e' noto
```

`spike` (impulso isolato) → rigetto duro verso la mediana di una finestra di riferimento larga; `regime` (cambio di livello sostenuto, riconosciuto guardando lunghezza e coerenza interna della sequenza di punti anomali) → valore grezzo, fiducia piena; `clean` → qualunque cosa lo scudo standard a 4 fasi abbia già prodotto, non il grezzo — un segnale continuo non anomalo ha comunque bisogno dello smorzamento morbido di `AdaptiveSignalStabilizer`.

Verificato sui 7 scenari di `test/testKalman.py` (`test/test_arbiter_orca_integration.py`): **mai peggio del default, meglio su 5/7** (impulsi isolati, corruzione sotto soglia, rumore a code pesanti, rottura di livello, buco di dati NaN), pari su un caso, identico sul rimanente per design corretto (segnale strutturato continuo → ricade sullo scudo standard). Dettagli, cronologia di un bug reale trovato e risolto (finestra di riferimento simmetrica → causale) nel [changelog](https://tatopenn-cell.github.io/Dense-Armor/changelog/) e nel docstring di `utility/arbiter.py`.

---

## `$ streaming --realtime`

Un robot reale gira a 30-100Hz e non può aspettare un array già registrato. `StreamingDeviationDetector` (`dense_armor.utility.streaming`) porta a latenza zero solo la metà causale di `classify_segments` — il flag di deviazione per-punto, non l'etichetta finale spike/regime, che richiede di guardare avanti nella sequenza e resta una domanda batch per design:

```python
from dense_armor.utility.streaming import StreamingDeviationDetector

det = StreamingDeviationDetector(radius=10, ref_mult=3, n_sigmas=3.0)
for x in flusso_sensore:
    is_deviante = det.update(x)
```

`MultiChannelStreamingDeviationDetector` e `classify_segments_multichannel` applicano la stessa logica già validata a più canali indipendenti (i giunti di un braccio robotico, gli assi di un IMU) senza richiedere un ciclo manuale — ogni canale mantiene la propria finestra di riferimento. Promosso da Dense-Evolution-Discovery dopo validazione su due domini fisici reali indipendenti (braccio robotico SO-101, IMU umano reale) — stessa disciplina già usata per `stable_frame_filter.py` e `velocity_gated_stable_mask`. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/streaming/).

---

## `$ cusum --detectability`

Un drift troppo lento per superare, punto per punto, la soglia istantanea di `classify_segments` sfugge ad Arbiter per design. `cusum_detector` (`dense_armor.utility.cusum`) accumula le piccole deviazioni nel tempo invece di giudicare ogni punto isolatamente:

```python
from dense_armor.utility.cusum import cusum_detector, detectability_report

flagged, cusum = cusum_detector(x, radius=10, ref_mult=3, k=0.5, h=20.0)

report = detectability_report(local_noise_scale=mad_locale, k=0.5, h=5.0, candidate_shift=10.0)
# {'false_alarm_arl': ..., 'detection_arl': ..., 'shift_in_sigma': ...}
```

`detectability_report` stima *prima* di lanciare un benchmark quanti campioni servono per rilevare uno shift dato il rumore locale reale del detector -- teoria Reynolds (1975)/Siegmund (1985), promossa da Dense-Evolution-Discovery dopo validazione su due domini fisici reali indipendenti (lidar, accelerometro): sul lidar la latenza reale batte sempre la stima teorica; sull'accelerometro il risultato è genuinamente misto -- documentato così com'è, non forzato a coincidere. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/cusum/).

---

## `$ mcp --server`

Un server MCP (`dense_armor.mcp_server`) espone 5 tool — `dense_armor_health`, `dense_armor_clean_signal` (Orca completo, con `use_arbiter`), `dense_armor_detect_anomalies` (solo classificazione), `dense_armor_robust_filter`, `dense_armor_heal_series` — così un agente (Claude Code, Claude Desktop, o qualunque client MCP) può ripulire una serie senza scrivere Python. Diretto e in-process (niente kernel HTTP separato, a differenza dell'adattatore di Dense-Evolution — Dense-Armor non ha una web UI da condividere):

```bash
pip install -e ".[mcp]"
claude mcp add dense_armor -- dense-armor-mcp
```

Vive apposta sotto `dense_armor.mcp_server`, non un semplice `mcp_server` — con Dense-Evolution installato nello stesso ambiente (il suo adattatore si chiama esattamente `mcp_server`), un nome non annidato è una collisione vera, verificata, non un'ipotesi. Vedi [`dense_armor/mcp_server/README.md`](dense_armor/mcp_server/README.md) per l'elenco completo dei tool e `test/test_mcp_server.py` per la verifica end-to-end di ognuno.

---

## `$ robust_filters --standalone`

Quattro rilevatori di anomalie classici (`dense_armor.utility.robust_filters`), indipendenti da `Armatura`/`Orca` — nessun modello dinamico, nessuno stato, solo aritmetica su una finestra locale centrata (pensati per pulizia offline/batch, non il ciclo causale real-time; adatti anche a un futuro porting embedded, dove non ci si potrà appoggiare a numpy):

```python
from dense_armor.utility.robust_filters import pressure_valve

pulito, anomalie, pressione, soglia_effettiva = pressure_valve(serie)
```

`pressure_valve` combina Chauvenet's criterion (1863), Tukey's fences/IQR, l'Hampel filter e il sigma-clipping iterativo — non con un voto (quanti dei 4 segnalano un punto), ma con la combinazione classica a minima varianza (stimatore BLUE): ogni metodo produce una coppia (centro, scala) locale, e i pesi sono derivati con un moltiplicatore di Lagrange (minimizza la varianza della combinazione pesata, vincolo Σw=1 → w_k ∝ 1/scala_k²) — un metodo la cui incertezza si gonfia (es. Chauvenet quando la finestra contiene già un outlier, la sua media/std non sono robuste) viene pesato automaticamente meno, senza scartarlo a mano. Decisione finale sempre binaria (marcato/sostituito con la mediana locale, o intatto).

La soglia stessa non è fissa: si confronta la finestra locale con una più ampia via divergenza di Jensen-Shannon, e si allarga quando le due distribuzioni divergono (una vera transizione di regime, non rumore) — mai il contrario, `soglia_effettiva >= soglia_pressione` sempre.

I quattro metodi singoli restano richiamabili anche uno per uno (`chauvenet_criterion`, `tukey_fences`, `hampel_filter`, `sigma_clip`) se serve un solo criterio invece della combinazione.

---

## `$ toolkit --standalone`

Sotto `core/`/`utility/` c'è anche una seconda parte del pacchetto, in gran parte indipendente da Armatura/Orca — la maggior parte di questi moduli non partecipa allo scudo anomalie, sono strumenti a sé che condividono solo il backend JAX/NumPy. Tre eccezioni: `UniversalMemoryGuard`, `apply_fast_resonance` e `AIHardwareProfiler`, richiamati anche da Orca (vedi `$ internals` sopra) — restano comunque utilizzabili standalone. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/toolkit/); qui il riepilogo.

**Pipeline e chunking** (`dense_armor.core`)

- `DynamicAICodegen` — compila una lista di nomi di operazioni (`relu`, `sigmoid`, `tanh`, `scale`, `dropout`, `clip`, `l2_normalize`, `identity`) in una pipeline JAX JIT-compilata via `lax.switch`, con esecuzione a blocchi per liste lunghe e gradiente via autodiff (`compute_gradients`).
- `ImageChunker` (`dense_armor.core.chunk`) — divide/ricompone un batch grande in blocchi a dimensione fissa, sia per array di dati sia per liste di istruzioni compilate.
- `UniversalMemoryGuard` — controlla RAM (e VRAM, se c'è una GPU NVIDIA) prima di un'allocazione pesante, calcola il numero di blocchi necessari per starci; solleva `MemoryPressureError` sotto soglia. Usato anche da `Orca` prima di ogni chunk pesante.

**Hardware e profiling** (`dense_armor.core`)

- `AIHardwareProfiler` — rileva CPU/RAM/backend disponibili e calcola una dimensione massima sicura di tensore per l'host corrente. Usato anche da `Orca` per auto-derivare `chunk_threshold` (scala proporzionalmente all'host invece di un valore fisso uguale per qualunque macchina; passare un valore esplicito lo sovrascrive senza nemmeno istanziare il profiler).
- `StochasticAdversarialNoise` — inietta rumore sintetico (bitflip, dropout, blur gaussiano) in un tensore, preservandone la norma; utile per generare dati di attacco quando si vuole testare un rilevatore.
- `PipelineProfiler` — misura la latenza JIT in microsecondi (warm-up di compilazione separato dal tempo a regime) di una pipeline `DynamicAICodegen` o di `AdaptiveSignalStabilizer`.

**Tensori e configurazioni** (`dense_armor.core`)

- `TensorVault` — libreria di matrici di trasformazione statiche (`invert`, `identity`, `edge_detector`, `blend`) e parametriche (`scale_project`, `amplify`, `bias_shift`), backend/precisione auto-rilevati.
- `ParametricScenarioSimulator` — simulazioni Monte Carlo parallele (`jax.vmap`) su uno stato scalare nel tempo, più un collasso decisionale stocastico condizionato dalla distribuzione.
- `BitwisePermutationEngine` — permuta gli elementi di un vettore combinatorio (spazio 2^n) via maschere di bit target/control.
- `SIGNAL_STABILIZER_PRESETS` (`dense_armor.core.preset`) — 4 configurazioni calibrate (`balanced_v2`, `cifar10_best_v1`, `pure_1d_time_v1`, `cifar10_hardened_lyapunov`) per i parametri di `AdaptiveSignalStabilizer`.

**Logging e provenance** (`dense_armor.core`)

- `MinimalConsoleFormatter` / `CompactJsonFormatter` (`dense_armor.core.logger`) — due `logging.Formatter`: uno leggibile a console, uno JSON compatto per file.
- `AIEngineVisualizer` — esporta un archivio di provenance firmato SHA-256 (parametri, ambiente di esecuzione, hash di integrità) e report testuali di varianza grezza/filtrata.

**Audio e I/O dati** (`dense_armor.utility`)

- `anwav(fpath)` — analizza un file WAV: picco, RMS, loudness stimata (LUFS), fattore di cresta, con verdetto di conformità.
- `diag(iorig, ifilt)` — confronto differenziale tra due segnali audio (percorsi file o array NumPy): fedeltà strutturale, energia rimossa, picco di distorsione.
- `lodat(fpath, dname)` (`dense_armor.utility.iodat`) — legge un tensore da un file HDF5 o NetCDF.
- `apply_fast_resonance(matrix, query)` (`dense_armor.utility.resonance_search`) — punteggio di similarità coseno tra una query e le righe di una matrice, modulato da `apply_damping_blend` (lo stesso operatore usato da Orca). Usato anche da `Orca` in modalità cieca per richiamare un riferimento pulito simile già visto in passato (vedi `$ internals` sopra).

Ognuno testato singolarmente (`test/test_chunk.py`, `test_compiler.py`, `test_memory.py`, `test_preset.py`, `test_tensor.py`, `test_noise.py`, `test_vector.py`, `test_profiler.py`, `test_visualizer.py`, `test_logger.py`, `test_anwav.py`, `test_diagnostic.py`, `test_iodat.py`, `test_resonance_search.py`). Richiede `pip install "dense-armor[audio,data]"` per `anwav`/`diagnostic` (scipy) e `iodat` (h5py/netCDF4).

```python
from dense_armor.core import DynamicAICodegen, UniversalMemoryGuard, TensorVault, AIHardwareProfiler
from dense_armor.core.chunk import ImageChunker
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.utility.anwav import anwav
from dense_armor.utility.iodat import lodat
```

---

## `$ margine d'errore`

```python
orca.margine_ingresso      orca.margine_ingresso_medio      orca.margine_ingresso_max
orca.margine_uscita        orca.margine_uscita_medio        orca.margine_uscita_max
```

`|valore ricevuto − valore corretto|` — quanto lo scudo ha dovuto spostare un dato per ripulirlo. Non è una covarianza calibrata in senso statistico stretto, ma correla bene nei test: basso quando la correzione è affidabile, alto quando lo scudo sta indovinando alla cieca.

Segnale complementare, non ridondante, con `use_arbiter=True`: `orca.incertezza_arbitro`/`incertezza_arbitro_media` dice quanto la *classificazione* stessa è ambigua (vicino al confine deviante/pulito o spike/regime), non quanto è stata grande la correzione — una correzione piccola con incertezza alta è un caso ambiguo andato bene per caso, non uno davvero sicuro. Vedi `$ orca --use_arbiter` sopra.

---

## `$ vs kalman-filter --honest`

Non sostituisce un Kalman filter — risolvono problemi diversi, punto.

Random walk, 15% dati mancanti, 3% spike enormi:

| metodo | MSE |
|---|---|
| nessuna protezione | ~21000 |
| Kalman *senza* gating anti-outlier (il caso comune) | ~7300 |
| Kalman *con* gating anti-outlier e dinamica nota | **~0.12** |
| Dense-Armor, modalità cieca | ~0.23 |

```diff
+ contro un Kalman non protetto (lo scenario piu' comune in pratica): vince nettamente
+ un solo spike enorme manda in tilt il gain di Kalman e lo trascina dietro di se'
+ zero setup: nessun modello di processo da conoscere, stimare o calibrare (niente Q/R)
+ funziona anche dove Kalman non si applica per niente: immagini, embedding, tensori generici
```

**Il vantaggio è la libertà, non la specializzazione.** Un Kalman filter ben progettato, calibrato su un processo dinamico *noto*, resta più preciso su quel singolo caso d'uso — ma richiede di conoscere in anticipo il modello del sistema e ricalibrarlo per ogni nuovo tipo di dato. Dense-Armor è un **filtro generale**: nessuna personalizzazione, nessuna conoscenza a priori richiesta, si applica così com'è a qualunque tensore (temporale o no). Il prezzo di questa libertà è un po' di precisione in meno nel caso specifico in cui esiste già un modello dinamico noto e calibrato — una perdita piccola (~0.23 vs ~0.12 di MSE nel nostro test) rispetto al vantaggio di non dover mai configurare nulla.

---

## `$ robustezza adversarial --tested`

9 test motore condivisi eseguiti fino in fondo, nessun crash, nessun NaN sfuggito. Nessuna difesa mai sotto il 64%. Codice reale, non solo numeri riportati: [`test/test_boundA.py`](test/test_boundA.py)–[`test_boundE.py`](test/test_boundE.py) — gli stessi attacchi (PGD/BIM/MI-FGSM, affine/elastico, Carlini-Wagner, DeepFool, Fourier) girano contro `dense_armor.core.engine.AdaptiveSignalStabilizer`, non un motore separato per il benchmark.

| attacco | tipo | difesa |
|---|---|---|
| PGD / BIM / MI-FGSM | gradiente, 1000 passi | mitigato, V finale 0.013-0.078 |
| affine / elastico | geometrico, 50k iter | contenuto, V_inf 0.05-0.14 |
| Fourier broadband | dominio frequenza, 50k iter FFT | **99.78%+** |
| Carlini-Wagner (L2) | ottimizzazione | 78.96% |
| Carlini-Wagner (L∞) | ottimizzazione | **64.39%** — il punto più debole trovato finora |
| DeepFool | ottimizzazione | 78.79% |
| combinato (tutti insieme) | 150.140 passi totali | nessun gradiente esplosivo |

Onesto: **C&W in norma L∞ è l'attacco che buca di più** tra quelli testati. Non è un fallimento — resta protezione reale — ma è la crepa più vicina a un cedimento tra tutte le prove fatte, e va saputo prima di affidarci contro quello scenario specifico.

---

## `$ limiti --known`

```
1. semantica       distingue deviazioni geometriche, non significati
2. modalita' cieca senza riferimento: buona per evitare collassi/NaN, non per
                    ricostruire con precisione un dato realmente perso
3. generalita'      zero calibrazione richiesta, si applica a qualunque tensore --
                    a discapito di un pizzico di precisione dove esiste gia' un
                    modello dinamico noto e calibrato (es. Kalman su serie pure)
4. deriva lenta     invisibile punto per punto, serve riferimento=baseline_storica
5. C&W norma L-inf  la difesa piu' debole misurata finora (64%, contro 79-83%
                    delle altre varianti di attacco testate) -- vedi tabella sopra
6. adversarial      attacchi costruiti apposta per mimare la coerenza del
   adattivo          segnale pulito (oltre a PGD/BIM/MI-FGSM/C&W/DeepFool/Fourier
                    gia' testati) non ancora coperti dalla suite
```

**Causa reale del punto 5, non solo il numero**: investigata a fondo, non ancora risolta. C&W
in norma L-inf costruisce una perturbazione spazialmente liscia su tutta la griglia in un
colpo solo (ottimizzazione a gradiente globale). Nessun controllo di coerenza puramente
locale (confronto di un punto con i suoi vicini immediati, quello che questo motore usa) puo'
distinguere una struttura spaziale genuinamente liscia da una costruita apposta per sembrarlo
-- e' lo stesso identico segnale statistico. Tentati e verificati empiricamente tre
interventi mirati (rate-limit sulla volatilita', ancora di coerenza a lungo termine, guinzaglio
rigido sulla deriva massima): nessuno ha spostato il numero, uno lo ha persino peggiorato.
Non e' un limite di taratura -- serve un riferimento esterno (non solo il contesto spaziale
locale) per risolverlo davvero.

---

## `$ license`

Business Source License 1.1 — uso gratuito non commerciale, converte in Apache 2.0 il `2029-06-01`. Vedi [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Progetto gemello di [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution) (simulatore di circuiti quantistici NISQ).
