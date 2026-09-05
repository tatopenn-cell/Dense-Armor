<p align="center">
  <img src="docs/assets/banner.svg" alt="Dense Armor -- shield for AI and robot I/O" width="900">
</p>

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

<p align="center"><strong>Runtime shield per segnali IA e comandi robotici. Nessun riaddestramento. Nessuna magia — solo damping adattivo e controllo verificato con test reali.</strong></p>

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

**Stessa disciplina, un secondo dominio**: da `$ rate_limiter` in poi il pacchetto copre anche la sicurezza di comandi e traiettorie per robot reali — limiti di velocità/accelerazione, control barrier function spaziali, generazione di traiettorie a jerk minimo, dinamica rigida da URDF vero (anche `.xacro`, anche giunti `<mimic>`), controllori passività+CBF fino a 6-DoF. Backend JAX condiviso con `Armatura`/`Orca`, stessa disciplina di verifica su dati/robot reali — un dominio applicativo diverso, non un pacchetto diverso.

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

## `$ rate_limiter --damping`

Un braccio robotico non può eseguire un salto istantaneo illimitato senza rischio -- `rate_limited_follower` (`dense_armor.utility.rate_limiter`) limita quanto velocemente un comando applicato può cambiare fisicamente (velocità + accelerazione), invece di cercare di classificare se una deviazione è reale:

```python
from dense_armor.utility.rate_limiter import rate_limited_follower

applicato = rate_limited_follower(comando_grezzo, max_vel=2.0, max_accel=1.0)
```

Fondato su Berscheid & Kroger (2021), "Jerk-limited Real-time Trajectory Generation" (RSS 2021, arXiv:2105.04830) — causale per costruzione, verificato direttamente. Promosso da Dense-Evolution-Discovery dopo validazione su due domini fisici reali indipendenti (SO-101, ALOHA bimanuale 14-DOF): vince sempre (400/400 trial reali) sulla metrica di sicurezza reale (salto massimo istantaneo), ma **non** è un ripulitore di segnale — su fedeltà media (RMSE) il quadro è genuinamente misto tra i due domini, non nascosto. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/rate_limiter/).

## `$ cbf_filter --spatial`

Un comando che si muove a velocità perfettamente sicura ma dritto verso un ostacolo resta pericoloso — `rate_limiter` limita QUANTO VELOCE, `cbf_filter` limita DOVE:

```python
from dense_armor.utility.cbf_filter import cbf_filtered_trajectory

applicato = cbf_filtered_trajectory(comando_grezzo, obstacle=5.0, safe_dist=2.0, alpha_gain=2.0)
```

Fondato su Ames et al. (2019), "Control Barrier Functions: Theory and Applications" (2019 ECC, arXiv:1903.11199) — stessa teoria di SAFER-Splat, applicata a un ostacolo geometrico noto invece della percezione Gaussian-Splatting via GPU (non disponibile su ogni macchina). Un problema numerico reale trovato e risolto lungo il percorso: la garanzia CBF è continua nel tempo, servono sotto-passi (20/campione, default) per reggere in discreto su comandi reali che possono saltare parecchio tra un campione e l'altro. Promosso da Dense-Evolution-Discovery dopo validazione su due domini fisici reali indipendenti (SO-101, ALOHA): invarianza 100% da partenze sicure su entrambi, minima invasività praticamente esatta (99.9%+ su SO-101, perfettamente esatta su ALOHA). `cbf_safety_filter_live` è la stessa matematica per un loop di controllo reale che reagisce a un tick sensore alla volta (un `dt` reale, non un array pre-registrato) — promossa dopo che un vero loop live ROS2/Ignition ne aveva bisogno e l'aveva dovuta ricostruire a mano. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/cbf_filter/).

## `$ trajectory --quintic`

`rate_limiter` limita QUANTO VELOCE, `cbf_filter` limita DOVE — ma nessuno dei due genera un riferimento da seguire. `quintic_trajectory` copre esattamente questo: un percorso liscio, a jerk minimo, tra due punti, per qualunque numero di giunti in una sola chiamata:

```python
from dense_armor.utility.trajectory import quintic_trajectory

t, q, v, a = quintic_trajectory(q0=[0.0], qf=[10.0], T=2.0)
```

Ridotto deliberatamente rispetto a due paper reali che propongono ottimizzatori molto più grandi (dinamica completa, URDF, coppie) — Lozer, Scalera, Boscariol & Gasparetto (*Robotics and Autonomous Systems*) e Fried & Paternain (arXiv:2412.07859), entrambi letti per intero prima di scrivere codice — al pezzo più semplice e universale: nessun URDF, nessuna dinamica, nessuna connessione al robot. Promosso da Dense-Evolution-Discovery dopo validazione su due domini fisici reali indipendenti (SO-101, ALOHA, 20 escursioni articolari reali): la velocità di picco del quintico è sempre più bassa di quella reale registrata per lo stesso inizio/fine/durata — atteso, non un bug, essendo il percorso più liscio possibile. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/trajectory/).

## `$ kinematic_controller --tracking`

`trajectory` genera un riferimento liscio, ma qualcosa deve trasformarlo in un comando reale — `kinematic_tracking_controller` fa questo, alla stessa scala a singolo integratore di `rate_limiter`/`cbf_filter`:

```python
from dense_armor.utility.kinematic_controller import kinematic_tracking_controller

u_des = kinematic_tracking_controller(q=[0.2], q_ref=[0.5], qd_ref=[1.0], kp=5.0)
```

`u = qd_ref + kp*(q_ref - q)` — per il sistema `qdot = u` questo rende l'errore di inseguimento esattamente `edot = -kp*e`: convergenza esponenziale in forma chiusa, per qualunque traiettoria di riferimento, verificata numericamente. Non è "basato su passività" nel senso dei paper che hanno motivato questa ricerca (Wu & Tan 2025, il vero bersaglio, dietro paywall senza copia aperta trovata; Scruggs, reale ma serve ottimizzazione convessa in dimensione infinita; Califano et al., reale ma serve meccanica Hamiltoniana) — onesto su questo, è più semplice. Promosso da Dense-Evolution-Discovery dopo validazione su due domini fisici reali (SO-101, ALOHA), incatenato con `quintic_trajectory`: ogni escursione reale recupera da un errore iniziale reale dichiarato e converge. Documentazione completa (auto-generata dai docstring reali) sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/kinematic_controller/).

## `$ rigid_body --urdf`

`rate_limiter`/`cbf_filter`/`trajectory`/`kinematic_controller` lavorano tutti a livello di singolo integratore: dai una velocità di giunto, ricevi una velocità di giunto sicura. `RigidBodyModel` (`dense_armor.dynamics.urdf_dynamics`) è diverso — richiede una vera descrizione fisica del robot (un URDF reale, anche `.xacro`) e restituisce dinamica vera a livello di coppia:

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
import jax.numpy as jnp

model = RigidBodyModel("panda.urdf")   # o un percorso .xacro, espanso automaticamente
q = jnp.zeros(model.n)
M = model.mass_matrix(q)
qdd = model.forward_dynamics(q, jnp.zeros(model.n), jnp.zeros(model.n))
```

`model.n` è il numero di gradi di libertà reali e indipendenti (un giunto con `<mimic>` non conta a parte). `mass_matrix`, `gravity_forces`, `bias_forces` e `forward_dynamics` (risolve `M(q)*qdd + C(q,qd)*qd + g(q) = tau`) usano la costruzione Lagrangiana standard via autodiff (`jax.grad`/`jax.jvp`), non simboli di Christoffel scritti a mano. `link_position`/`link_jacobian`/`link_pose`/`link_spatial_jacobian` funzionano per qualunque link nominato nell'URDF, non solo l'end effector.

Promosso da Dense-Evolution-Discovery (Experiment 62) dopo validazione su tre robot reali indipendenti (Kinova Gen3 7-DoF, Kinova Gen3 6-DoF, Franka Emika Panda — manufacturer diverso, giunti prismatici): matrice di massa simmetrica/positiva-definita e conservazione dell'energia con la convergenza RK4 corretta su tutti e tre. Documentazione completa sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/urdf_dynamics/).

## `$ passivity_cbf --controller`

`RigidBodyModel` dà M(q), gravità e dinamica in avanti per qualunque robot — `solve_control_qp` (`dense_armor.dynamics.passivity_cbf_controller`) li usa per guidare quel robot verso un target nello spazio operativo in sicurezza, garantendo passività dell'errore di inseguimento e distanza dalle singolarità cinematiche, entrambe come vincoli in un piccolo QP:

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03)
```

`eps` è l'indice di manipolabilità minimo che il controllore mantiene — `mu` (restituito) non scende mai molto sotto, anche quando il target comandato spingerebbe altrimenti il robot dritto in una singolarità. I limiti reali di posizione/velocità di ogni giunto (dal tag `<limit>` dell'URDF) sono un terzo vincolo, aggiunto solo dove il robot li dichiara davvero.

Fondato su Kurtz, Wensing & Lin (2021, arXiv:2109.13349). Promosso da Dense-Evolution-Discovery (Experiment 61→63) dopo validazione sugli stessi tre robot di `RigidBodyModel`, ognuno spinto verso la propria vera singolarità: manipolabilità mantenuta entro lo 0.1-1.8% della soglia dichiarata in ogni caso. Un bug OSQP reale trovato e risolto lungo il percorso (infeasibility del QP passività+CBF, risolta ricadendo sul solo CBF). Documentazione completa sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/passivity_cbf_controller/).

## `$ six_dof_cbf --pose`

`passivity_cbf_controller` insegue solo la posizione di un link. `six_dof_pbc_cbf_controller.solve_control_qp` estende lo stesso QP alla posa intera — posizione e orientamento insieme — usando lo Jacobiano spaziale 6xN del link invece del solo Jacobiano di traslazione:

```python
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des,
                                    r_des, w_des, wd_des, eps=0.03)
```

`r_des` è l'orientamento desiderato (matrice di rotazione), `w_des`/`wd_des` la velocità/accelerazione angolare desiderata in world frame. L'errore di orientamento usa la formula SO(3) di Lee, Leok & McClamroch (2010) — liscia ovunque, senza il gimbal-lock reale di una formulazione roll-pitch-yaw.

Promosso da Dense-Evolution-Discovery (Experiment 65). Validato con compensazione di gravità esatta a errore zero (precisione macchina) e convergenza reale in anello chiuso (offset iniziale 10cm/30°, RK4 su 1000 tick, errore finale 1e-6 m / 1e-4 rad) — poi sugli stessi tre robot di `passivity_cbf_controller`, dove è emersa una seconda infeasibility OSQP reale (la manipolabilità 6-DoF può essere ben sotto quella 3-DoF alla stessa configurazione) risolta con un terzo livello di fallback (CBF da solo, senza box). Documentazione completa sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/six_dof_pbc_cbf_controller/).

## `$ xacro --macros`

I produttori pubblicano le descrizioni robot come macro `.xacro` (blocchi parametrizzati, espressioni matematiche, `xacro:include`), non come URDF piatto. `RigidBodyModel` ora accetta `.xacro` direttamente:

```python
model = RigidBodyModel("panda_arm_hand.urdf.xacro")
model.n   # 8 -- 7 giunti braccio + 1 coordinata pinza indipendente
```

Il pacchetto reale `xacro` (lo stesso espansore dell'ecosistema ROS, nessuna installazione ROS richiesta) fa l'espansione — nulla su macro/math/condizionali è reimplementato qui.

Promosso da Dense-Evolution-Discovery (Experiment 66), che ha trovato e risolto un'inconsistenza reale nelle macro pubblicate del Franka Panda (`clvrai/furniture`): un link di attacco della mano commentato nella macro del braccio ma richiesto da quella della mano. Nuova dipendenza: `xacro`. Documentazione completa sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/xacro_support/).

## `$ mimic --joints`

Le due dita di una pinza si muovono insieme: chiuderne una chiude anche l'altra. URDF lo esprime con un tag `<mimic joint="..." multiplier="..." offset="..."/>` — l'angolo del giunto slave è sempre `multiplier * angolo_master + offset`, mai una variabile libera. `RigidBodyModel` ora lo rispetta invece di dare al giunto slave una coordinata indipendente propria:

```python
model.n                                    # 8, non 9 -- le due dita condividono un solo DOF reale
model.mimic_map["panda_finger_joint2"]     # (master_dof_idx, multiplier, offset)
```

La cinematica diretta sostituisce `q[master] * multiplier + offset` per l'angolo del giunto mimic; lo Jacobiano geometrico costruito a mano scala la colonna locale del giunto mimic per `multiplier` e la somma nella colonna del suo master, invece di darle una colonna propria.

Promosso da Dense-Evolution-Discovery (Experiment 67), verificato contro una differenza finita centrale reale di `link_pose` (non solo plausibilità): muovere il master di 0.02 sposta entrambe le punte delle dita di esattamente 0.02 in direzioni opposte, e lo Jacobiano scritto a mano corrisponde alla derivata numerica entro 1e-5. Documentazione completa sul [sito](https://tatopenn-cell.github.io/Dense-Armor/api/mimic_joints/).

---

## `$ mcp --server`

Un server MCP (`dense_armor.mcp_server`) espone 8 tool — `dense_armor_health`, `dense_armor_clean_signal` (Orca completo, con `use_arbiter`), `dense_armor_detect_anomalies` (solo classificazione), `dense_armor_robust_filter`, `dense_armor_heal_series`, più `dense_armor_stream_start`/`_update`/`_end` (sessione stateful per un flusso sensore in tempo reale, un canale multiplo alla volta) — così un agente (Claude Code, Claude Desktop, o qualunque client MCP) può ripulire una serie, o seguire un flusso sensore live, senza scrivere Python. Diretto e in-process (niente kernel HTTP separato, a differenza dell'adattatore di Dense-Evolution — Dense-Armor non ha una web UI da condividere):

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
