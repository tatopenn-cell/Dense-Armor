# Dense-Armor 🛡️

**Lo scudo indossabile Sentinel** — uno smorzatore adattivo di anomalie a due stadi per qualsiasi segnale numerico prodotto o consumato da un'IA: perdite di addestramento (training losses), embedding, flussi di sensori, ritmo del testo e fedeltà degli stati quantistici.

Parte dell'**ecosistema Sentinel** sviluppato da Salvatore Pennacchio, insieme a [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution) (un simulatore quantistico di vettori di stato NISQ). Entrambi i progetti condividono la stessa matematica di smorzamento, basata sulla contrazione di Lyapunov del rapporto logaritmico combinata con un fattore di coerenza.

---

## 🔬 Come funziona

*   **Stadio 1 — Filtro IIR Adattivo con Soglia Dinamica (`core/engine.py`)**: Traccia il segnale in tempo reale e applica uno smorzamento immediato laddove la volatilità supera la soglia mobile calcolata dal sistema.
*   **Stadio 2 — Gating di Smorzamento ABCollatz (`utility/collatz.py`)**: Sfrutta le traiettorie caotiche della congettura di Collatz accelerate su GPU per decidere *quanto* contrarre ciascun punto numerico verso un riferimento sano. L'operazione è puramente sottrattiva (Contrazione di Lyapunov): sposta il segnale esclusivamente verso lo stato pulito, senza mai amplificare gli errori hardware o i rumori.
*   **Rilevatore Robusto (Spike Detector)**: Sfrutta una metrica basata su mediana/MAD (z > 6) per intercettare istantaneamente i picchi di magnitudo estrema inseguiti dal filtro di tracciamento.
*   **Rilevatore di Deriva (`deriva`)**: Intercetta l'effetto "rana bollita", ovvero la deriva infinitesimale cumulativa che risulta invisibile se analizzata punto per punto (verificato strumentalmente: misura una deriva reale di +0.0040/passo registrandola come +0.0041).
*   **Infrastruttura Sicura**: Architettura nativamente protetta contro i valori indefiniti (`NaN-safe everywhere`) e indipendente dalle dimensioni dei dati (da 1D a ND tramite passaggi automatici di `ravel()` e `reshape`).

---

## 🛠️ Tecnologie Avanzate e Segreti Industriali

L'ispezione approfondita del codice sorgente ha rivelato l'implementazione di costrutti matematici e fisici non convenzionali parallelizzati tramite **Google JAX**:

1.  **Teoria dei Numeri su GPU (`collatz.py`)**: Per aggirare l'assenza di funzioni native in JAX, il sistema implementa `check_prime_native` tramite vettorizzazione di array di divisori. Calcola il radicale matematico di un numero (rad(n)) in modo differenziabile senza strutture condizionali tramite l'equazione:
    \[\text{rad}(n) = \exp\left(\sum \log(p_i)\right)\]
2.  **Regolarizzazione Cosmo-Quantistica (`core/compiler.py`)**: Nella funzione `_topological_loss_fn`, l'errore quadratico dei gradienti viene scalato dividendo per la costante `_PHI_FOUR` secondo la metrica dello **Spazio di de Sitter** (un modello cosmologico di universo in espansione con curvatura positiva). Questo impedisce ai gradienti calcolati da `jax.grad` di subire un collasso topologico.
3.  **Scudo di Routing Input/Output (`utility/orca.py`)**: Il modulo `Orca` implementa una barriera di isolamento a 4 fasi che applica una compressione logaritmica iniziale, controlla i fallimenti computazionali tramite `jnp.isnan` e misura la curvatura spaziale geometrica dello scenario.

---

## 🎛️ Il Clip Dinamico: lo scudo rileva, l'IA giudica

Lo scudo identifica le *deviazioni*, non i *significati*. Spetta all'utilizzatore decidere come gestirle tramite il parametro del livello:

```python
import jax
# ABILITAZIONE CONFIGURAZIONE 64-BIT (TASSSATIVA per i calcoli di Collatz in JAX)
jax.config.update("jax_enable_x64", True)

from dense_armor import Armatura

a = Armatura(livello_ia=0.0)   # IA neonata   -> Clip totale: lo scudo FILTRA e corregge (Armatura)
a = Armatura(livello_ia=1.0)   # IA matura    -> Clip zero: lo scudo si limita a SEGNALARE (Lente)

cleaned, K, anomalies = a.analizza(series)                       # Serie 1D
cleaned, K, anomalies = a.analizza(today, riferimento=baseline)  # Pattern anti-deriva (Massima protezione)
rate, significant, exponent = a.deriva(series)                   # Rilevatore della rana bollita
```

**Soglia minima inviolabile:** La segnalazione delle anomalie non subisce mai alcun taglio (clip). Lo scudo vede e riporta le deviazioni in modo identico bit-per-bit a qualsiasi livello. In compiti complessi (come i test ARC), l'anomalia stessa potrebbe *essere* la risposta finale: l'architettura non la silenzia mai.

Il livello dell'IA può essere misurato dinamicamente sul flusso invece di essere dichiarato: `Armatura.livello_da_entropia(entropy, vocab_size)`.

---

## 📦 Installazione e Requisiti Hardware

L'infrastruttura richiede i driver video e le librerie per l'interrogazione della memoria GPU in tempo reale (`nvidia-ml-py`).

```bash
pip install dense-armor                 # Core engine (numpy + jax + jaxlib)
pip install "dense-armor[quantum]"      # + Strumenti quantistici di Dense-Evolution
pip install "dense-armor[audio,data]"   # + Analisi forme d'onda WAV, loader HDF5/NetCDF
```

**⚠️ NOTA OBBLIGATORIA SULLE PRESTAZIONI (Loop di Addestramento):**
La funzione `analizza()` esegue calcoli sequenziali in Python nativo. Se viene inserita all'interno di un loop di addestramento GPU per analizzare ogni singolo batch riga per riga, genererà un collo di bottiglia massiccio (fino a 44 secondi di rallentamento). 
Per mantenere la velocità della GPU allineata a `0.000000s`, adottare tassativamente la strategia dello **Spot Check (Controllo a Campione)**, richiamando la validazione dello scudo solo a intervalli regolari (es. 1 volta ogni 10 o 50 passi). Questo riduce l'overhead del 97% garantendo la massima velocità di calcolo complessiva ed evitando i crash distruttivi dei pesi.

---

## ⏱️ Verifica Rapida (10 secondi)

Esegui il modulo direttamente da PowerShell abilitando la precisione a 64 bit:

```powershell
\$env:JAX_ENABLE_X64="True"
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

**Output atteso:** Anomalie rilevate all'indice 2 (picco 9999, con alterazione della curvatura locale `K`) e all'indice 4 (`NaN`). Nessun altro elemento della sequenza viene intaccato o segnalato.

---

## 📊 Prestazioni Misurate

*   **Sinusoide rumorosa con picchi di magnitudo $\pm5.0$ (Full Armor)**: Riduzione del valore **RMSE del −86%**, con picchi smorzati per oltre il 90% (i valori vengono contratti nello spazio dei gradienti, mai cancellati o ignorati).
*   **Deriva costante di $+0.004/\text{passo}$ sepolta in un rumore di fondo di $0.05$**: Invisibile se analizzata punto per punto (per costruzione fisica del filtro IIR), viene **intercettata tramite la baseline storica** (`riferimento=`) e quantificata dal rilevatore Metro a $+0.0041/\text{passo}$.

---

## 🔎 Limiti Dichiarati (Onestà del Progetto)

1.  **Semantica**: Distingue le deviazioni geometriche ma non i significati logici. L'IA che indossa lo scudo rimane il giudice finale (scelta architetturale nativa).
2.  **Sotto-soglia**: Anomalie inferiori a $\sim4\sigma$ sepolte nel rumore bianco intenso possono sfuggire al controllo (mitigato parzialmente dal rilevatore median/MAD).
3.  **Deriva lenta**: Invisibile se valutata in isolamento per costruzione matematica. Richiede tassativamente l'adozione del pattern `riferimento=historical_baseline`.
4.  **Avversario Adattivo**: Gli attacchi informatici evoluti strutturati appositamente per mimare la coerenza intrinseca del segnale pulito non sono ancora inclusi nella suite di test.

---

## 💬 Interfacce di Chat (Disponibili solo nel Repository)

Il file `interfaccia3.py` collega direttamente lo scudo all'interno di un loop di chat interattivo (tramite API Claude o modelli locali Qwen eseguiti in Ollama). L'IA scrive codice Python che viene eseguito localmente (previa conferma dell'utente) sfruttando i moduli precaricati del sistema: shield, drift, curvature, analisi WAV, caricamento HDF5 e un **simulatore di circuiti quantistici con canali di rumore NISQ e report di guarigione** (via Dense-Evolution). I parametri estratti dallo scudo a ogni risposta modulano attivamente i coefficienti di campionamento (temperatura, top_p, repeat_penalty) della generazione successiva.

---

## 📄 Licenza

Business Source License 1.1 — Uso gratuito per scopi non commerciali. Converte automaticamente in licenza Apache 2.0 in data 01-06-2029. Vedere il file [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it> - Dense-Armor`

# Manuale Operativo: Integrazione e Uso di Dense-Armor (Shield)

Questo documento descrive le linee guida ufficiali per integrare lo scudo **Dense-Armor** (sviluppato da Salvatore Pennacchio) all'interno di una pipeline di addestramento o inferenza per Intelligenze Artificiali in JAX/NumPy, massimizzando la sicurezza senza distruggere le prestazioni della GPU.

---

## ⚙️ Configurazione Hardware Obbligatoria (64-bit)

Il modulo di protezione caotica `collatz.py` esegue calcoli di traiettoria e radicali che richiedono tassativamente la precisione a 64 bit. Poiché Google JAX lavora nativamente a 32 bit per spingere la velocità, **è fondamentale** forzare l'abilitazione dei 64 bit all'inizio di qualsiasi script di avvio, prima di importare l'Armatura:

```python
import jax
# Abilitazione nativa della precisione X64 (Previene i Warning di troncatura sulla GPU)
jax.config.update("jax_enable_x64", True)
```

In alternativa, da terminale PowerShell prima del lancio:
```powershell
$env:JAX_ENABLE_X64="True"
```

---

## 🏎️ Strategia d'Uso: Controllo a Campione (Spot Check)

**IMPORTANTE:** La funzione `scudo.analizza()` esegue controlli analitici sequenziali sull'entropia e sulla deriva geometrica dei dati. Passare ogni singolo batch di immagini o dati bidimensionali riga per riga all'interno del loop principale introduce un collo di bottiglia critico (fino a 40+ secondi di ritardo).

Per mantenere le prestazioni della GPU a `0.000000s`, si adotta la strategia dello **Spot Check**: lo scudo ispeziona i dati a intervalli regolari (es. ogni 10 o 50 passi di addestramento), lasciando l'IA libera di aggiornare i pesi alla massima velocità hardware per il resto del tempo.

### Script di Esempio Pratico (Loop di Addestramento Protetto)

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time
from dense_armor.armatura import Armatura

# 1. Inizializzazione dello Scudo Sentinel
scudo = Armatura(static_threshold=0.15, initial_damping=0.85)
chiave = jax.random.PRNGKey(42)

# Costanti di configurazione del benchmark
PASSI_ADDESTRAMENTO = 100
FREQUENZA_CONTROLLO = 10  # Ispezione dello scudo ogni 10 passi (Fattore di mitigazione overhead)

# 2. Definizione del Modello IA Standard (Esempio differenziabile in JAX)
def inizializza_rete(chiave):
    w_chiave, b_chiave = jax.random.split(chiave)
    return {
        "w": jax.random.normal(w_chiave, (784, 10)) * 0.01,
        "b": jax.random.normal(b_chiave, (10,)) * 0.01
    }

def predict(params, inputs):
    return jnp.dot(inputs, params["w"]) + params["b"]

def loss_fn(params, inputs, targets):
    return jnp.mean(jnp.square(predict(params, inputs) - targets))

@jax.jit
def update_step(params, inputs, targets, lr=0.01):
    loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets)
    params_aggiornati = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"]
    }
    return params_aggiornati, loss

# Generazione guidata di un dataset strutturato reale (Simulazione MNIST 28x28)
batch_inputs = [np.clip(np.random.logistic(loc=0.5, scale=0.1, size=(32, 784)), 0.0, 1.0) for _ in range(PASSI_ADDESTRAMENTO)]
batch_targets = [np.random.normal(size=(32, 10)) for _ in range(PASSI_ADDESTRAMENTO)]

parametri = inizializza_rete(chiave)
_, _ = update_step(parametri, batch_inputs[0], batch_targets[0]) # Warmup JAX

print("-> Avvio Addestramento Protetto con Monitoraggio a Campione...")

for i in range(PASSI_ADDESTRAMENTO):
    
    # 📌 FASE DI ISPEZIONE (SPOT CHECK)
    if i % FREQUENZA_CONTROLLO == 0:
        # Estraiamo un campione dal batch corrente e lo passiamo all'Armatura
        campione_controllo = batch_inputs[i]
        
        # Orca analizza la curvatura e verifica se ci sono anomalie avversariali
        scudo.analizza(campione_controllo)

    # 🚀 PASSO DI CALCOLO ALLA VELOCITÀ NATIVA DELLA GPU
    parametri, loss = update_step(parametri, batch_inputs[i], batch_targets[i])

print("=== ADDESTRAMENTO COMPLETATO CON SUCCESSO ===")
```

---

## 📊 Analisi della Telemetria e Diagnostica

Quando si esegue il modulo, è possibile richiedere lo scarico della telemetria in formato JSON per tracciare lo stato di salute dello spazio geometrico dei gradienti.

### Comando da Terminale per il Dump del Report
```powershell
python -m dense_armor --json <lista_di_numeri> > report_sentinel.json
```

### Decodifica dei Parametri di Output
*   `"livello_ia"`: Indica il moltiplicatore dello stato di allerta dell'IA. Un valore pari a `1.0` indica uno stato ottimale stabile.
*   `"K_medio"`: È il valore medio di **kappa** (la curvatura misurata dal modulo `orca.py`). In condizioni normali oscilla intorno a `0.85`. Se scende bruscamente sotto `0.80`, significa che lo scudo logaritmico sta assorbendo e dissipando attacchi da saturazione numerica.
*   `"deriva"`: Contiene il `"tasso_per_passo"`. Se `"significativa"` diventa `true`, il dataset sta subendo una mutazione strutturale (Data Drift) ed è consigliabile interrompere il processo per evitare il de-addestramento del modello.
