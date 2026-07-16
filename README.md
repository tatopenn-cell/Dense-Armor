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
  <img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg">
  <img alt="license" src="https://img.shields.io/badge/license-BSL_1.1-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="training" src="https://img.shields.io/badge/training%20required-no-brightgreen.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
</p>

<p align="center"><strong>Runtime shield per input/output di modelli IA. Nessun riaddestramento. Nessuna magia — solo damping adattivo verificato con test reali.</strong></p>

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

Non tocca i pesi. Non riaddestra niente. Gira a runtime su qualunque tensore JAX/NumPy.

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
pytest tests/ -v
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

Due stadi in sequenza, nessuno dei due amplifica mai un errore — solo damping verso il pulito:

```
STADIO 1  core/engine.py           STADIO 2  utility/collatz.py
──────────────────────────         ──────────────────────────────
stabilizzatore adattivo            gating basato su Collatz
soglia dinamica su volatilità      decide QUANTO smorzare ogni punto
recente, smorza chi la supera      verso il riferimento pulito
```

Senza riferimento pulito (modalità cieca): rigetto degli outlier gravi via mediana locale, poi Stadio 1 in versione causale — usa tutta la storia della serie, non solo i vicini immediati, per stimare cosa "dovrebbe" essere quel punto.

---

## `$ margine d'errore`

```python
orca.margine_ingresso      orca.margine_ingresso_medio      orca.margine_ingresso_max
orca.margine_uscita        orca.margine_uscita_medio        orca.margine_uscita_max
```

`|valore ricevuto − valore corretto|` — quanto lo scudo ha dovuto spostare un dato per ripulirlo. Non è una covarianza calibrata in senso statistico stretto, ma correla bene nei test: basso quando la correzione è affidabile, alto quando lo scudo sta indovinando alla cieca.

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

9 test motore condivisi eseguiti fino in fondo, nessun crash, nessun NaN sfuggito. Nessuna difesa mai sotto il 64%:

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

---

## `$ license`

Business Source License 1.1 — uso gratuito non commerciale, converte in Apache 2.0 il `2029-06-01`. Vedi [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Parte dell'ecosistema **Sentinel**, insieme a [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution) (simulatore di circuiti quantistici NISQ).
