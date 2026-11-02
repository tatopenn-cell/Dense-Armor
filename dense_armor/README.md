# `$ architettura interna`

Riferimento per chi vuole contribuire al codice, non per chi vuole solo installare e usare il pacchetto — per quello vedi il [README principale](../README.md). Qui si mappano tutti i moduli di `core/` e `utility/`, oltre a quelli già coperti (`damping_operator.py`, `hybrid_engine.py`, `engine.py`, `orca.py`, `collatz.py`).

---

## `$ core/`

| modulo | cosa fa |
|---|---|
| `damping_operator.py` | `apply_damping_blend` — fonde due segnali come `(1 - K) * grezzo + K * riferimento`; il peso `K` del riferimento sta tra circa 0,104 e 0,896 e arriva al massimo già a una distanza di circa 0,05 (per un valore grezzo di 1). |
| `hybrid_engine.py` | `hybrid_shield` — motore a trigger binario di `Armor` (dalla v1.1.0), adattato dalla logica phi_ab/vettore-dinamico verificata in Dense-Evolution. Vedi CHANGELOG [1.1.0] per il dettaglio degli adattamenti (baseline su valori guariti, IPG su valori grezzi, distanza normalizzata sulla volatilità locale). |
| `engine.py` | `AdaptiveSignalStabilizer` — stabilizzatore causale ricorsivo (via `jax.lax.scan`): segue il segnale nel tempo, smorza dove la volatilità recente supera una soglia dinamica. Nucleo dello Stadio 1 di **Orca** (dalla v1.1.0 `Armor` non lo usa più, vedi `hybrid_engine.py`). |
| `memory.py` | `UniversalMemoryGuard` — controlla RAM libera (e VRAM via `nvidia-smi`, se disponibile) prima di allocazioni pesanti; solleva `MemoryPressureError` se sotto soglia. `calculate_optimal_chunks` stima il partizionamento sicuro. Usato anche da `Orca._gc_se_ram_bassa` (prima un check `psutil` ad-hoc duplicato lì, ora delegato qui). |
| `noise.py` | `AIHardwareProfiler` (rileva CPU/GPU/TPU e imposta limiti di sicurezza sui tensori). |

---

## `$ utility/`

Le sottocartelle di `utility/` sono 32, in sei gruppi: `online/` (`learn`, `optim`, `linear`, `timeseries`, `cluster`, `tree`, `forest`, `ensemble`, `bandit`, `active`, `facto`, `reco`, `rules`, `bayes`, `neighbors`, `multiclass`, `multioutput`, `proba`, `anomaly`, `drift`, `protect`), `deep/` (`neural`, comprese le reti per il linguaggio con la cache statica di chiavi e valori), `signals/` (`stream`, `stats`, `preprocessing`, `vision`, `datasets`), `robot/` (`control`), `evaluation/` (`evaluate`, `metrics`) e `infrastructure/` (`compose`, `data`). I vecchi percorsi, per esempio `dense_armor.utility.bandit`, restano validi (`_compat.py`).

| modulo | cosa fa |
|---|---|
| `orca.py` | `Orca` — lo scudo input/output a 4 fasi. Vedi README principale. |
| `curvature.py` | `curvature(x_current, x_reference, scale=1.0)` — calcola la derivata analitica esplicita di `sum((x-ref)²)`, normalizzata in `[0,1)`. Usata come misura di "quanto un punto si è allontanato" nello scudo di ingresso. `scale` fissa in unità reali il punto di saturazione (default 1.0, comportamento invariato); vedi `docs/api/curvature.md` per un caso reale (SO-101) in cui il default satura quasi subito e serve uno scale fisico per un segnale graduale. |
| `metro.py` | `Metro` — scaler logaritmico anti-underflow: proietta un valore infinitesimo in uno spazio numerico visibile (`enc`) e lo riporta indietro (`dec`). Usato dal rilevatore di deriva lenta di `Armor`. |
| `resonance_search.py` | `apply_fast_resonance` — cerca pattern noti confrontando un vettore query con una matrice di riferimento, pesando coseno-similarità "base" e "post-blend" (via `damping_operator`). Usato anche da `Orca` per richiamare, in modalità cieca, un riferimento pulito già visto in passato e simile all'input corrotto corrente (`_recall_reference`/`_remember_reference`, soglia `reference_recall_min_score=0.90`). |
| `iodat.py` | Caricamento dati: dataset HDF5/NetCDF. |
| `diagnostic.py` | `diag(iorig, ifilt)` — confronto differenziale tra un audio originale e uno filtrato (accetta sia percorsi file `.wav` che array NumPy), calcola varianza del residuo e fedeltà percentuale. |

---

## `$ algoritmi --detail`

Due meccaniche del codice che vale la pena capire a fondo se ci si mette le mani:

**Compressione logaritmica dello scudo di ingresso** (`orca.py`, `_execute_4_phase_input_shield`): prima di filtrare, ogni valore viene riscalato a `fact = 10^(val_e - log10(|v|))` (con `val_e = -4.0` di default) — questo comprime l'intero range di magnitudini in una banda stretta e controllata, così che lo stabilizzatore (tarato per operare su scala ~unitaria) si comporti in modo coerente sia che il dato originale sia `1e-3` sia che sia `1e6`. Il segno del valore originale si preserva automaticamente (il fattore di scala è sempre positivo). Alla fine si divide per lo stesso fattore per tornare alla scala originale.

**Fix: riferimento che attraversa lo zero**: `fact` dipende da `log10(|valore_pulito|)` — un riferimento pulito quasi-zero (anche solo un residuo di floating point, es. `sin(4π) ≈ -4.9e-16`, mai esattamente `0.0`) faceva esplodere `fact` e, con esso, il controvalore compresso del CORROTTO (che condivide lo stesso fattore ma non è vicino a zero) — un salto di ordini di grandezza rispetto ai punti vicini, sufficiente a sballare la memoria causale di `AdaptiveSignalStabilizer` per l'intero batch, non solo per quel punto. Due correzioni: (1) `exp10_cl` non viene più amplificato oltre la scala target (`val_e`) — un pulito già a/sotto quella scala non ha nulla da guadagnare da un fattore ancora più estremo; (2) il valore compresso del corrotto (`co_chunk`) viene comunque limitato a un multiplo ampio (20×) della scala target, indipendentemente dalla causa esatta del salto. Verificato: MSE su un riferimento sinusoidale che attraversa lo zero da ~9 (peggio della modalità cieca) a ~0.02 (`test/test_orca.py::test_riferimento_che_attraversa_lo_zero_non_esplode_piu`).


## `$ nota`

Le stime di throughput/performance del vecchio documento (analisi/secondo, moltiplicatori asintotici) sono state tolte da qui — non perché inventate: erano risultati di test veri fatti dall'autore, non fabbricati. Sono state rimosse solo perché non le ho ri-misurate personalmente in questa sessione, quindi non posso metterci la firma sopra in un documento che punta all'onestà verificabile. Chi vuole numeri di prestazione aggiornati può rilanciare quei test sul proprio hardware.
