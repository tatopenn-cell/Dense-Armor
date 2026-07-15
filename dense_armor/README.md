# `$ architettura interna`

Riferimento per chi vuole contribuire al codice, non per chi vuole solo installare e usare il pacchetto — per quello vedi il [README principale](../README.md). Qui si mappano tutti i moduli di `core/` e `utility/`, oltre a quelli già coperti (`damping_operator.py`, `engine.py`, `orca.py`, `collatz.py`).

---

## `$ core/`

| modulo | cosa fa |
|---|---|
| `damping_operator.py` | `apply_damping_blend` — fonde due tensori reali con un guadagno non lineare dipendente dalla loro distanza. Vedi README principale per i dettagli. |
| `engine.py` | `AdaptiveSignalStabilizer` — stabilizzatore causale ricorsivo (via `jax.lax.scan`): segue il segnale nel tempo, smorza dove la volatilità recente supera una soglia dinamica. Nucleo dello Stadio 1. |
| `compiler.py` | `DynamicAICodegen` — traduce sequenze di operazioni (`identity`, `relu`, `sigmoid`, `tanh`, `scale`, `dropout`, `clip`, `l2_normalize`) in pipeline JAX eseguibili via `jax.lax.switch`, con calcolo gradienti differenziabile (`compute_gradients`). |
| `chunk.py` | `ImageChunker` — spezza array/batch grandi in sotto-chunk (`split_array`/`merge_chunks`) per evitare colli di bottiglia di memoria; esegue pipeline del compilatore a blocchi fissi senza forzare ricompilazioni JIT. |
| `memory.py` | `UniversalMemoryGuard` — controlla RAM libera (e VRAM via `nvidia-smi`, se disponibile) prima di allocazioni pesanti; solleva `MemoryPressureError` se sotto soglia. `calculate_optimal_chunks` stima il partizionamento sicuro. |
| `tensor.py` | `TensorVault` — libreria di matrici di trasformazione statiche (`invert`, `identity`, `edge_detector`, `blend`) e parametriche (`scale_project`, `amplify`, `bias_shift`), con rilevamento automatico di backend/precisione. |
| `vector.py` | `BitwisePermutationEngine` (swap di elementi via maschere di bit su vettori di dimensione 2ⁿ) e `ParametricScenarioSimulator` (simulazioni Monte Carlo parallele via `vmap`, più `collapse_decision` per una scelta stocastica pesata da una distribuzione). |
| `noise.py` | `AIHardwareProfiler` (rileva CPU/GPU/TPU e imposta limiti di sicurezza sui tensori) e `StochasticAdversarialNoise` (inietta `bitflip`/`dropout_noise`/`gaussian_blur` per testare la robustezza). |
| `profiler.py` | `PipelineProfiler` — misura la latenza JIT in microsecondi, separando il warm-up di compilazione XLA dal regime stabile. |
| `logger.py` | Due formatter di logging: uno per console leggibile, uno JSON compatto per telemetria strutturata. |
| `visualizer.py` | `AIEngineVisualizer` — esporta report di provenienza firmati SHA-256. |

---

## `$ utility/`

| modulo | cosa fa |
|---|---|
| `orca.py` | `Orca` — lo scudo input/output a 4 fasi. Vedi README principale. |
| `collatz.py` | `ABCollatz` — gating basato sulla congettura di Collatz. Vedi sotto per il dettaglio dell'algoritmo. |
| `curvature.py` | `curvature(x_current, x_reference)` — calcola la derivata analitica esplicita di `sum((x-ref)²)`, normalizzata in `[0,1)`. Usata come misura di "quanto un punto si è allontanato" nello scudo di ingresso. |
| `metro.py` | `Metro` — scaler logaritmico anti-underflow: proietta un valore infinitesimo in uno spazio numerico visibile (`enc`) e lo riporta indietro (`dec`). Usato dal rilevatore di deriva lenta di `Armatura`. |
| `resonance_search.py` | `apply_fast_resonance` — cerca pattern noti confrontando un vettore query con una matrice di riferimento, pesando coseno-similarità "base" e "post-blend" (via `damping_operator`). |
| `anwav.py` / `iodat.py` | Caricamento dati: forme d'onda audio grezze e dataset HDF5/NetCDF. |
| `diagnostic.py` | `diag(iorig, ifilt)` — confronto differenziale tra un audio originale e uno filtrato (accetta sia percorsi file `.wav` che array NumPy), calcola varianza del residuo e fedeltà percentuale. |

---

## `$ algoritmi --detail`

Due meccaniche del codice che vale la pena capire a fondo se ci si mette le mani:

**Compressione logaritmica dello scudo di ingresso** (`orca.py`, `_execute_4_phase_input_shield`): prima di filtrare, ogni valore viene riscalato a `fact = 10^(val_e - log10(|v|))` (con `val_e = -4.0` di default) — questo comprime l'intero range di magnitudini in una banda stretta e controllata, così che lo stabilizzatore (tarato per operare su scala ~unitaria) si comporti in modo coerente sia che il dato originale sia `1e-3` sia che sia `1e6`. Il segno del valore originale si preserva automaticamente (il fattore di scala è sempre positivo). Alla fine si divide per lo stesso fattore per tornare alla scala originale.

**Radicale di Collatz** (`collatz.py`, `calculate_jax_rad`): calcola il prodotto dei fattori primi distinti di un numero, `rad(n) = exp(Σ log(p_i))`, usando l'esponenziale della somma dei logaritmi invece di un ciclo condizionale — questo lo rende differenziabile e vettorizzabile su GPU senza `if`/`while`, cosa che JAX non gestirebbe altrimenti in modo efficiente in un contesto JIT-compilato.

---

## `$ nota`

Le stime di throughput/performance del vecchio documento (analisi/secondo, moltiplicatori asintotici) sono state tolte da qui — non perché inventate: erano risultati di test veri fatti dall'autore, non fabbricati. Sono state rimosse solo perché non le ho ri-misurate personalmente in questa sessione, quindi non posso metterci la firma sopra in un documento che punta all'onestà verificabile. Chi vuole numeri di prestazione aggiornati può rilanciare quei test sul proprio hardware.
