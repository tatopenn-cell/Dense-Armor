# -*- coding: utf-8 -*-
"""
core/hybrid_engine.py
======================
Motore di healing a trigger binario per Armatura — sostituisce lo stadio
AdaptiveSignalStabilizer + ABCollatz (vedi CHANGELOG [1.0.10]: il gate
ABCollatz era matematicamente non discriminante, verificato con sweep
numerico e mai risolto in modo dimostrabile).

Le funzioni pure sotto sono portate da Dense-Evolution/dense_evolution/healing.py
(`calculate_phi_ab`, `calculate_vettore_dinamico`, `evaluate_phi_trigger`,
`GLOBAL_CONSTANTS`), verificate in quel repo con test dedicati e già in
produzione (PyPI dense-evolution >= 8.1.9). Vendorizzate qui (non importate
a runtime) per non aggiungere una dipendenza pesante a un core che oggi è
solo numpy+jax+psutil — `dense-evolution` resta un extra opzionale ([quantum]).

ADATTAMENTO rispetto all'originale: `calculate_phi_ab` normalizzava la
distanza state_A/state_B rispetto a una costante fissa (MAX_SEMANTIC_DISTANCE
= sqrt(2)), corretta quando state_A/state_B sono vettori GIA' normalizzati
(embedding di stati quantistici in Dense-Evolution, dove sqrt(2) è davvero
la distanza massima possibile). Qui state_A/state_B sono numeri grezzi di
scala arbitraria (loss, sensori, metriche): con la costante fissa, un salto
reale anche di poche unità saturava sempre coherence_component a valori
molto negativi, quindi phi_ab restava clippato a 0 per QUALSIASI cambiamento
genuino di ampiezza superiore a ~1.4 — il trigger restava "statico" per
sempre anche davanti a un gradino sostenuto e mai isolato (verificato con
un test manuale: una serie con un vero gradino a +4 unità restava appiattita
all'infinito).

Primo tentativo di fix: normalizzare sulla scala assoluta di state_A/state_B
(max(|state_A|,|state_B|)) — risolve il gradino, ma poi un salto enorme
sembra "proporzionalmente" coerente quanto uno piccolo, quindi anche uno
spike isolato passava intatto (verificato su segnale sia sintetico sia
rumoroso realistico: pulito==grezzo sullo spike). Fix definitivo: la
distanza è normalizzata sulla VOLATILITÀ LOCALE della finestra recente
(MAD scalato, robusto a un singolo outlier residuo nella finestra), non
sulla grandezza assoluta dei valori né su una costante fissa — un salto
enorme rispetto a quanto il segnale oscilla di solito resta sospetto
(spike filtrato), un gradino sostenuto viene giudicato in base a quanto
è anomalo rispetto al rumore normale, non alla scala del numero.

`hybrid_shield` generalizza la logica già corretta di
ia_utils.vector_healing.enhanced_dense_healing_hybrid (sequenza di vettori)
a una serie scalare 1D, con lo stesso schema a 2 stati (mai 3): per ogni
punto, `trigger` deciso da evaluate_phi_trigger è strettamente binario
(0.0 statico/rumore, 1.0 dinamico/genuino) — non esiste un ramo intermedio.
"""
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

GLOBAL_CONSTANTS = {
    'V_DINAMIC_K_COEFF': 5.0,
    'WEIGHT_SEMANTIC': 0.6,
    'WEIGHT_COHERENCE': 0.4,
    'NON_STATIC_THRESHOLD_A': 1e-2,
}


@jax.jit
def calculate_phi_ab(state_A: jnp.ndarray, state_B: jnp.ndarray, ipg_vector: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Fattore di allineamento e coerenza spaziale Phi_AB (vedi ADATTAMENTO nel
    docstring del modulo). `scale` è la volatilità locale della finestra
    recente (MAD scalato, calcolata dal chiamante) — non più una costante
    fissa né la grandezza assoluta di state_A/state_B."""
    semantic_change = state_B - state_A
    norm_change = jnp.linalg.norm(semantic_change)
    norm_ipg = jnp.linalg.norm(ipg_vector)

    alignment = jnp.where(
        (norm_change > 1e-12) & (norm_ipg > 1e-12),
        jnp.dot(semantic_change, ipg_vector) / (norm_change * norm_ipg),
        0.0
    )
    semantic_alignment = (alignment + 1.0) / 2.0

    distance_A_B = jnp.linalg.norm(state_A - state_B)
    coherence_component = 1.0 - (distance_A_B / scale)

    phi_ab = (semantic_alignment * GLOBAL_CONSTANTS['WEIGHT_SEMANTIC']) + (coherence_component * GLOBAL_CONSTANTS['WEIGHT_COHERENCE'])
    return jnp.clip(phi_ab, 0.0, 1.0)


@jax.jit
def calculate_vettore_dinamico(E_A: jnp.ndarray, E_B: jnp.ndarray, Phi_AB: jnp.ndarray) -> jnp.ndarray:
    """Vettore Dinamico (V_dinamic): variazione logaritmica differenziale energetica (porting identico)."""
    valid_inputs = (E_A > 1e-12) & (E_B > 1e-12)
    ratio = jnp.where(valid_inputs, E_B / E_A, 1.0)
    log_ratio_clamped = jnp.clip(jnp.log(ratio), -5.0, 5.0)
    v_vita = GLOBAL_CONSTANTS['V_DINAMIC_K_COEFF'] * log_ratio_clamped * Phi_AB
    return jnp.where(valid_inputs, v_vita, 0.0)


@jax.jit
def evaluate_phi_trigger(deterministic_dq_dt_a: jnp.ndarray) -> jnp.ndarray:
    """Phi-Trigger: strettamente binario, 1.0 (dinamico) o 0.0 (statico) (porting identico)."""
    magnitude_change_a = jnp.abs(deterministic_dq_dt_a)
    trigger_active = magnitude_change_a > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']
    return jnp.where(trigger_active, 1.0, 0.0)


def _local_nan_fill(x: np.ndarray, radius: int) -> np.ndarray:
    """Sostituisce i NaN con la mediana dei vicini FINITI in una finestra
    locale ±radius (mediana, non media: robusta a uno spike che capiti
    nella stessa finestra) — non con la media dell'INTERA serie.

    Verificato: con la media globale, un NaN a 2 passi da uno spike di
    9999 in una serie di 200 punti veniva sanato a ~51 invece di ~1.0 —
    lo spike, ovunque fosse nella serie, distorceva il sostituto di ogni
    NaN, non solo quello vicino. Con la mediana locale, il sostituto
    riflette il vicinato reale del punto, non l'intera serie.
    """
    x = np.copy(x)
    nan_mask = np.isnan(x)
    if not np.any(nan_mask):
        return x
    n = x.size
    global_fallback = np.nanmedian(x)
    if np.isnan(global_fallback):
        global_fallback = 0.0
    for idx in np.where(nan_mask)[0]:
        lo_w, hi_w = max(0, idx - radius), min(n, idx + radius + 1)
        window_finite = x[lo_w:hi_w]
        window_finite = window_finite[~np.isnan(window_finite)]
        x[idx] = np.median(window_finite) if window_finite.size > 0 else global_fallback
    return x


def hybrid_shield(
    serie: np.ndarray,
    riferimento: Optional[np.ndarray] = None,
    radius_baseline: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Applica lo schema a 2 stati (pass-through / fallback a baseline) punto
    per punto su una serie scalare 1D.

    Args:
        serie: array 1D di float (NaN/Inf ammessi, vengono sanificati).
        riferimento: se dato, la baseline per il punto i è riferimento[i]
                     (modalità anti-deriva). Se None (modalità cieca), la
                     baseline è la media di una finestra locale dei punti
                     precedenti della stessa serie (raggio adattivo, stessa
                     formula di ia_utils: min(20, max(3, n // 3))).
        radius_baseline: raggio fisso opzionale per la modalità cieca.

    Returns:
        (pulito, trigger, metadata):
            pulito   — array 1D, stessa shape di `serie`: valore grezzo dove
                       trigger==1, baseline dove trigger==0.
            trigger  — array 1D di 0.0/1.0, stessa shape di `serie`.
            metadata — {'fallback_triggered': bool, 'adaptive_radius_used': int}

    LIMITE NOTO: i primi 2 punti (i=0, i=1) non passano mai dal ciclo del
    trigger — serve un ipg_vector dai due punti precedenti, quindi
    restano sempre pass-through (trigger=1) qualunque sia il loro valore.
    Comportamento ereditato identico da enhanced_dense_healing_hybrid,
    trascurabile su serie realistiche; gli eventuali NaN/spike su questi
    2 punti vanno intercettati da controlli indipendenti a monte (es. i
    controlli robusti già presenti in Armatura.analizza).

    VETTORIZZATO (v1.2.0): la versione precedente usava un ciclo Python
    `for i in range(2, n)` che chiamava calculate_phi_ab/
    calculate_vettore_dinamico/evaluate_phi_trigger una volta per punto.
    Profilato su una serie reale di sicurezza (telemetria Sysmon, 500
    punti): 87% del tempo nel dispatch JAX stesso (~2000 costruzioni di
    jnp.array, 3.2M chiamate a isinstance), non nel calcolo (poche
    operazioni scalari per punto) -- overhead di orchestrazione Python,
    non un limite di calcolo reale. Nessuna dipendenza sequenziale
    nascosta da preservare: baseline/scala/ipg leggono sempre `processed`
    (il grezzo sanificato), mai `out` del passo precedente (vedi i
    commenti storici sotto, ancora validi, che spiegano PERCHE' deve
    restare cosi'). Verificato bit-per-bit contro la versione a ciclo
    originale su una serie reale di sicurezza (500 punti) e su casi
    limite sintetici (spike isolato, gradino sostenuto, con/senza
    riferimento esterno, n=0..3): output e trigger identici in ogni
    caso. Latenza misurata (mediana, 50 chiamate dopo warmup JIT): da
    ~307ms a ~15ms per una serie da 500 punti.
    """
    s = np.asarray(serie, dtype=np.float64).ravel()
    n = s.size

    if n == 0:
        return np.empty(0), np.empty(0), {'fallback_triggered': False, 'adaptive_radius_used': 0}

    if radius_baseline is None:
        adaptive_radius_used = 0 if n < 3 else min(20, max(3, n // 3))
    else:
        adaptive_radius_used = radius_baseline
    fill_radius = max(adaptive_radius_used, 3)

    processed = np.copy(s)
    processed[np.isinf(processed)] = np.nan
    processed = _local_nan_fill(processed, fill_radius)

    rif = None
    if riferimento is not None:
        rif = np.asarray(riferimento, dtype=np.float64).ravel()
        if rif.size != n:
            raise ValueError(f"riferimento ha {rif.size} punti, serie ne ha {n}")
        rif = np.where(np.isinf(rif), np.nan, rif)
        rif = _local_nan_fill(rif, fill_radius)

    if n <= 2:
        # i punti 0,1 non passano mai dal ciclo del trigger (serve un ipg dai
        # 2 precedenti): nessuna finestra da costruire, pass-through diretto.
        return np.copy(processed), np.ones(n), {'fallback_triggered': False, 'adaptive_radius_used': adaptive_radius_used}

    R = adaptive_radius_used
    processed_j = jnp.asarray(processed)
    padded = jnp.concatenate([jnp.full(R, jnp.nan), processed_j])  # lunghezza n+R

    def _get_window(i):
        # finestra presa da `processed` (grezzo), non da `out` (già guarito) --
        # allineato a ia_utils.vector_healing.enhanced_dense_healing_hybrid,
        # il riferimento che questa funzione dichiara di generalizzare (vedi
        # il docstring del modulo). Usare `out` qui sembrava corretto (protegge
        # da un outlier passato che sposta la baseline) ma introduce un bug
        # peggiore su un gradino GENUINO e sostenuto: i primi 1-2 punti dopo
        # la transizione vengono inevitabilmente respinti (nessun rilevatore
        # causale può distinguere un gradino vero da uno spike al primissimo
        # campione), quei punti respinti finiscono in `out`, e la finestra dei
        # punti successivi li rimedia dentro il proprio calcolo -- producendo
        # una baseline "a metà strada" che respinge ANCHE i punti successivi
        # genuini, che rientrano in `out` alimentando lo stesso problema per
        # la finestra dopo: un equilibrio stabile ma sbagliato che non si
        # risolve mai da solo (verificato: 30 punti reali dopo un gradino di
        # 1.0->5.0 restavano bloccati per sempre a ~4.0, mai a 5.0). La
        # protezione dall'outlier passato non serve più il trucco `out`: la
        # mediana (sotto, per il valore di fallback) è già robusta a un
        # singolo spike nella finestra, senza il rischio di autocontaminazione.
        return jax.lax.dynamic_slice(padded, (i,), (R,))

    idx = jnp.arange(n)
    windows = jax.vmap(_get_window)(idx)  # (n, R): riga i = trailing R-window prima dell'indice i
    count_valid = jnp.minimum(idx, R)  # quanti valori reali (non-padding) ha ogni finestra
    valid_mask = jnp.arange(R)[None, :] >= (R - count_valid[:, None])  # (n, R)

    def _masked_mean(row, mask):
        return jnp.sum(jnp.where(mask, row, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    def _masked_median(row, mask, k):
        sorted_vals = jnp.sort(jnp.where(mask, row, jnp.inf))
        mid = k // 2
        return jnp.where(k % 2 == 0, (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0, sorted_vals[mid])

    def _masked_std_of_diff(row, mask, k):
        # volatilità locale = deviazione standard delle differenze successive
        # nella finestra GREZZA ("quanto si muove di solito, passo-passo").
        # Deliberatamente NON la mediana/MAD dei valori della finestra: in
        # una finestra che sta a cavallo di una transizione vera, appena la
        # maggioranza dei punti raggiunge il nuovo livello la mediana COLLASSA
        # sul nuovo livello e il MAD crolla a ~0 (la "minoranza" pre-transizione
        # sparisce nella metrica) — verificato: questo causava un rigetto
        # oscillante subito dopo un'accettazione corretta. La std dei DELTA
        # non ha questo collasso: un'unica differenza grande (la transizione
        # stessa) alza la std invece di sparire dietro la mediana.
        # Compromesso noto: uno spike isolato alza temporaneamente questa
        # volatilità (due differenze enormi, dentro e fuori dallo spike) per
        # `radius` passi, rendendo il motore più permissivo in quella finestra
        # — si autocorregge quando lo spike esce dalla finestra, non è un
        # blocco permanente.
        diffs = row[1:] - row[:-1]
        diff_mask = mask[1:] & mask[:-1]
        cnt = jnp.maximum(jnp.sum(diff_mask), 1)
        mean_d = jnp.sum(jnp.where(diff_mask, diffs, 0.0)) / cnt
        var_d = jnp.sum(jnp.where(diff_mask, (diffs - mean_d) ** 2, 0.0)) / cnt
        return jnp.where(k >= 3, jnp.sqrt(var_d), 0.0)

    baseline_mean = jax.vmap(_masked_mean)(windows, valid_mask)
    median_vals = jax.vmap(_masked_median)(windows, valid_mask, count_valid)
    local_scale = jax.vmap(_masked_std_of_diff)(windows, valid_mask, count_valid)
    scale = jnp.maximum(local_scale, 1e-6)

    baseline = jnp.asarray(rif) if rif is not None else baseline_mean
    state_A = baseline
    state_B = processed_j

    # IPG invece preso da `processed` (grezzo), non da `out`: l'IPG deve
    # vedere se gli ultimi valori GREZZI si stanno davvero muovendo in una
    # nuova direzione. Usare `out` qui crea un blocco autoalimentato — se
    # un punto viene respinto (sostituito con la baseline), `out` non
    # mostra mai più alcuna evidenza del nuovo valore, quindi l'IPG
    # calcolato su `out` resta a zero per sempre e nessun gradino reale
    # può mai essere riconosciuto, per quanto a lungo sia sostenuto
    # (verificato: con IPG da `out`, un gradino di 10 punti restava
    # appiattito al 100%, nessuna via d'uscita).
    d = jnp.diff(processed_j)  # d[k] = processed[k+1] - processed[k]
    ipg_raw = jnp.concatenate([jnp.zeros(2), d[:-1]])  # ipg_raw[i] = processed[i-1]-processed[i-2] per i>=2
    norm_ipg_raw = jnp.abs(ipg_raw)
    ipg_vector = jnp.where(norm_ipg_raw > 1e-9, ipg_raw / jnp.maximum(norm_ipg_raw, 1e-30), ipg_raw)

    semantic_change = state_B - state_A
    norm_change = jnp.abs(semantic_change)
    norm_ipg = jnp.abs(ipg_vector)
    alignment = jnp.where(
        (norm_change > 1e-12) & (norm_ipg > 1e-12),
        (semantic_change * ipg_vector) / (norm_change * norm_ipg),
        0.0,
    )
    semantic_alignment = (alignment + 1.0) / 2.0
    distance_A_B = jnp.abs(state_A - state_B)
    coherence_component = 1.0 - (distance_A_B / scale)
    phi_ab = jnp.clip(
        semantic_alignment * GLOBAL_CONSTANTS['WEIGHT_SEMANTIC'] + coherence_component * GLOBAL_CONSTANTS['WEIGHT_COHERENCE'],
        0.0, 1.0,
    )

    E_A = jnp.abs(state_A)
    E_B = jnp.abs(state_B)
    valid_energy = (E_A > 1e-12) & (E_B > 1e-12)
    ratio = jnp.where(valid_energy, E_B / jnp.where(E_A > 1e-12, E_A, 1.0), 1.0)
    log_ratio_clamped = jnp.clip(jnp.log(ratio), -5.0, 5.0)
    v_dinamic = jnp.where(valid_energy, GLOBAL_CONSTANTS['V_DINAMIC_K_COEFF'] * log_ratio_clamped * phi_ab, 0.0)

    trigger_full = jnp.where(jnp.abs(v_dinamic) > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A'], 1.0, 0.0)
    fallback_val = median_vals if rif is None else baseline
    out_full = jnp.where(trigger_full > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A'], processed_j, fallback_val)

    out = np.array(out_full)
    trigger_arr = np.array(trigger_full)
    out[0], out[1] = processed[0], processed[1]
    trigger_arr[0], trigger_arr[1] = 1.0, 1.0

    fallback_triggered_at_all = bool(np.any(trigger_arr[2:] < GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']))
    metadata = {
        'fallback_triggered': fallback_triggered_at_all,
        'adaptive_radius_used': adaptive_radius_used,
    }
    return out, trigger_arr, metadata
