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

    out = np.copy(processed)
    trigger_arr = np.ones(n)  # punti 0,1 (mai valutati dal ciclo) restano pass-through
    fallback_triggered_at_all = False

    for i in range(2, n):
        lo = max(0, i - adaptive_radius_used)
        # finestra baseline presa da `out` (già guarito), non da `processed`
        # (grezzo): altrimenti un outlier passato (es. uno spike a i-3) resta
        # nella finestra locale nella sua forma originale non sanata e continua
        # a spostare la baseline verso l'alto/basso per `radius` passi dopo di
        # sé (verificato: senza questo, un gradino subito dopo uno spike isolato
        # veniva sovrastimato dalla baseline oltre il valore reale del gradino).
        baseline = rif[i] if rif is not None else np.mean(out[lo:i])

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
        window_raw = processed[lo:i]
        if window_raw.size >= 3:
            local_scale = float(np.std(np.diff(window_raw)))
        else:
            local_scale = 0.0
        scale = max(local_scale, 1e-6)

        state_A = jnp.array([baseline])
        state_B = jnp.array([processed[i]])

        # IPG invece preso da `processed` (grezzo), non da `out`: l'IPG deve
        # vedere se gli ultimi valori GREZZI si stanno davvero muovendo in una
        # nuova direzione. Usare `out` qui crea un blocco autoalimentato — se
        # un punto viene respinto (sostituito con la baseline), `out` non
        # mostra mai più alcuna evidenza del nuovo valore, quindi l'IPG
        # calcolato su `out` resta a zero per sempre e nessun gradino reale
        # può mai essere riconosciuto, per quanto a lungo sia sostenuto
        # (verificato: con IPG da `out`, un gradino di 10 punti restava
        # appiattito al 100%, nessuna via d'uscita).
        ipg_raw = np.array([processed[i - 1] - processed[i - 2]])
        norm_ipg_raw = np.linalg.norm(ipg_raw)
        ipg_vector = jnp.array(ipg_raw / norm_ipg_raw) if norm_ipg_raw > 1e-9 else jnp.array(ipg_raw)

        phi_ab = calculate_phi_ab(state_A, state_B, ipg_vector, jnp.float64(scale))
        E_A = jnp.linalg.norm(state_A)
        E_B = jnp.linalg.norm(state_B)
        v_dinamic = calculate_vettore_dinamico(E_A, E_B, phi_ab)
        trigger = float(evaluate_phi_trigger(v_dinamic))
        trigger_arr[i] = trigger

        if trigger > GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']:
            out[i] = processed[i]
        else:
            out[i] = baseline
            fallback_triggered_at_all = True

    metadata = {
        'fallback_triggered': fallback_triggered_at_all,
        'adaptive_radius_used': adaptive_radius_used,
    }
    return out, trigger_arr, metadata
