# -*- coding: utf-8 -*-
r"""
Sentinel Metrology Framework - Adaptive Damping Blend Operator
=========================================================
Sottosistema: fonde due segnali (es. output grezzo e riferimento) con un
guadagno non lineare che dipende dalla loro distanza -- differenza piccola
tra i due -> si preferisce il segnale grezzo; differenza grande -> si
preferisce il riferimento, in modo smorzato e sempre limitato.
Autore del Framework: Salvatore Pennacchio (Napoli, 2026)
Percorso: shield_/core/damping_operator.py

Nota: non e' calcolo quantistico. I nomi delle costanti (PHI, ecc.) sono
iperparametri numerici fissi, non stati/operatori quantistici -- non c'e'
aritmetica complessa ne' spazio di Hilbert coinvolti.
"""

import jax
import jax.numpy as jnp
import numpy as np

# =========================================================================
# COSTANTI NUMERICHE FISSE (XLA-SAFE, precompilate per stabilita' binaria)
# =========================================================================
_PHI_128        = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0  # sezione aurea, usata solo come iperparametro
_ZETA_TOTAL_128 = np.longdouble(60.16762236065194)          # costante numerica fissa (limite superiore di damping)

# Cast a tipi primitivi per prevenire eccezioni hardware nei file binari (.bin)
_PHI        = float(_PHI_128)
_ALPHA      = float(0.25)                           # Base simmetrica (1/4)
_SIGMA      = float(np.longdouble(1.0) / (_PHI_128 ** 4)) # ~0.1458980337
_K_MAX_ZETA = float(_ZETA_TOTAL_128 / (_PHI_128 ** 4))  # Limite di damping superiore (~8.778)


@jax.jit
def apply_damping_blend(gradient_state: jnp.ndarray, noise_matrix: jnp.ndarray) -> jnp.ndarray:
    r"""
    Fonde `gradient_state` (segnale grezzo) e `noise_matrix` (riferimento)
    con un coefficiente K variabile:
    - differenza piccola tra i due segnali -> K basso, si preferisce il grezzo.
    - differenza grande -> K alto, si preferisce il riferimento (smorzamento).
    - [PATCH 2026]: isola e cancella i NaN a runtime prima del blend.
    """
    eps = 1e-7

    # =========================================================================
    # SANIFICAZIONE NaN (nessun valore non finito deve propagarsi nel blend)
    # =========================================================================
    is_nan_gradient = jnp.isnan(gradient_state)
    is_nan_noise = jnp.isnan(noise_matrix)

    gradient_state_safe = jnp.where(is_nan_gradient, noise_matrix, gradient_state)
    noise_matrix_safe = jnp.where(is_nan_noise, gradient_state_safe, noise_matrix)
    # =========================================================================

    # 1. Distanza assoluta tra i due segnali
    diff = jnp.abs(gradient_state_safe - noise_matrix_safe)

    # 2. Coerenza relativa: quanto la distanza e' piccola rispetto alla scala del segnale
    c_gate = jnp.float32(_ALPHA + _SIGMA)
    coherence = jnp.clip(1.0 - (diff / (2.0 * jnp.maximum(jnp.abs(gradient_state_safe), 1e-3))), 0.0, 1.0)

    # 3. Curva di damping a sigmoide: sale gradualmente con la distanza, satura a _K_MAX_ZETA
    x_steer = jnp.clip(diff, 0.0, 5.0)
    steering = 1.0 / (1.0 + jnp.exp(-(x_steer - 2.0)))
    h_damping = steering * jnp.float32(_K_MAX_ZETA)

    # 4. Guadagno anomalo: decresce iperbolicamente con la distanza, poi limitato
    c_anom = jnp.float32(1.0 / (_PHI ** 2))
    K_anomalous_raw = c_anom / (c_anom + diff + eps)
    K_anomalous = jnp.clip(K_anomalous_raw, jnp.float32(_ALPHA), jnp.float32(_K_MAX_ZETA))

    # 5. Combinazione dei due guadagni pesata dalla coerenza
    K_operator = (1.0 - coherence) * h_damping + coherence * K_anomalous

    # Se uno degli input originari era NaN, forziamo K al massimo smorzamento
    K_operator = jnp.where(jnp.logical_or(is_nan_gradient, is_nan_noise), jnp.float32(1.0 - (_ALPHA - _SIGMA)), K_operator)

    # Limitazione dinamica del guadagno per evitare blend degeneri (K=0 o K=1 assoluti)
    k_min_bound = jnp.float32(_ALPHA - _SIGMA)
    k_max_bound = jnp.float32(1.0 - (_ALPHA - _SIGMA))
    K_operator = jnp.clip(K_operator, k_min_bound, k_max_bound)

    # Blend finale: media pesata tra segnale grezzo e riferimento
    blended = (1.0 - K_operator) * gradient_state_safe + K_operator * noise_matrix_safe

    # Pulizia finale: nessun NaN deve uscire dalla funzione
    blended = jnp.where(jnp.isnan(blended), noise_matrix_safe, blended)

    return blended
