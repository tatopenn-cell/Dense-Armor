# -*- coding: utf-8 -*-
"""Regressione per il fix di direzione di K_anomalous in apply_damping_blend
(dense_armor/core/damping_operator.py): la formula originale
c_anom/(c_anom+diff) dava K_anomalous ALTO a differenza piccola e BASSO a
differenza grande -- l'opposto di quanto dichiarato nella docstring della
funzione. Questo test riproduce il protocollo usato per verificare il fix
(sinusoide + rumore gaussiano a 4 livelli, riferimento = mediana mobile,
30 seed per livello) e caratterizza sia il miglioramento (rumore medio/alto)
sia il limite noto non ancora risolto (rumore molto basso)."""
import numpy as np
import jax.numpy as jnp
from dense_armor.core.damping_operator import (
    apply_damping_blend, _PHI, _ALPHA, _SIGMA, _K_MAX_ZETA,
)

N_SEED = 30
LIVELLI_RUMORE = [0.05, 0.3, 1.0, 3.0]


def _segnale_pulito():
    t = np.linspace(0, 20, 500)
    return np.sin(t) * 5 + 10


def _mediana_mobile(x, w=3):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        a, b = max(0, i - w), min(n, i + w + 1)
        out[i] = np.median(x[a:b])
    return out


def _blend_originale_pre_fix(gradient_state, noise_matrix):
    """Reimplementazione della formula K_anomalous PRIMA del fix (direzione
    invertita), tenuta solo qui come riferimento per il confronto -- non
    esiste piu' nel codice di produzione."""
    eps = 1e-7
    is_nan_gradient = jnp.isnan(gradient_state)
    is_nan_noise = jnp.isnan(noise_matrix)
    g_safe = jnp.where(is_nan_gradient, noise_matrix, gradient_state)
    n_safe = jnp.where(is_nan_noise, g_safe, noise_matrix)
    diff = jnp.abs(g_safe - n_safe)
    coherence = jnp.clip(1.0 - (diff / (2.0 * jnp.maximum(jnp.abs(g_safe), 1e-3))), 0.0, 1.0)
    x_steer = jnp.clip(diff, 0.0, 5.0)
    steering = 1.0 / (1.0 + jnp.exp(-(x_steer - 2.0)))
    h_damping = steering * jnp.float32(_K_MAX_ZETA)
    c_anom = jnp.float32(1.0 / (_PHI ** 2))
    K_anomalous_raw = c_anom / (c_anom + diff + eps)  # formula pre-fix
    K_anomalous = jnp.clip(K_anomalous_raw, jnp.float32(_ALPHA), jnp.float32(_K_MAX_ZETA))
    K_operator = (1.0 - coherence) * h_damping + coherence * K_anomalous
    K_operator = jnp.where(jnp.logical_or(is_nan_gradient, is_nan_noise),
                            jnp.float32(1.0 - (_ALPHA - _SIGMA)), K_operator)
    k_min_bound = jnp.float32(_ALPHA - _SIGMA)
    k_max_bound = jnp.float32(1.0 - (_ALPHA - _SIGMA))
    K_operator = jnp.clip(K_operator, k_min_bound, k_max_bound)
    blended = (1.0 - K_operator) * g_safe + K_operator * n_safe
    return jnp.where(jnp.isnan(blended), n_safe, blended)


def _differenze_rmse(livello):
    """Per un livello di rumore, ritorna l'array (30,) di (rmse_fix - rmse_originale)."""
    clean = _segnale_pulito()
    diffs = np.empty(N_SEED)
    for seed in range(N_SEED):
        rng = np.random.default_rng(seed)
        rumoroso = clean + rng.normal(0, livello, size=clean.shape)
        rif = _mediana_mobile(rumoroso, w=3)
        out_fix = np.array(apply_damping_blend(jnp.array(rumoroso), jnp.array(rif)))
        out_orig = np.array(_blend_originale_pre_fix(jnp.array(rumoroso), jnp.array(rif)))
        rmse_fix = np.sqrt(np.mean((out_fix - clean) ** 2))
        rmse_orig = np.sqrt(np.mean((out_orig - clean) ** 2))
        diffs[seed] = rmse_fix - rmse_orig
    return diffs


def test_fix_batte_originale_a_rumore_medio_alto_molto_alto():
    """Margine minimo: differenza media > 3 deviazioni standard sui 30 seed,
    per non far scattare il test su semplice rumore statistico."""
    for livello in [0.3, 1.0, 3.0]:
        diffs = _differenze_rmse(livello)
        margine = 3.0 * diffs.std()
        assert diffs.mean() < -margine, (
            f"a rumore={livello}: il fix dovrebbe battere l'originale con margine "
            f"statisticamente significativo, media={diffs.mean():.4f} margine={margine:.4f}"
        )


def test_limite_noto_fix_peggiora_a_rumore_molto_basso():
    """Limite noto e documentato, NON un fallimento: a rumore molto basso
    (segnale gia' quasi pulito) il fix e' sistematicamente un po' peggiore
    dell'originale -- serve pesare di piu' coherence/h_damping invece di
    K_anomalous per risolverlo (non ancora fatto). Questo test conferma il
    limite noto invece di nasconderlo: se in futuro viene risolto, questo
    assert va aggiornato consapevolmente, non silenziato."""
    diffs = _differenze_rmse(0.05)
    margine = 3.0 * diffs.std()
    assert diffs.mean() > margine, (
        "limite noto atteso non riprodotto: il fix non risulta piu' "
        "sistematicamente peggiore dell'originale a rumore molto basso -- "
        "se e' stato risolto, aggiornare/rimuovere questo test consapevolmente"
    )
