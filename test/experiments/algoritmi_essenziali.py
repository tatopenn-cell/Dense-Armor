"""
Versioni "all'osso" dei due meccanismi centrali dello shield (damping
adattivo e gating ABCollatz), separate dal pacchetto pubblicato: solo
numpy puro, un ciclo causale semplice, nessuna dipendenza da JAX/Orca/
compressione. Pensate per essere confrontabili con algoritmi classici
(mediana, Kalman...) sugli stessi tre criteri per cui quelli restano
fondamentali: semplicita', costo di calcolo, interpretabilita'.

Non toccano ne' sostituiscono lo shield vero (dense_armor/core/engine.py,
dense_armor/utility/collatz.py) -- sono una distillazione a parte, per
capire sperimentalmente (vedi test_algoritmi_essenziali.py) se e quanto
ciascun meccanismo contribuisce da solo, e se combinarli aiuta o no.
"""
import numpy as np


def damping_essenziale(
    serie: np.ndarray, soglia_base: float = 0.15, k_min: float = 0.10, k_max: float = 0.85
) -> np.ndarray:
    """Nucleo del damping adattivo di AdaptiveSignalStabilizer, ridotto
    all'essenziale: differenza rispetto al proprio output precedente ->
    volatilita' media mobile -> soglia dinamica -> gain basso (k_min) se
    coerente, alto (k_max) se anomalo. Niente compressione, niente
    coerenza spaziale/vicini, niente scaling L2 -- solo il meccanismo
    causale di base."""
    serie = np.asarray(serie, dtype=np.float64)
    out = np.empty_like(serie)
    out[0] = serie[0] if np.isfinite(serie[0]) else 0.0
    vol = 0.0
    alpha = 0.05
    for i in range(1, len(serie)):
        x = serie[i]
        x_safe = x if np.isfinite(x) else out[i - 1]
        diff = abs(x_safe - out[i - 1])
        vol = (1.0 - alpha) * vol + alpha * diff
        soglia_dinamica = soglia_base + vol
        anomalo = (diff > soglia_dinamica) or not np.isfinite(x)
        k = k_max if anomalo else k_min
        out[i] = (1.0 - k) * out[i - 1] + k * x_safe
    return out


def _radicale(n: int) -> int:
    """Prodotto dei fattori primi distinti di n (trial division, nessuna
    tabella precompilata -- versione leggibile, non quella vettorizzata
    JAX di calculate_jax_rad)."""
    n = int(abs(round(n)))
    if n == 0:
        return 0
    if n == 1:
        return 1
    r, m, p = 1, n, 2
    while p * p <= m:
        if m % p == 0:
            r *= p
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        r *= m
    return r


def _passo_collatz(n: int) -> int:
    n = int(n)
    return n // 2 if n % 2 == 0 else 3 * n + 1


def abcollatz_essenziale(
    serie: np.ndarray, k_min: float = 0.10, k_max: float = 0.85
) -> np.ndarray:
    """Nucleo del gating ABCollatz di evaluate_abc_discrepancy, ridotto
    all'essenziale: rumore rispetto al proprio output precedente -> indice
    intero -> un passo di Collatz -> discrepanza col radicale -> gate
    sigmoide -> blend. Auto-referenziale (usa il proprio output precedente
    come riferimento, non uno esterno) per restare confrontabile alla pari
    con damping_essenziale e con un Kalman causale."""
    serie = np.asarray(serie, dtype=np.float64)
    out = np.empty_like(serie)
    out[0] = serie[0] if np.isfinite(serie[0]) else 0.0
    for i in range(1, len(serie)):
        x = serie[i]
        x_safe = x if np.isfinite(x) else out[i - 1]
        noise = abs(x_safe - out[i - 1])
        idx = int(round(noise * 100)) + 3
        collatz_val = _passo_collatz(idx)
        discrepanza = abs(_radicale(collatz_val) - abs(x_safe))
        steering = 1.0 / (1.0 + np.exp(-discrepanza))
        gate = k_min + steering * (k_max - k_min)
        if not np.isfinite(x):
            gate = k_max
        out[i] = x_safe - gate * (x_safe - out[i - 1])
    return out


def combinato_essenziale(serie: np.ndarray) -> np.ndarray:
    """Damping essenziale seguito da ABCollatz essenziale in sequenza --
    stesso ordine (Stadio 1 -> Stadio 2) dello shield vero, coi due nuclei
    ridotti all'osso invece delle versioni JAX complete."""
    return abcollatz_essenziale(damping_essenziale(serie))
