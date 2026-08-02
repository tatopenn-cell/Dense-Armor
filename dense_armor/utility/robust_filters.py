# -*- coding: utf-8 -*-
"""
robust_filters.py -- quattro rilevatori di anomalie "a basso costo", nessuna
meccanica pesante (nessun modello dinamico, nessuno stato, nessuna
differenziazione automatica): statistiche classiche, note e usate da decenni
in telemetria/astronomia, ognuna calcolata su una finestra locale centrata
(come healing_filter.py in questo stesso pacchetto -- non causale, pensate
per pulizia offline/batch di una serie gia' registrata, non per il ciclo
real-time di core/hybrid_engine.py).

Ogni funzione ha la stessa firma (x, radius=..., **soglia) -> (pulito, anomalie):
    pulito   -- array 1D, stessa shape di x: valore grezzo dove non anomalo,
                mediana locale dove anomalo.
    anomalie -- lista di indici (int) segnalati come anomali.

I quattro metodi, in ordine di introduzione storica:
- Chauvenet (1863): soglia di rigetto basata sulla probabilita' attesa che un
  campione della finestra locale (size N) si discosti tanto per puro caso --
  se il numero atteso di tali eventi (N * P) e' < 0.5, il punto e' un
  outlier statisticamente "troppo improbabile per il campione a disposizione".
- Tukey's fences / IQR (anni '70): nessuna assunzione sulla distribuzione,
  solo quartili -- oltre Q1-1.5*IQR o Q3+1.5*IQR e' outlier.
- Hampel filter: mediana + MAD scalato (1.4826*MAD, costante di consistenza
  per una gaussiana) come soglia -- la generalizzazione robusta di "3 sigma"
  che non usa mai una media/std sensibile agli outlier che sta cercando.
- Sigma-clipping iterativo: standard in astronomia (stacking di immagini,
  fotometria) -- ricalcola media/std della finestra escludendo i punti oltre
  N sigma, ripete finche' stabile o max_iters, poi giudica il punto centrale
  contro le statistiche "ripulite".
"""
import math
from typing import List, Tuple

import numpy as np


def _window(x: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = len(x)
    lo, hi = max(0, i - radius), min(n, i + radius + 1)
    return x[lo:hi]


def chauvenet_criterion(x: np.ndarray, radius: int = 10) -> Tuple[np.ndarray, List[int]]:
    """Criterio di Chauvenet (1863) su finestra locale di raggio `radius`.

    Per il punto i, con la finestra come campione (media mu, std sigma, size
    N): d = |x[i] - mu| / sigma; P = erfc(d / sqrt(2)) (probabilita' a due code
    di un tale scostamento sotto ipotesi gaussiana). Se N*P < 0.5 -- il numero
    atteso di scostamenti pari a questo, ripetendo l'esperimento, e' inferiore
    a mezzo campione -- il punto e' rigettato come outlier.

    Nota: usa mean/std (non robuste), come nella formulazione originale --
    su una finestra con PIU' di un outlier, mean/std sono gia' distorte dagli
    altri outlier prima ancora di giudicare il punto i (limite noto e
    documentato del criterio originale, non specifico di questa implementazione;
    per finestre con outlier multipli preferire hampel_filter o tukey_fences,
    che usano statistiche robuste)."""
    n = len(x)
    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 3:
            continue
        mu, sigma = float(np.mean(w)), float(np.std(w))
        if sigma < 1e-12:
            continue
        d = abs(x[i] - mu) / sigma
        p = math.erfc(d / math.sqrt(2))
        if w.size * p < 0.5:
            anomalie.append(i)
            out[i] = float(np.median(w))
    return out, anomalie


def tukey_fences(x: np.ndarray, radius: int = 10, k: float = 1.5) -> Tuple[np.ndarray, List[int]]:
    """Tukey's fences / IQR su finestra locale di raggio `radius`.

    Q1/Q3 = 25/75-esimo percentile della finestra, IQR = Q3-Q1. Il punto i e'
    outlier se fuori da [Q1 - k*IQR, Q3 + k*IQR] (k=1.5 e' lo standard, k=3.0
    la variante "extreme outliers"). Nessuna assunzione sulla distribuzione --
    l'unico dei quattro a non presupporre normalita'."""
    n = len(x)
    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 4:
            continue
        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - k * iqr, q3 + k * iqr
        if x[i] < lo_fence or x[i] > hi_fence:
            anomalie.append(i)
            out[i] = float(np.median(w))
    return out, anomalie


def hampel_filter(x: np.ndarray, radius: int = 10, n_sigmas: float = 3.0) -> Tuple[np.ndarray, List[int]]:
    """Hampel filter su finestra locale di raggio `radius`.

    mediana e MAD (Median Absolute Deviation) della finestra, scalato per
    1.4826 (costante di consistenza: rende scaled_MAD uno stimatore
    non distorto della deviazione standard SE il dato fosse gaussiano,
    ma senza mai calcolare una media/std vera -- ogni singola statistica
    usata qui e' gia' robusta, a differenza di Chauvenet). Outlier se
    |x[i] - mediana| > n_sigmas * scaled_MAD."""
    n = len(x)
    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 3:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        scaled_mad = 1.4826 * mad
        if scaled_mad < 1e-12:
            if abs(x[i] - med) > 1e-9:
                anomalie.append(i)
                out[i] = med
            continue
        if abs(x[i] - med) > n_sigmas * scaled_mad:
            anomalie.append(i)
            out[i] = med
    return out, anomalie


def sigma_clip(x: np.ndarray, radius: int = 10, n_sigmas: float = 3.0, max_iters: int = 5) -> Tuple[np.ndarray, List[int]]:
    """Sigma-clipping iterativo (standard in astronomia: stacking, fotometria)
    su finestra locale di raggio `radius`.

    Sulla finestra (esclude il punto i stesso, cosi' un vero outlier al
    centro non puo' auto-proteggersi gonfiando la propria std): calcola
    media/std, scarta i punti oltre n_sigmas, ricalcola su quelli rimasti,
    ripete fino a stabilita' o max_iters. Il punto i e' outlier se fuori da
    n_sigmas dalla media/std FINALI (ripulite dagli altri outlier della
    finestra, a differenza di chauvenet_criterion che usa mean/std grezze)."""
    n = len(x)
    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neighbor_idx = [j for j in range(lo, hi) if j != i]
        if len(neighbor_idx) < 3:
            continue
        w = x[neighbor_idx].astype(float)
        mask = np.ones(w.shape, dtype=bool)
        for _ in range(max_iters):
            subset = w[mask]
            if subset.size < 2:
                break
            mu, sigma = float(np.mean(subset)), float(np.std(subset))
            if sigma < 1e-12:
                break
            new_mask = np.abs(w - mu) <= n_sigmas * sigma
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        subset = w[mask]
        if subset.size < 1:
            continue
        mu, sigma = float(np.mean(subset)), float(np.std(subset))
        if sigma < 1e-12:
            if abs(x[i] - mu) > 1e-9:
                anomalie.append(i)
                out[i] = float(np.median(subset))
            continue
        if abs(x[i] - mu) > n_sigmas * sigma:
            anomalie.append(i)
            out[i] = float(np.median(subset))
    return out, anomalie


def pressure_valve(
    x: np.ndarray, radius: int = 10, soglia_pressione: float = 8.0, n_sigmas: float = 3.0,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Orchestratore automatico dei quattro rilevatori sopra: non un voto
    (quanti metodi segnalano il punto), una combinazione a MINIMA VARIANZA
    vincolata -- lo stimatore BLUE (Best Linear Unbiased Estimator) classico
    della statistica, derivato con un moltiplicatore di Lagrange.

    Ogni metodo produce una coppia (centro, scala) locale: Chauvenet usa
    media/std (non robuste, la formulazione originale -- la debolezza e'
    voluta, vedi sotto), sigma-clipping usa media/std ripulite
    iterativamente dagli outlier della finestra, Hampel usa mediana/MAD
    scalato (1.4826*MAD), Tukey usa mediana/IQR scalato (IQR/1.349, la
    stessa costante che rende IQR uno stimatore di sigma per una gaussiana).
    Quattro stime indipendenti dello stesso "vero" centro locale, ognuna con
    la propria incertezza (scala).

    Combinarle non con una media semplice ma cercando i pesi w_k che
    minimizzano la VARIANZA della combinazione pesata Sum(w_k * centro_k),
    vincolati a Sum(w_k) = 1 (combinazione non distorta) -- un problema di
    ottimizzazione vincolata, risolto con un moltiplicatore di Lagrange
    lambda:
        L(w, lambda) = Sum(w_k^2 * scala_k^2) - lambda * (Sum(w_k) - 1)
        dL/dw_k = 0  =>  w_k = lambda / (2 * scala_k^2)
        dal vincolo  =>  w_k = (1/scala_k^2) / Sum_j(1/scala_j^2)
    Il risultato (peso inversamente proporzionale al QUADRATO della propria
    incertezza) e' lo stimatore a minima varianza tra tutte le combinazioni
    lineari non distorte di stime indipendenti (Gauss-Markov) -- non e' un
    peso scelto a mano, e' quello che il vincolo impone. Un metodo la cui
    scala si gonfia (es. Chauvenet quando la finestra contiene gia' un
    outlier, la sua std non e' robusta) viene automaticamente pesato meno,
    senza bisogno di scartarlo esplicitamente.

    centro_combinato = Sum(w_k * centro_k)
    scala_combinata  = 1 / sqrt(Sum(1/scala_k^2))   (la stessa derivazione
                        da' anche la varianza della combinazione stessa --
                        SEMPRE piu' stretta di qualunque scala_k singola,
                        e' la proprieta' che rende utile combinare piu'
                        stime indipendenti)
    pressione        = |x[i] - centro_combinato| / scala_combinata

    Una singola soglia finale (soglia_pressione) decide -- trigger binario,
    come il resto di questo ecosistema (core/hybrid_engine.py,
    dense_evolution/healing.py): mai un terzo stato intermedio, mai un
    blend tra grezzo e pulito. Non e' piu' un valore "in sigma" nel senso
    classico (scala_combinata e' piu' stretta di ogni scala_k, quindi la
    stessa deviazione reale produce una pressione piu' alta che con un
    singolo metodo) -- soglia_pressione=8.0 e' il default calibrato
    direttamente (vedi test/test_robust_filters.py): su rumore gaussiano
    puro (N=300, seed fisso) i percentili della pressione sono
    50%=1.4, 95%=4.6, 99%=6.0, 99.9%=7.6 -- 8.0 lascia 1/300 falsi
    positivi mantenendo comunque un vero outlier (pressione~88) e una
    coppia di outlier ravvicinati (pressione~40) ben sopra soglia con
    ampio margine. `n_sigmas` resta il parametro del sotto-passo
    sigma-clipping interno (quante sigma oltre cui un vicino viene escluso
    nell'iterazione, non la soglia finale).

    Nessuna configurazione richiesta dal chiamante oltre i default: pensato
    per un contesto dove chi legge il risultato non deve scegliere pesi o
    soglie per metodo, solo leggere `anomalie`/`pressione`. Verificato
    (test/test_robust_filters.py): pochi falsi positivi su rumore
    gaussiano puro, rileva un vero outlier con pressione ben sopra soglia,
    non tocca un gradino genuino sostenuto.

    Ritorna (pulito, anomalie, pressione): `pressione` e' un array 1D di
    float, la deviazione combinata continua per ogni punto (non solo la
    decisione binaria finale)."""
    n = len(x)
    pressione = np.zeros(n)

    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 4:
            continue

        centri, scale = [], []

        mu, sigma = float(np.mean(w)), float(np.std(w))
        if sigma > 1e-12:
            centri.append(mu)
            scale.append(sigma)

        med = float(np.median(w))
        q1, q3 = np.percentile(w, [25, 75])
        iqr_scale = (q3 - q1) / 1.349
        if iqr_scale > 1e-12:
            centri.append(med)
            scale.append(iqr_scale)

        mad = float(np.median(np.abs(w - med)))
        scaled_mad = 1.4826 * mad
        if scaled_mad > 1e-12:
            centri.append(med)
            scale.append(scaled_mad)

        neighbor_idx = [j for j in range(max(0, i - radius), min(n, i + radius + 1)) if j != i]
        if len(neighbor_idx) >= 3:
            w_neighbors = x[neighbor_idx].astype(float)
            mask = np.ones(w_neighbors.shape, dtype=bool)
            for _ in range(5):
                subset = w_neighbors[mask]
                if subset.size < 2:
                    break
                mu_s, sigma_s = float(np.mean(subset)), float(np.std(subset))
                if sigma_s < 1e-12:
                    break
                new_mask = np.abs(w_neighbors - mu_s) <= n_sigmas * sigma_s
                if np.array_equal(new_mask, mask):
                    break
                mask = new_mask
            subset = w_neighbors[mask]
            if subset.size >= 1:
                mu_s, sigma_s = float(np.mean(subset)), float(np.std(subset))
                if sigma_s > 1e-12:
                    centri.append(mu_s)
                    scale.append(sigma_s)

        if not centri:
            continue

        centri_arr = np.array(centri)
        scale_arr = np.array(scale)
        pesi = (1.0 / scale_arr ** 2)
        pesi /= pesi.sum()  # w_k = (1/scala_k^2) / Sum_j(1/scala_j^2), dal moltiplicatore di Lagrange

        centro_combinato = float(np.sum(pesi * centri_arr))
        scala_combinata = float(1.0 / np.sqrt(np.sum(1.0 / scale_arr ** 2)))

        if scala_combinata > 1e-12:
            pressione[i] = abs(x[i] - centro_combinato) / scala_combinata

    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        if pressione[i] > soglia_pressione:
            anomalie.append(i)
            out[i] = float(np.median(_window(x, i, radius)))
    return out, anomalie, pressione
