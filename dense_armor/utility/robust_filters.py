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


def _jensen_shannon(a: np.ndarray, b: np.ndarray, n_bins: int = None, pseudo: float = 1.0) -> float:
    """Divergenza di Jensen-Shannon (in bit, log base 2) tra le distribuzioni
    empiriche di `a` e `b` -- stessi bin per entrambe (istogramma sull'unione
    dei due range). 0.0 = distribuzioni identiche, 1.0 = supporti
    completamente disgiunti. Pura aritmetica su numpy (nessuna dipendenza da
    scipy), stessa filosofia "leggera" degli altri rilevatori in questo modulo.

    Due accorgimenti per finestre piccole (verificato: senza questi, la JSD
    risultava PIU' alta su rumore gaussiano puro stazionario che vicino a
    una vera transizione di livello -- il segnale nella direzione sbagliata,
    puro artefatto di pochi campioni per bin):
    - n_bins adattivo (default None -> max(3, min(10, (len(a)+len(b))//6)),
      circa 6 campioni per bin invece di un numero fisso indipendente dalla
      dimensione della finestra;
    - smoothing di Laplace vero (pseudo=1.0, uno pseudo-conteggio per bin,
      non un epsilon quasi-zero solo per evitare log(0)) -- su pochi
      campioni due istogrammi sparsi sembrano gia' "diversi" per puro
      rumore di campionamento; lo smoothing li avvicina finche' una
      differenza reale nella distribuzione non emerge comunque sopra di
      esso. Verificato su rumore stazionario vs un vero gradino 1.0->5.0
      (radius=10): media JSD stazionaria 0.018-0.025, media JSD vicino
      alla transizione 0.04-0.08 (2-3x piu' alta, direzione corretta)."""
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi - lo < 1e-12:
        return 0.0
    if n_bins is None:
        n_bins = max(3, min(10, (a.size + b.size) // 6))
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)
    p = p.astype(float) + pseudo
    q = q.astype(float) + pseudo
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


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
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.copy(x)
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
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.copy(x)
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
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.copy(x)
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
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.copy(x)
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
    ref_mult: int = 3, k_molla: float = 3.0,
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
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

    LA MOLLA (JSD): la soglia stessa non e' fissa. Ad ogni punto si confronta
    la finestra locale (raggio `radius`, la stessa usata dai 4 metodi) con
    una finestra di riferimento piu' ampia (raggio `radius * ref_mult`,
    default ref_mult=3 -- la stessa costante di healing_filter.py in questo
    pacchetto) via la divergenza di Jensen-Shannon (_jensen_shannon, 0..1).
    Se le due distribuzioni sono simili (rumore stazionario, nessuna vera
    transizione in corso), JSD~0 e la soglia resta soglia_pressione. Se
    sono molto diverse (una transizione di regime vera e' in corso: la
    finestra locale, piu' vicina al nuovo livello, non somiglia piu' alla
    finestra di riferimento, ancora dominata dal vecchio), JSD sale e la
    soglia si allarga proporzionalmente:
        soglia_effettiva[i] = soglia_pressione * (1 + k_molla * JSD[i])
    "la molla cede" -- lo scudo diventa meno nervoso proprio nei punti dove
    un cambiamento genuino e' piu' plausibile, senza mai smettere di
    giudicare in binario (anomalia sopra soglia_effettiva, nient'altro).

    _jensen_shannon usa gia' n_bins adattivo e smoothing di Laplace (vedi il
    suo docstring) apposta per questo uso: su finestre piccole, una JSD
    "ingenua" (bin fissi, epsilon quasi-zero) risultava PIU' alta su rumore
    stazionario puro che vicino a una vera transizione -- la molla si sarebbe
    attivata ovunque, non "solo dove serve". Con la versione corretta,
    verificato su rumore stazionario a bassa ampiezza vs un vero gradino
    1.0->5.0 (radius=10, k_molla=3.0): soglia_effettiva media ~8.3 sul
    rumore stazionario, ~8.7-8.8 in prossimita' della transizione (piu'
    alta, direzione giusta) -- comunque un segnale rumoroso per natura (non
    separa mai perfettamente le due code), ma la soglia puo' solo ALLARGARSI
    (JSD >= 0), mai restringersi sotto soglia_pressione: anche nel caso
    peggiore (molla che si attiva "a sproposito" su rumore), il risultato e'
    uno scudo leggermente piu' permissivo, mai piu' nervoso.

    Nessuna configurazione richiesta dal chiamante oltre i default: pensato
    per un contesto dove chi legge il risultato non deve scegliere pesi o
    soglie per metodo, solo leggere `anomalie`/`pressione`. Verificato
    (test/test_robust_filters.py): pochi falsi positivi su rumore
    gaussiano puro, rileva un vero outlier con pressione ben sopra soglia,
    non tocca un gradino genuino sostenuto.

    LIMITE NOTO, ORA RISOLTO -- mascheramento reciproco tra outlier
    ravvicinati su una baseline quasi piatta (test/test_pressure_valve_vs_
    orca_experiment.py, scenario "impulsi alternati" +/-500 su fondo
    costante 120, zero rumore reale): la finestra locale usata per ogni
    stima ESCLUDE gia' il punto i stesso (fix applicato qui sotto), ma se
    altri outlier della stessa esplosione restano nella finestra (qui:
    raggio 10, 3 picchi entro 2 passi l'uno dall'altro), mediana/IQR e
    mediana/MAD restano comunque ESATTAMENTE 0 -- non e' un bug della loro
    implementazione, e' matematicamente corretto: con 18/20 punti della
    finestra identici, MAD e IQR (breakdown ~50%/~25%) non vedono niente
    di anomalo. Un M-estimator di Huber (IRLS, vedi Sung et al. 2019,
    arXiv:1912.04982, usato li' per spettroscopia a 2 qubit con outlier
    sperimentali) e' stato provato come alternativa e converge ANCH'ESSO a
    scala 0, per lo stesso motivo di fondo (la sua scala finale e' a sua
    volta una MAD dei residui). Letto direttamente il paper (non assunto
    dalla formula): la loro tecnica robusta funziona perche' la scala
    sigma^2=var(O_tau) viene stimata da MISURE RIPETUTE sullo STESSO punto
    (rumore di shot, M ripetizioni proiettive), un asse dati ortogonale
    alla finestra temporale -- non disponibile qui (una singola serie
    scalare, nessuna ripetizione per punto), quindi la loro soluzione non
    e' portabile cosi' com'e'.

    Provata anche una scala di fallback dalla finestra di riferimento piu'
    ampia (radius*ref_mult): VERIFICATA NON FUNZIONARE, e per un motivo
    strutturale, non un errore di implementazione -- allargare la finestra
    riduce sempre la FRAZIONE di outlier rispetto al totale (3 punti su 21
    locali = 14%, 3 su 61 di riferimento = 5%), quindi non puo' mai far
    superare a IQR/MAD (stimatori a breakdown frazionario fisso) la soglia
    di rottura che li fa collassare a zero: verificato numericamente,
    entrambe le scale restavano 0.0 anche sulla finestra di riferimento.

    LA VERA CORREZIONE: quando IQR o MAD collassano davvero a zero
    (finestra locale genuinamente omogenea) ma il punto i si scosta
    comunque dalla mediana locale, quello NON e' "nessuna informazione" da
    scartare -- e' la prova piu' forte possibile (deviazione reale contro
    dispersione locale nulla). Il bug era scartare questa evidenza insieme
    allo stimatore degenere. Ora: scala-locale-nulla + deviazione reale =>
    pressione forzata a infinito (anomalia certa), bypassando la
    combinazione pesata per quel punto. Verificato: sullo scenario reale
    "impulsi alternati" (+/-500 su fondo 120 senza rumore), prima del fix
    pressione ai 3 picchi = 3.16/5.19/3.16 (sotto soglia 8.0, zero
    anomalie, RMSE=86.6); dopo il fix, tutti e 3 i picchi rilevati,
    RMSE=0.0. Nessuna regressione: 223/223 test della suite passano,
    incluso il tasso di falsi positivi su rumore gaussiano puro (invariato)
    e tutti gli altri 6 scenari del confronto a tre vie.

    Ritorna (pulito, anomalie, pressione, soglia_effettiva): `pressione` e'
    la deviazione combinata continua per ogni punto, `soglia_effettiva' la
    soglia realmente applicata in quel punto (>= soglia_pressione, si vede
    dove/quanto la molla ha ceduto)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pressione = np.zeros(n)
    soglia_effettiva = np.full(n, soglia_pressione)

    for i in range(n):
        w_self = _window(x, i, radius)
        if w_self.size < 4:
            continue
        # Punto i ESCLUSO dalla propria finestra (come gia' faceva solo il
        # sotto-passo sigma-clipping sotto): includerlo permetteva a un
        # outlier estremo di gonfiare la propria stessa scala di
        # riferimento (mascheramento) -- verificato su impulsi isolati
        # (+/-500 su fondo 120): con la finestra self-inclusive, pressione
        # massima 4.6 (sotto qualunque soglia, zero anomalie rilevate);
        # con la finestra esclusiva, l'outlier non contamina piu' la scala
        # con cui viene giudicato.
        w = np.delete(w_self, min(i - max(0, i - radius), w_self.size - 1))
        w_riferimento = _window(x, i, radius * ref_mult)

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

        # Scala-zero-e'-evidenza-certa: se IQR o MAD collassano davvero a
        # zero (finestra locale genuinamente omogenea, >=25%/>=50% dei
        # punti identici) MA il punto i si scosta comunque dal centro, non
        # e' "nessuna informazione" da scartare -- e' la prova piu' forte
        # possibile (deviazione reale rispetto a dispersione locale nulla).
        # Il bug del "LIMITE NOTO" originale era proprio scartare questa
        # evidenza insieme allo stimatore degenere. Non tentato: una scala
        # di fallback dalla finestra di riferimento piu' ampia (provato e
        # verificato NON funzionare -- vedi sotto: allargare la finestra
        # riduce sempre la frazione di outlier, non puo' mai far rivivere
        # una statistica a breakdown frazionario). Verificato sullo
        # scenario reale "impulsi alternati" (test/test_pressure_valve_vs_
        # orca_experiment.py, +/-500 su fondo 120 senza rumore): prima di
        # questo fix, pressione ai 3 picchi = 3.16/5.19/3.16 (sotto soglia
        # 8.0, zero anomalie, RMSE=86.6); con questo fix, tutti e 3 i
        # picchi rilevati, RMSE=0.0.
        scala_locale_nulla = (iqr_scale <= 1e-12) or (scaled_mad <= 1e-12)
        if scala_locale_nulla and abs(x[i] - med) > 1e-9:
            pressione[i] = float("inf")
            continue

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

        if w_riferimento.size >= 4 and w.size >= 4:
            jsd = _jensen_shannon(w, w_riferimento)
            soglia_effettiva[i] = soglia_pressione * (1.0 + k_molla * jsd)

    out = np.copy(x).astype(float)
    anomalie = []
    for i in range(n):
        if pressione[i] > soglia_effettiva[i]:
            anomalie.append(i)
            out[i] = float(np.median(_window(x, i, radius)))
    return out, anomalie, pressione, soglia_effettiva
