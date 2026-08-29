# -*- coding: utf-8 -*-
"""
utility/arbiter.py
===================
Instrada ogni punto verso lo specialista giusto invece di forzarne uno
solo su tutto il segnale -- nato dal confronto reale in
test/test_pressure_valve_vs_orca_experiment.py: hybrid_shield (phi_ab)
vince nettamente sugli impulsi isolati (A, B, C, G) ma perde su un
cambio di regime sostenuto (F), pressure_valve fa l'esatto contrario.
Nessuno dei due batte l'altro ovunque -- la vera differenza tra i due
casi e' la DURATA della deviazione, non la sua ampiezza.

APPROCCIO: per ogni punto, si confronta con una finestra di riferimento
LARGA (raggio*ref_mult, la stessa scala usata dalla "molla" JSD di
pressure_valve) -- deliberatamente non con la finestra locale stretta,
che nel benchmark si e' vista collassare (mediana/MAD/IQR esattamente 0)
quando piu' outlier vicini finiscono nella stessa finestra piccola. Una
volta segnati i punti "devianti", la LUNGHEZZA della sequenza consecutiva
di punti devianti decide: corta (<= spike_run_max) = impulso isolato,
lunga = cambio di regime genuino.

CASO DEGENERE (baseline esattamente piatta, scala robusta = 0): non e'
un problema di stimatore -- vedi il LIMITE NOTO nel docstring di
pressure_valve, un M-estimator di Huber converge comunque a scala 0 per
lo stesso motivo strutturale. Qui non si divide mai per una scala zero:
se la dispersione della finestra di riferimento e' sotto epsilon, QUALSIASI
scostamento non nullo dalla mediana e' trattato come deviante senza
bisogno di una soglia in sigma (su una baseline davvero costante, "diverso
da costante" e' gia' la risposta, non serve altro).

LIMITE NOTO (autocorrezione, verificato dopo l'integrazione in Orca --
vedi test/test_arbiter_orca_integration.py): lo scenario F (rottura
strutturale, salto di livello permanente) NON viene rilevato come
'regime' da classify_segments -- deviazione[i] resta 0 su tutta la
sequenza attorno alla transizione. Una prima verifica aveva scambiato
questo per un successo (RMSE 0.0 con route_and_correct standalone), ma
era un artefatto: i dati grezzi di quello scenario di test coincidono
GIA' esattamente col target (la "corruzione" e' il salto stesso, non
rumore aggiunto), quindi "passa tutto grezzo" da' RMSE 0 indipendentemente
da come viene etichettato ogni punto. Causa reale della mancata
rilevazione: la finestra di riferimento LARGA e simmetrica (raggio*3),
usata proprio a cavallo della transizione, contiene meta' punti al
vecchio livello e meta' al nuovo -- la sua "scala" (MAD) si gonfia con
questa bimodalita' invece di restare piccola, diluendo la stessa
deviazione che dovrebbe far scattare la soglia. Attraverso la pipeline
reale di Orca (utility/orca.py, _applica_arbitro) questo si traduce in un
comportamento SICURO ma non migliorativo: lo scenario F ricade
sull'output gia' prodotto dallo scudo standard di Orca, identico
(9.4007) con o senza l'arbitro -- non peggiora, ma non risolve nemmeno
il problema che l'aveva motivato. Non risolto qui: servirebbe una
finestra di riferimento ASIMMETRICA (solo il lato "prima" del punto
candidato, causale) invece di una centrata, per non diluire la propria
scala con l'altro lato della transizione.
"""
from typing import Tuple

import numpy as np


def _window(x: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = len(x)
    lo, hi = max(0, i - radius), min(n, i + radius + 1)
    return x[lo:hi]


def _robust_center_scale(w: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    return med, 1.4826 * mad


def classify_segments(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2, eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna (etichette, deviazione, incertezza):
      etichette   -- array di stringhe 'clean'/'spike'/'regime', una per punto.
      deviazione  -- |x[i] - riferimento| / scala (o solo |x[i]-riferimento|
                     se la scala e' degenere), stessa unita' per ogni punto.
      incertezza  -- 0..1, quanto il punto e' vicino al confine di decisione
                     (soglia deviante/pulito, o soglia spike/regime sulla
                     lunghezza della sequenza) -- non quanto e' "corretto",
                     ma quanto la classificazione stessa e' precaria.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    deviante = np.zeros(n, dtype=bool)
    deviazione = np.zeros(n, dtype=np.float64)
    scarto_normalizzato = np.zeros(n, dtype=np.float64)  # |dev|/soglia, per l'incertezza
    med_rif = np.zeros(n, dtype=np.float64)  # mediana della finestra di riferimento, per punto

    for i in range(n):
        w_rif = _window(x, i, radius * ref_mult)
        if w_rif.size < 4:
            continue
        med, scala = _robust_center_scale(w_rif)
        med_rif[i] = med
        scarto = abs(x[i] - med)
        if scala < eps:
            # Baseline degenere: nessuna divisione, "diverso da costante"
            # e' gia' la risposta. scarto_normalizzato satura a un valore
            # alto/basso netto (mai vicino al confine per costruzione),
            # cosi' l'incertezza su questi punti resta bassa -- il caso
            # degenere e' l'UNICO dove la decisione e' davvero inequivocabile.
            deviazione[i] = scarto
            deviante[i] = scarto > eps
            scarto_normalizzato[i] = 0.0 if deviante[i] else 10.0
        else:
            deviazione[i] = scarto / scala
            deviante[i] = deviazione[i] > n_sigmas
            scarto_normalizzato[i] = deviazione[i] / n_sigmas

    etichette = np.full(n, "clean", dtype=object)
    incertezza = np.zeros(n, dtype=np.float64)

    # Incertezza sulla soglia deviante/pulito: vicino a scarto_normalizzato==1
    # (cioe' esattamente al confine n_sigmas) -> massima ambiguita'.
    incertezza_soglia = np.exp(-4.0 * (scarto_normalizzato - 1.0) ** 2)

    i = 0
    while i < n:
        if not deviante[i]:
            incertezza[i] = incertezza_soglia[i]
            i += 1
            continue
        j = i
        while j < n and deviante[j]:
            j += 1
        run_len = j - i
        if run_len <= spike_run_max:
            label = "spike"
        else:
            # Una sequenza lunga e' un vero cambio di regime solo se i suoi
            # punti si ASSESTANO su un nuovo livello coerente -- non basta
            # che siano "diversi dal vecchio", devono anche essere simili
            # TRA LORO. Verificato: 3 impulsi alternati (+500,-500,+500,
            # scenario A) sono una sequenza di lunghezza 3 (> spike_run_max
            # di default 2), ma la loro dispersione INTERNA e' enorme
            # (std~471) -- senza questo controllo venivano scambiati per un
            # regime e passati grezzi, RMSE 86.6 invece di ~0. Una sequenza
            # che si assesta davvero (scenario F) ha dispersione interna
            # piccola rispetto alla dimensione del salto stesso.
            run_vals = x[i:j]
            run_spread = float(np.std(run_vals))
            run_jump = float(abs(np.median(run_vals) - med_rif[i]))
            coerente = run_spread < 0.5 * max(run_jump, eps)

            # Coerenza interna non basta: un collasso temporaneo che poi
            # TORNA al vecchio livello (scenario C, 3 zeri poi ritorno a
            # 120) e' internamente coerente quanto un vero regime (spread
            # interno zero in entrambi i casi) -- la differenza vera si
            # vede SOLO guardando cosa succede subito dopo la sequenza:
            # un regime vero (scenario F) resta vicino al NUOVO livello,
            # un collasso torna verso quello VECCHIO. Senza questo
            # controllo scenario C restava RMSE 20.8 invece di ~0
            # (scambiato per regime, passato grezzo invece di respinto).
            post_lo, post_hi = j, min(n, j + radius)
            if post_hi > post_lo:
                post_med = float(np.median(x[post_lo:post_hi]))
                run_med = float(np.median(run_vals))
                persiste = abs(post_med - run_med) < abs(post_med - med_rif[i])
            else:
                persiste = True  # fine serie: nessun "dopo" da controllare

            label = "regime" if (coerente and persiste) else "spike"
        etichette[i:j] = label
        # Incertezza sulla soglia spike/regime: vicino a run_len==spike_run_max
        # -> massima ambiguita' sulla lunghezza; combinata (media) con quella
        # sulla soglia deviante/pulito dei singoli punti della sequenza.
        incertezza_run = float(np.exp(-1.0 * (run_len - spike_run_max) ** 2))
        incertezza[i:j] = 0.5 * incertezza_soglia[i:j] + 0.5 * incertezza_run
        i = j

    return etichette, deviazione, incertezza


def route_and_correct(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applica la classificazione e corregge ogni punto secondo l'etichetta:
      'clean'/'regime' -> passa il valore grezzo (un cambio di regime
                          genuino non va respinto, stessa filosofia del
                          gate costante di Orca);
      'spike'          -> sostituito con la mediana della finestra di
                          riferimento larga (robusta, non contaminata
                          dal singolo impulso breve).

    Ritorna (pulito, etichette, incertezza)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    etichette, _, incertezza = classify_segments(x, radius, ref_mult, n_sigmas, spike_run_max)

    pulito = np.copy(x)
    for i in range(n):
        if etichette[i] == "spike":
            w_rif = _window(x, i, radius * ref_mult)
            pulito[i] = float(np.median(w_rif))

    return pulito, etichette, incertezza
