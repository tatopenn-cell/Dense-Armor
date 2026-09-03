# -*- coding: utf-8 -*-
"""
utility/streaming.py
=====================
Promosso da Dense-Evolution-Discovery (Esperimenti 48-49) dopo
validazione su due domini fisici reali indipendenti (braccio robotico
teleoperato SO-101, IMU umano reale UCI HAR) -- stessa disciplina di
promozione gia' usata per `stable_frame_filter.py`.

MOTIVAZIONE: un robot reale gira a 30-100Hz in tempo reale, non puo'
aspettare un array gia' registrato -- ma leggendo `arbiter.classify_
segments` riga per riga (non per assunzione) si trova un vincolo reale:
l'etichetta finale spike/regime guarda `radius` punti AVANTI alla fine
di una sequenza deviante (righe 178-186 di arbiter.py, il controllo
"persiste") per decidere se la sequenza si assesta o torna indietro --
quella meta' non puo' essere streaming a latenza zero. Un vero anello
di sicurezza robotico non ha comunque bisogno di quella distinzione in
tempo reale (e' una domanda di triage, risolvibile dopo); ha bisogno
di "questo punto e' deviante ADESSO" -- esattamente il calcolo per-
punto di classify_segments PRIMA della logica di lunghezza-sequenza.
Solo questa meta' viene portata qui.

CORRETTEZZA: `StreamingDeviationDetector`, alimentato un punto alla
volta, deve riprodurre esattamente (bit-per-bit) l'array `deviante`
interno di `classify_segments` -- verificato su dati reali (non solo
sintetici) in Dense-Evolution-Discovery prima di questa promozione,
ri-verificato qui in test/test_streaming.py sulla telemetria reale
gia' congelata in questo repo (test/agent_v2/telemetry_v2_frozen.jsonl).

PRESTAZIONI: buffer semplice (deque) che ricalcola mediana/MAD ogni
passo -- O(span), non una struttura a due heap O(log span). Per le
finestre gia' usate in tutto questo progetto (10-100 punti), misurato
a ~18.6kHz sostenibile su hardware reale, oltre 180x il tasso reale di
un anello di controllo robotico (30-100Hz) -- nessun compromesso di
prestazioni reale da giustificare con maggiore complessita'.

SUPPORTO MULTI-CANALE: ogni esperimento robotico reale in Dense-
Evolution-Discovery (le 6 giunture del braccio LeRobot, i piu' assi
dell'IMU) ha richiesto un ciclo manuale per-canale attorno a funzioni
pensate per un solo segnale 1D. `classify_segments_multichannel` e
`MultiChannelStreamingDeviationDetector` applicano la STESSA logica
per-canale gia' validata, senza richiedere quel ciclo -- ergonomia, non
un nuovo algoritmo. Ogni canale mantiene la propria finestra di
riferimento e la propria scala indipendenti per costruzione (un giunto
veloce e uno quasi fermo hanno scale di rumore naturali molto diverse;
una scala condivisa desensibilizzerebbe quello lento o darebbe falsi
allarmi su quello attivo).
"""
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .arbiter import _robust_center_scale, classify_segments


class StreamingDeviationDetector:
    """Flag di deviazione causale a latenza zero, un punto alla volta --
    stessa finestra causale (radius*ref_mult), stessa mediana/MAD
    robusta, stessa regola di baseline degenere di classify_segments'
    (arbiter.py), NON la distinzione spike/regime (che richiede uno
    sguardo in avanti, resta una domanda batch/offline per design, non
    una svista).

    Examples
    --------
    >>> det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> stream = list(rng.normal(0, 1, 20)) + [50.0]  # un grande outlier alla fine
    >>> flags = [det.update(x) for x in stream]
    >>> flags[-1]
    True
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3, n_sigmas: float = 3.0, eps: float = 1e-9):
        self.radius = radius
        self.ref_mult = ref_mult
        self.n_sigmas = n_sigmas
        self.eps = eps
        self._span = radius * ref_mult
        self._buffer = deque(maxlen=self._span)
        self.last_deviation = 0.0  # informativo, non parte del contratto di correttezza

    def update(self, x: float) -> bool:
        """Alimenta un nuovo punto. Ritorna True se DEVIA rispetto alla
        finestra dei punti strettamente PRECEDENTI (mai includendo se
        stesso) -- stessa semantica di `_window_causal` in arbiter.py."""
        if len(self._buffer) < 4:
            self._buffer.append(x)
            self.last_deviation = 0.0
            return False

        med, scala = _robust_center_scale(np.array(self._buffer))
        scarto = abs(x - med)
        if scala < self.eps:
            self.last_deviation = scarto
            deviante = scarto > self.eps
        else:
            self.last_deviation = scarto / scala
            deviante = self.last_deviation > self.n_sigmas

        self._buffer.append(x)
        return deviante


class MultiChannelStreamingDeviationDetector:
    """N istanze indipendenti di `StreamingDeviationDetector`, una per
    canale -- il flag di ogni canale e' calcolato esattamente come se
    `StreamingDeviationDetector` girasse su quel canale da solo (i
    canali non si influenzano mai a vicenda)."""

    def __init__(self, n_channels: int, radius: int = 10, ref_mult: int = 3,
                 n_sigmas: float = 3.0, eps: float = 1e-9):
        self.n_channels = n_channels
        self._detectors: List[StreamingDeviationDetector] = [
            StreamingDeviationDetector(radius=radius, ref_mult=ref_mult, n_sigmas=n_sigmas, eps=eps)
            for _ in range(n_channels)
        ]

    def update(self, x_vec) -> np.ndarray:
        """x_vec: lunghezza n_channels. Ritorna array bool (n_channels,)."""
        x_vec = np.asarray(x_vec, dtype=np.float64).ravel()
        if x_vec.size != self.n_channels:
            raise ValueError(f"attesi {self.n_channels} canali, ricevuti {x_vec.size}")
        return np.array([det.update(float(v)) for det, v in zip(self._detectors, x_vec)], dtype=bool)


def classify_segments_multichannel(
    X: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2, eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n_campioni, n_canali). Applica `classify_segments` (arbiter.py)
    indipendentemente a ogni colonna -- stesso identico risultato per
    canale che chiamarla singolarmente, solo impilato invece di
    richiedere un ciclo manuale al chiamante.

    Ritorna (etichette, deviazione, incertezza), ognuno (n_campioni,
    n_canali)."""
    X = np.asarray(X, dtype=np.float64)
    n, c = X.shape
    etichette = np.empty((n, c), dtype=object)
    deviazione = np.zeros((n, c), dtype=np.float64)
    incertezza = np.zeros((n, c), dtype=np.float64)
    for j in range(c):
        e, d, u = classify_segments(X[:, j], radius=radius, ref_mult=ref_mult,
                                     n_sigmas=n_sigmas, spike_run_max=spike_run_max, eps=eps)
        etichette[:, j] = e
        deviazione[:, j] = d
        incertezza[:, j] = u
    return etichette, deviazione, incertezza
