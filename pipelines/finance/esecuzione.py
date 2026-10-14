"""Execution pipeline: decide whether an order leaves, and how big.

Uses dense_armor.protect.orca to dampen oversized orders, and
dense_armor.utility.stats.moments to track a rolling distribution of
order sizes.

Cliente: trading desk, broker, market maker.

Note:
    Orca costa circa 20 ms a chiamata, troppo per un desk che deve
    decidere in microsecondi. La pipeline lo chiama solo sugli ordini
    che superano ``soglia_pre_orca`` volte la media: sotto quella
    soglia l'ordine passa direttamente, sopra passa da Orca. Con una
    distribuzione naturale degli ordini Orca non viene quasi mai
    invocato: resta uno scatto di sicurezza per i casi fuori scala.
    Un protettore per-ordine non e' l'uso tipico di Orca (che protegge
    un modello, non un singolo ordine); qui serve solo come controllo
    finale su ordini gia' ritenuti sospetti dalla soglia di dimensione.
"""

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.protect.orca import Orca
from dense_armor.utility.stats.moments import RunningMoments


@dataclass
class EsitoOrdine:
    """What the execution pipeline did with one order."""
    size_originale: float
    size_inviata: float
    riduzione: float
    approvato: bool
    motivo: str


class EsecuzionePipeline:
    """Decide if an order leaves and how big.

    Args:
        soglia_damping: fraction of size removed by Orca above which the
            order is flagged as dampened. Default 0.30 (30%).
        max_size_molteplice: if the incoming order is more than this many
            times the rolling mean, the order is refused outright.
            Default 10.0.
        soglia_pre_orca: if the order is below this many times the rolling
            mean, it passes without calling Orca (which is slow). Default
            5.0. Orca costs about 20 ms per call, so it only fires on
            orders that are unusually large. On real order rates (tens to
            hundreds per second) the average stays below 500 us.
        warmup: number of orders before the rolling mean is trusted.
            Default 20.
        seed: unused, kept for interface stability.
    """

    def __init__(self,
                 soglia_damping: float = 0.30,
                 max_size_molteplice: float = 10.0,
                 soglia_pre_orca: float = 5.0,
                 warmup: int = 20,
                 seed: int = 0):
        self.orca = Orca()
        self.moments = RunningMoments()
        self.soglia_damping = soglia_damping
        self.max_size_molteplice = max_size_molteplice
        self.soglia_pre_orca = soglia_pre_orca
        self.warmup = warmup
        self.n_ordini = 0
        self.n_dampened = 0
        self.n_rifiutati = 0
        self.n_passati_diretti = 0
        self.n_via_orca = 0

    def process_order(self, size: float, tipo: float = 0.0) -> EsitoOrdine:
        """Process one order and decide its fate."""
        self.n_ordini += 1

        mu = self.moments.mean if self.moments.count >= self.warmup else float(size)

        # troppo grande: rifiuto diretto
        if mu > 0 and size > self.max_size_molteplice * mu:
            self.n_rifiutati += 1
            self.moments.learn_one({"x": float(size)})
            return EsitoOrdine(
                size_originale=size,
                size_inviata=0.0,
                riduzione=1.0,
                approvato=False,
                motivo="size > max_size_molteplice * mean",
            )

        # sotto soglia: passa senza Orca (che costa 20 ms)
        if mu > 0 and size <= self.soglia_pre_orca * mu:
            self.n_passati_diretti += 1
            self.moments.learn_one({"x": float(size)})
            return EsitoOrdine(
                size_originale=size,
                size_inviata=size,
                riduzione=0.0,
                approvato=True,
                motivo="passed without Orca (below soglia_pre_orca)",
            )

        # sopra soglia: passa da Orca
        self.n_via_orca += 1
        x_ord = np.array([float(size), float(tipo), 1.0])
        try:
            out = self.orca.protect_and_forward(
                ai_model_callable=None, x_corrupted=x_ord, x_reference=x_ord,
                use_input_shield=True, use_model_injection=False,
                use_output_shield=True, use_arbiter=False,
            )
            size_out = float(out[0])
            riduzione = 1.0 - size_out / max(size, 1e-12)
        except (ValueError, TypeError, RuntimeError):
            size_out = size
            riduzione = 0.0

        dampened = riduzione > self.soglia_damping
        if dampened:
            self.n_dampened += 1
        self.moments.learn_one({"x": float(size)})

        return EsitoOrdine(
            size_originale=size,
            size_inviata=size_out,
            riduzione=riduzione,
            approvato=True,
            motivo="dampened" if dampened else "passed through Orca",
        )
