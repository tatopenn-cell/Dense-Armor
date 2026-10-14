"""Execution pipeline: decide whether an order leaves, and how big.

Uses dense_armor.protect.orca to dampen oversized orders, and
dense_armor.utility.stats.moments to track a rolling distribution of
order sizes.

Cliente: trading desk, broker, market maker.

Note:
    Orca costa circa 20 ms a chiamata, troppo per un desk che deve
    decidere in microsecondi. La pipeline lo chiama solo sugli ordini
    che superano ``orca_pre_threshold`` volte la media: sotto quella
    threshold l'ordine passa direttamente, sopra passa da Orca. Con una
    distribuzione naturale degli ordini Orca non viene quasi mai
    invocato: resta uno scatto di sicurezza per i casi fuori scala.
    Un protettore per-ordine non e' l'uso tipico di Orca (che protegge
    un modello, non un singolo ordine); qui serve solo come controllo
    finale su ordini gia' ritenuti sospetti dalla threshold di dimensione.
"""

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.protect.orca import Orca
from dense_armor.utility.stats.moments import RunningMoments


@dataclass
class OrderResult:
    """What the execution pipeline did with one order."""

    original_size: float
    sent_size: float
    reduction: float
    approved: bool
    reason: str


class ExecutionPipeline:
    """Decide if an order leaves and how big.

    Args:
        damping_threshold: fraction of size removed by Orca above which the
            order is flagged as dampened. Default 0.30 (30%).
        max_size_multiple: if the incoming order is more than this many
            times the rolling mean, the order is refused outright.
            Default 10.0.
        orca_pre_threshold: if the order is below this many times the rolling
            mean, it passes without calling Orca (which is slow). Default
            5.0. Orca costs about 20 ms per call, so it only fires on
            orders that are unusually large. On real order rates (tens to
            hundreds per second) the average stays below 500 us.
        warmup: number of orders before the rolling mean is trusted.
            Default 20.
        seed: unused, kept for interface stability.
    """

    def __init__(
        self,
        damping_threshold: float = 0.30,
        max_size_multiple: float = 10.0,
        orca_pre_threshold: float = 5.0,
        warmup: int = 20,
        seed: int = 0,
    ):
        self.orca = Orca()
        self.moments = RunningMoments()
        self.damping_threshold = damping_threshold
        self.max_size_multiple = max_size_multiple
        self.orca_pre_threshold = orca_pre_threshold
        self.warmup = warmup
        self.n_ordini = 0
        self.n_dampened = 0
        self.n_rifiutati = 0
        self.n_passati_diretti = 0
        self.n_via_orca = 0

    def process_order(self, size: float, tipo: float = 0.0) -> OrderResult:
        """Process one order and decide its fate."""
        self.n_ordini += 1

        mu = self.moments.mean if self.moments.count >= self.warmup else float(size)

        # troppo grande: rifiuto diretto
        if mu > 0 and size > self.max_size_multiple * mu:
            self.n_rifiutati += 1
            self.moments.learn_one({"x": float(size)})
            return OrderResult(
                original_size=size,
                sent_size=0.0,
                reduction=1.0,
                approved=False,
                reason="size > max_size_multiple * mean",
            )

        # sotto threshold: passa senza Orca (che costa 20 ms)
        if mu > 0 and size <= self.orca_pre_threshold * mu:
            self.n_passati_diretti += 1
            self.moments.learn_one({"x": float(size)})
            return OrderResult(
                original_size=size,
                sent_size=size,
                reduction=0.0,
                approved=True,
                reason="passed without Orca (below orca_pre_threshold)",
            )

        # sopra threshold: passa da Orca
        self.n_via_orca += 1
        x_ord = np.array([float(size), float(tipo), 1.0])
        try:
            out = self.orca.protect_and_forward(
                ai_model_callable=None,
                x_corrupted=x_ord,
                x_reference=x_ord,
                use_input_shield=True,
                use_model_injection=False,
                use_output_shield=True,
                use_arbiter=False,
            )
            size_out = float(out[0])
            reduction = 1.0 - size_out / max(size, 1e-12)
        except (ValueError, TypeError, RuntimeError):
            size_out = size
            reduction = 0.0

        dampened = reduction > self.damping_threshold
        if dampened:
            self.n_dampened += 1
        self.moments.learn_one({"x": float(size)})

        return OrderResult(
            original_size=size,
            sent_size=size_out,
            reduction=reduction,
            approved=True,
            reason="dampened" if dampened else "passed through Orca",
        )
