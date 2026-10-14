"""Risk pipeline: how much can we lose today, and when to warn.

Uses dense_armor.utility.stats.quantiles to keep the tail without
storing the history, and dense_armor.utility.drift to detect a slow
change in volatility.

Cliente: risk manager, fund, bank.
"""

from dataclasses import dataclass

from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.stats.quantiles import DDSketch


@dataclass
class EsitoRischio:
    """Risk snapshot at one day."""

    giorno: int
    perdita: float
    var_99: float
    var_999: float
    oltre_limite: bool
    deriva_volatilita: bool


class RischioPipeline:
    """Daily risk snapshot: VaR at 99% and 99.9%, plus a volatility alarm.

    Args:
        alpha: relative-error target for the DDSketch. Default 0.01 (1%).
        limite_perdita: absolute loss above which a day is flagged.
            Default 0.05 (5%).
        seed: random seed (unused here, kept for interface symmetry).
    """

    def __init__(
        self, alpha: float = 0.01, limite_perdita: float = 0.05, seed: int = 0
    ):
        self.sketch = DDSketch(alpha=alpha)
        self.cusum_vol = CUSUMDriftDetector(radius=20, ref_mult=3, k=0.5, h=40.0)
        self.limite_perdita = limite_perdita
        self.n_giorni = 0

    def process_day(self, perdita: float) -> EsitoRischio:
        """Process one day's loss (positive number = money lost).

        The loss is what actually happened. The VaR is what we expect from
        the tail so far.
        """
        self.n_giorni += 1

        # aggiorna la coda con la perdita di oggi
        self.sketch.learn_one({"x": float(perdita)})
        var99 = float(self.sketch.quantile(0.99))
        var999 = float(self.sketch.quantile(0.999))

        oltre = perdita > self.limite_perdita

        # CUSUM sulla perdita: se le perdite aumentano, allarme
        self.cusum_vol.update(float(perdita))
        deriva = bool(getattr(self.cusum_vol, "drift_detected", False))

        return EsitoRischio(
            giorno=self.n_giorni,
            perdita=float(perdita),
            var_99=var99,
            var_999=var999,
            oltre_limite=bool(oltre),
            deriva_volatilita=deriva,
        )
