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
class RiskResult:
    """Risk snapshot at one day."""

    day: int
    loss: float
    var_99: float
    var_999: float
    over_limit: bool
    volatility_drift: bool


class RiskPipeline:
    """Daily risk snapshot: VaR at 99% and 99.9%, plus a volatility alarm.

    Args:
        alpha: relative-error target for the DDSketch. Default 0.01 (1%).
        loss_limit: absolute loss above which a day is flagged.
            Default 0.05 (5%).
        seed: random seed (unused here, kept for interface symmetry).
    """

    def __init__(
        self, alpha: float = 0.01, loss_limit: float = 0.05, seed: int = 0
    ):
        self.sketch = DDSketch(alpha=alpha)
        self.cusum_vol = CUSUMDriftDetector(radius=20, ref_mult=3, k=0.5, h=40.0)
        self.loss_limit = loss_limit
        self.n_days = 0

    def process_day(self, loss: float) -> RiskResult:
        """Process one day's loss (positive number = money lost).

        The loss is what actually happened. The VaR is what we expect from
        the tail so far.
        """
        self.n_days += 1

        # aggiorna la coda con la loss di oggi
        self.sketch.learn_one({"x": float(loss)})
        var99 = float(self.sketch.quantile(0.99))
        var999 = float(self.sketch.quantile(0.999))

        oltre = loss > self.loss_limit

        # CUSUM sulla loss: se le perdite aumentano, allarme
        self.cusum_vol.update(float(loss))
        deriva = bool(getattr(self.cusum_vol, "drift_detected", False))

        return RiskResult(
            day=self.n_days,
            loss=float(loss),
            var_99=var99,
            var_999=var999,
            over_limit=bool(oltre),
            volatility_drift=deriva,
        )
