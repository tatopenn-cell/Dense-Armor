"""Single-ticker pipeline: is today strange for this stock?

Uses dense_armor.utility.anomaly.loda and .hst on the (return, volume)
pair of a single ticker.

Cliente: analista fondamentale, retail avanzato.

Note:
    LODA and HST keep their own adaptive thresholds. On a volatile
    ticker the HST threshold stabilises at a low value, and the
    number of flagged days is high: 43 on Apple over five years.
    That is the correct behaviour of an online detector on raw
    data. To reduce the noise, filter the input with Hampel before
    LODA and HST: that is what ``FinancePipeline`` does, and the
    count drops to a handful.
"""

from dataclasses import dataclass

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA


@dataclass
class TickerResult:
    """Today's verdict on one ticker."""

    day: int
    return_: float
    relative_volume: float
    loda_score: float
    hst_score: float
    anomalous: bool


class SingleTickerPipeline:
    """Verdict on one ticker, one day at a time.

    Args:
        range_ret: (min, max) of the daily return, from calibration.
        range_vol: (min, max) of the relative volume, from calibration.
        seed: random seed for projections and splits.
    """

    def __init__(
        self,
        range_ret: tuple[float, float],
        range_vol: tuple[float, float],
        seed: int = 0,
    ):
        self.loda = LODA(
            n_projections=50, n_bins=10, window=200, range_init=50, seed=seed
        )
        self.hst = HalfSpaceTrees(
            n_trees=15,
            depth=8,
            window=100,
            feature_ranges=[range_ret, range_vol],
            seed=seed,
        )
        self.n_days = 0

    def process_day(self, return_: float, relative_volume: float) -> TickerResult:
        """Verdict on one day of the ticker."""
        x = {"ret": float(return_), "vol": float(relative_volume)}
        s_loda = float(self.loda.score_one(x))
        s_hst = float(self.hst.score_one(x))
        anomalous = bool(s_loda > self.loda.threshold and s_hst > self.hst.threshold)
        self.loda.learn_one(x)
        self.hst.learn_one(x)
        self.n_days += 1
        return TickerResult(
            day=self.n_days - 1,
            return_=float(return_),
            relative_volume=float(relative_volume),
            loda_score=s_loda,
            hst_score=s_hst,
            anomalous=anomalous,
        )
