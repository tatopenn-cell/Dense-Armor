"""Unsupervised fraud detection: no labels needed.

The supervised FraudPipeline needs historical labels (this
transaction was fraud, this was not). Real clients often do not
have them: a new merchant, a new country, a new payment type has
no history. This pipeline watches the transaction stream and
scores how strange each transaction is, with no labels at all.

Three Dense-Armor primitives, all unsupervised:

- **LODA** (Pevny 2016): random 1-D projections, one histogram per
  projection. Rare bins give a high score.
- **Half-Space Trees** (Tan et al. 2011): an ensemble of trees,
  score from the node mass of the leaf a point reaches.
- **Online Robust Mahalanobis** (Guillot et al. 2025): distance
  from the robust geometric median, under the median covariation
  matrix.

The three scores live on different scales, so each is turned into
its empirical quantile (percentile rank) over the recent history,
and the ensemble score is the average of the three quantiles.
"""

from collections import deque
from dataclasses import dataclass

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.anomaly.mahalanobis import (
    OnlineRobustMahalanobis,
)


@dataclass
class UnsupervisedResult:
    """Verdict on one transaction, no labels involved.

    Attributes:
        index: progressive index in the pipeline run.
        score_loda: LODA score, raw.
        score_hst: Half-Space Trees score, raw.
        score_mahalanobis: Online Robust Mahalanobis score, raw.
        quantile_loda: LODA score converted to a percentile rank
            in the recent history (0 to 1).
        quantile_hst: same, for HST.
        quantile_mahalanobis: same, for Mahalanobis.
        score_ensemble: mean of the three quantiles.
        flagged: True if the ensemble quantile is above the
            threshold.
    """

    index: int
    score_loda: float
    score_hst: float
    score_mahalanobis: float
    quantile_loda: float
    quantile_hst: float
    quantile_mahalanobis: float
    score_ensemble: float
    flagged: bool


def _quantile_of(history, value: float) -> float:
    """Empirical quantile of ``value`` in ``history`` (0..1)."""
    if not history:
        return 0.5
    n_less = sum(1 for v in history if v < value)
    return n_less / len(history)


class UnsupervisedFraudPipeline:
    """Three unsupervised detectors on a transaction stream.

    Args:
        history_size: number of recent scores kept to compute the
            quantile of the new one. Default 500.
        threshold: ensemble quantile above which a transaction is
            flagged. Default 0.99 (top 1%).
        seed: random seed for LODA and HST.
    """

    def __init__(
        self,
        history_size: int = 500,
        threshold: float = 0.99,
        seed: int = 0,
    ):
        self.loda = LODA(
            n_projections=50, n_bins=10, window=200, range_init=50, seed=seed
        )
        self.hst = HalfSpaceTrees(
            n_trees=25, depth=10, window=200, range_init=50, seed=seed
        )
        self.maha = OnlineRobustMahalanobis(n_init=100)
        self.history_size = int(history_size)
        self.threshold = float(threshold)
        self._hist_loda = deque(maxlen=history_size)
        self._hist_hst = deque(maxlen=history_size)
        self._hist_maha = deque(maxlen=history_size)
        self.n_transactions = 0

    def learn_one(self, x: dict):
        self.loda.learn_one(x)
        self.hst.learn_one(x)
        self.maha.learn_one(x)
        self.n_transactions += 1
        return self

    def score_one(self, x: dict) -> UnsupervisedResult:
        sl = float(self.loda.score_one(x))
        sh = float(self.hst.score_one(x))
        sm = float(self.maha.score_one(x))

        ql = _quantile_of(self._hist_loda, sl)
        qh = _quantile_of(self._hist_hst, sh)
        qm = _quantile_of(self._hist_maha, sm)
        se = (ql + qh + qm) / 3.0

        self._hist_loda.append(sl)
        self._hist_hst.append(sh)
        self._hist_maha.append(sm)

        return UnsupervisedResult(
            index=self.n_transactions,
            score_loda=sl,
            score_hst=sh,
            score_mahalanobis=sm,
            quantile_loda=ql,
            quantile_hst=qh,
            quantile_mahalanobis=qm,
            score_ensemble=se,
            flagged=bool(se >= self.threshold),
        )
