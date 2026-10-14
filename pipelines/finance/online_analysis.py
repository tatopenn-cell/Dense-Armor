"""Online analysis: LODA and Half-Space Trees on one ticker.

Same input as :mod:`analisi_statica`, but the thresholds are
adaptive and the detectors learn one day at a time. No fixed
cutoff to choose. Two independent detectors must agree.

Client: analyst who wants to see what the adaptive tools catch
on top of the fixed-threshold baseline.

Note:
    This is not a new algorithm. It chains the same tools that
    the real-time pipelines use (LODA, Half-Space Trees) on a
    static CSV, so the two can be compared on identical input.
"""

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA


@dataclass
class OnlineResult:
    """Result of one online analysis.

    Attributes:
        n_days: total days processed.
        n_anomalous: days where both LODA and HST were above
            their own adaptive thresholds.
        anomalous_indices: indices of those days.
        returns_: the return of those days.
        final_loda_threshold: the adaptive LODA threshold at the
            end of the run.
        final_hst_threshold: the adaptive HST threshold at the
            end of the run.
    """

    n_days: int
    n_anomalous: int
    anomalous_indices: list[int]
    returns_: list[float]
    final_loda_threshold: float
    final_hst_threshold: float


class OnlineAnalysis:
    """Adaptive anomaly detection on a full return series.

    Args:
        range_ret: (min, max) of the daily return, from a
            calibration window. Half-Space Trees needs a known
            working range.
        range_vol: (min, max) of the relative volume, from the
            same calibration window.
        seed: seed for the random projections and tree splits.

    Example:
        >>> from pipelines.finance.analisi_online import OnlineAnalysis
        >>> a = OnlineAnalysis(range_ret=(-0.05, 0.05),
        ...                   range_vol=(0.5, 2.0), seed=0)
        >>> r = a.analyze([0.001, -0.04, 0.002, 0.05, -0.001],
        ...                [1.0, 1.5, 1.0, 1.2, 1.0])
        >>> isinstance(r.n_anomalous, int)
        True
    """

    def __init__(
        self,
        range_ret: tuple[float, float],
        range_vol: tuple[float, float],
        seed: int = 0,
    ):
        self.loda = LODA(
            n_projections=50,
            n_bins=10,
            window=200,
            range_init=50,
            seed=seed,
        )
        self.hst = HalfSpaceTrees(
            n_trees=15,
            depth=8,
            window=100,
            feature_ranges=[range_ret, range_vol],
            seed=seed,
        )

    def analyze(
        self,
        returns_,
        relative_volumes,
    ) -> OnlineResult:
        """Run LODA and HST on the whole series, one day at a time.

        Args:
            returns_: iterable of daily returns.
            relative_volumes: iterable of relative volumes.

        Returns:
            A :class:`OnlineResult`.
        """
        r = np.asarray(returns_, dtype=float)
        v = np.asarray(relative_volumes, dtype=float)
        n = r.size
        anomalous_indices: list[int] = []
        rend_anomali: list[float] = []
        for i in range(n):
            x = {"ret": float(r[i]), "vol": float(v[i])}
            sl = float(self.loda.score_one(x))
            sh = float(self.hst.score_one(x))
            if sl > self.loda.threshold and sh > self.hst.threshold:
                anomalous_indices.append(i)
                rend_anomali.append(float(r[i]))
            self.loda.learn_one(x)
            self.hst.learn_one(x)
        return OnlineResult(
            n_days=n,
            n_anomalous=len(anomalous_indices),
            anomalous_indices=anomalous_indices,
            returns_=rend_anomali,
            final_loda_threshold=float(self.loda.threshold),
            final_hst_threshold=float(self.hst.threshold),
        )
