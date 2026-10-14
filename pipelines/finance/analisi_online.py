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
class RisultatoOnline:
    """Result of one online analysis.

    Attributes:
        n_giorni: total days processed.
        n_anomali: days where both LODA and HST were above
            their own adaptive thresholds.
        indici_anomali: indices of those days.
        rendimenti: the return of those days.
        soglia_loda_finale: the adaptive LODA threshold at the
            end of the run.
        soglia_hst_finale: the adaptive HST threshold at the
            end of the run.
    """

    n_giorni: int
    n_anomali: int
    indici_anomali: list[int]
    rendimenti: list[float]
    soglia_loda_finale: float
    soglia_hst_finale: float


class AnalisiOnline:
    """Adaptive anomaly detection on a full return series.

    Args:
        range_ret: (min, max) of the daily return, from a
            calibration window. Half-Space Trees needs a known
            working range.
        range_vol: (min, max) of the relative volume, from the
            same calibration window.
        seed: seed for the random projections and tree splits.

    Example:
        >>> from pipelines.finance.analisi_online import AnalisiOnline
        >>> a = AnalisiOnline(range_ret=(-0.05, 0.05),
        ...                   range_vol=(0.5, 2.0), seed=0)
        >>> r = a.analizza([0.001, -0.04, 0.002, 0.05, -0.001],
        ...                [1.0, 1.5, 1.0, 1.2, 1.0])
        >>> isinstance(r.n_anomali, int)
        True
    """

    def __init__(
        self,
        range_ret: tuple[float, float],
        range_vol: tuple[float, float],
        seed: int = 0,
    ):
        self.loda = LODA(
            n_projections=50, n_bins=10, window=200,
            range_init=50, seed=seed,
        )
        self.hst = HalfSpaceTrees(
            n_trees=15, depth=8, window=100,
            feature_ranges=[range_ret, range_vol], seed=seed,
        )

    def analizza(
        self,
        rendimenti, volumi_rel,
    ) -> RisultatoOnline:
        """Run LODA and HST on the whole series, one day at a time.

        Args:
            rendimenti: iterable of daily returns.
            volumi_rel: iterable of relative volumes.

        Returns:
            A :class:`RisultatoOnline`.
        """
        r = np.asarray(rendimenti, dtype=float)
        v = np.asarray(volumi_rel, dtype=float)
        n = r.size
        indici_anomali: list[int] = []
        rend_anomali: list[float] = []
        for i in range(n):
            x = {"ret": float(r[i]), "vol": float(v[i])}
            sl = float(self.loda.score_one(x))
            sh = float(self.hst.score_one(x))
            if sl > self.loda.threshold and sh > self.hst.threshold:
                indici_anomali.append(i)
                rend_anomali.append(float(r[i]))
            self.loda.learn_one(x)
            self.hst.learn_one(x)
        return RisultatoOnline(
            n_giorni=n,
            n_anomali=len(indici_anomali),
            indici_anomali=indici_anomali,
            rendimenti=rend_anomali,
            soglia_loda_finale=float(self.loda.threshold),
            soglia_hst_finale=float(self.hst.threshold),
        )
