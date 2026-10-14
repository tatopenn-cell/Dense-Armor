"""Static analysis: baseline anyone knows.

Fixed thresholds, the whole history in memory, one pass. This is
the classic method: you look at the historical return distribution
and pick a cutoff. Any return above the cutoff is "anomalous".

Client: analyst with a CSV, teacher explaining what a cutoff is,
anyone who wants a baseline before trying something more complex.

Note:
    This is deliberately basic. It uses only numpy, no Dense-Armor
    tool. It is the reference the online pipeline is compared to.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class StaticResult:
    """Result of one static analysis.

    Attributes:
        threshold: the fixed threshold used, in absolute value.
        n_days: total days analysed.
        n_anomalous: days whose return passed the threshold.
        anomalous_indices: indices of those days in the input array.
        returns_: the return of those days.
    """

    threshold: float
    n_days: int
    n_anomalous: int
    anomalous_indices: list[int]
    returns_: list[float]


class StaticAnalysis:
    """Fixed-threshold anomaly detection on a full return series.

    Args:
        threshold: absolute return above which a day is anomalous.
            Default 0.03 (3%).

    Example:
        >>> import numpy as np
        >>> from pipelines.finance.analisi_statica import StaticAnalysis
        >>> a = StaticAnalysis(threshold=0.03)
        >>> rend = np.array([0.001, -0.04, 0.002, 0.05, -0.001])
        >>> r = a.analyze(rend)
        >>> r.n_anomalous
        2
    """

    def __init__(self, threshold: float = 0.03):
        self.threshold = float(threshold)

    def analyze(self, returns_: np.ndarray) -> StaticResult:
        """Flag every return above ``threshold`` in absolute value.

        Args:
            returns_: array of daily returns.

        Returns:
            A :class:`StaticResult` with indices and returns.
        """
        r = np.asarray(returns_, dtype=float)
        maschera = np.abs(r) > self.threshold
        indici = np.where(maschera)[0].tolist()
        valori = r[maschera].tolist()
        return StaticResult(
            threshold=self.threshold,
            n_days=int(r.size),
            n_anomalous=int(maschera.sum()),
            anomalous_indices=indici,
            returns_=valori,
        )
