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
class RisultatoStatico:
    """Result of one static analysis.

    Attributes:
        soglia: the fixed threshold used, in absolute value.
        n_giorni: total days analysed.
        n_anomali: days whose return passed the threshold.
        indici_anomali: indices of those days in the input array.
        rendimenti: the return of those days.
    """

    soglia: float
    n_giorni: int
    n_anomali: int
    indici_anomali: list[int]
    rendimenti: list[float]


class AnalisiStatica:
    """Fixed-threshold anomaly detection on a full return series.

    Args:
        soglia: absolute return above which a day is anomalous.
            Default 0.03 (3%).

    Example:
        >>> import numpy as np
        >>> from pipelines.finance.analisi_statica import AnalisiStatica
        >>> a = AnalisiStatica(soglia=0.03)
        >>> rend = np.array([0.001, -0.04, 0.002, 0.05, -0.001])
        >>> r = a.analizza(rend)
        >>> r.n_anomali
        2
    """

    def __init__(self, soglia: float = 0.03):
        self.soglia = float(soglia)

    def analizza(self, rendimenti: np.ndarray) -> RisultatoStatico:
        """Flag every return above ``soglia`` in absolute value.

        Args:
            rendimenti: array of daily returns.

        Returns:
            A :class:`RisultatoStatico` with indices and returns.
        """
        r = np.asarray(rendimenti, dtype=float)
        maschera = np.abs(r) > self.soglia
        indici = np.where(maschera)[0].tolist()
        valori = r[maschera].tolist()
        return RisultatoStatico(
            soglia=self.soglia,
            n_giorni=int(r.size),
            n_anomali=int(maschera.sum()),
            indici_anomali=indici,
            rendimenti=valori,
        )
