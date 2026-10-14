"""Single-ticker pipeline: is today strange for this stock?

Uses dense_armor.utility.anomaly.loda and .hst on the (return, volume)
pair of a single ticker.

Cliente: analista fondamentale, retail avanzato.

Note:
    Le soglie di LODA e HST sono quelle adattive, cosi' come gli
    strumenti le calcolano. Su un ticker volatile la soglia adattiva di
    HST si stabilizza su valori bassi, e il numero di giorni segnalati
    e' alto (43 su Apple nei 5 anni). E' il comportamento corretto di un
    rilevatore online su dati grezzi. Se si vuole meno rumore, va
    filtrato l'ingresso con Hampel prima di LODA e HST: e' quello che fa
    ``FinanzaPipeline`` e il conteggio scende a 4.
"""

from dataclasses import dataclass

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA


@dataclass
class EsitoTitolo:
    """Today's verdict on one ticker."""

    giorno: int
    rendimento: float
    volume_rel: float
    loda_score: float
    hst_score: float
    anomalo: bool


class AnomaliaTitoloPipeline:
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
        self.n_giorni = 0

    def process_day(self, rendimento: float, volume_rel: float) -> EsitoTitolo:
        """Verdict on one day of the ticker."""
        x = {"ret": float(rendimento), "vol": float(volume_rel)}
        s_loda = float(self.loda.score_one(x))
        s_hst = float(self.hst.score_one(x))
        anomalo = bool(s_loda > self.loda.threshold and s_hst > self.hst.threshold)
        self.loda.learn_one(x)
        self.hst.learn_one(x)
        self.n_giorni += 1
        return EsitoTitolo(
            giorno=self.n_giorni - 1,
            rendimento=float(rendimento),
            volume_rel=float(volume_rel),
            loda_score=s_loda,
            hst_score=s_hst,
            anomalo=anomalo,
        )
