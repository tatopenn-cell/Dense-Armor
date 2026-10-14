"""Five Dense-Armor tools in sequence, on a stock-market tick stream."""

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.anomaly.filters import HampelScorer
from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.protect.orca import Orca


@dataclass
class TickEsito:
    """What every stage did on a single tick."""

    tick: int
    prezzo: float
    volume: float
    prezzo_pulito: float
    scartato_da_armatura: bool
    deriva_segnalata: bool
    loda_score: float
    hst_score: float
    anomalia: bool
    orca_riduzione: float
    orca_smorzato: bool


class FinanzaPipeline:
    """Five Dense-Armor stages on a stream of (price, relative volume).

    Args:
        range_ret: (min, max) of the daily return, from a calibration window.
        range_vol: (min, max) of the relative volume (volume / mean volume).
        soglia_armatura: Hampel score above which the tick is discarded.
            Default 5.0 (about five robust sigmas).
        soglia_orca: norm reduction above which an order counts as dampened.
            Default 0.50 (half of the norm removed).
        seed: random seed for the projections and the tree splits.
    """

    def __init__(
        self,
        range_ret: tuple[float, float],
        range_vol: tuple[float, float],
        soglia_armatura: float = 5.0,
        soglia_orca: float = 0.50,
        seed: int = 0,
    ):
        self.hampel = HampelScorer(radius=20, n_sigmas=soglia_armatura)
        self.cusum = CUSUMDriftDetector(radius=20, ref_mult=3, k=0.5, h=30.0)
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
        self.orca = Orca()
        self.soglia_orca = soglia_orca
        self.storico: list[float] = []
        self.n_tick = 0

    def process_tick(self, prezzo: float, volume_rel: float) -> TickEsito:
        """Process one tick and return what every stage did."""
        sc_h = self.hampel.score_one({"x": float(prezzo)})
        scartato = sc_h > 5.0
        prezzo_pulito = float("nan") if scartato else float(prezzo)
        self.hampel.learn_one({"x": float(prezzo)})

        if not scartato:
            self.storico.append(prezzo_pulito)
        self.n_tick += 1

        deriva = False
        if not scartato:
            self.cusum.update(prezzo_pulito)
            deriva = bool(getattr(self.cusum, "drift_detected", False))

        if len(self.storico) > 2 and not scartato:
            ret = (prezzo_pulito - self.storico[-2]) / self.storico[-2]
        else:
            ret = 0.0
        x = {"a": float(ret), "b": float(volume_rel)}
        s_loda = self.loda.score_one(x)
        s_hst = self.hst.score_one(x)
        anomalia = bool(s_loda > self.loda.threshold and s_hst > self.hst.threshold)
        self.loda.learn_one(x)
        self.hst.learn_one(x)

        x_ord = np.array([float(volume_rel), float(ret), 1.0])
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
            riduzione = 1.0 - float(
                np.linalg.norm(out) / max(np.linalg.norm(x_ord), 1e-12)
            )
        except (ValueError, TypeError, RuntimeError):
            riduzione = 0.0
        smorzato = riduzione > self.soglia_orca

        return TickEsito(
            tick=self.n_tick - 1,
            prezzo=float(prezzo),
            volume=float(volume_rel),
            prezzo_pulito=prezzo_pulito,
            scartato_da_armatura=scartato,
            deriva_segnalata=deriva,
            loda_score=float(s_loda),
            hst_score=float(s_hst),
            anomalia=anomalia,
            orca_riduzione=float(riduzione),
            orca_smorzato=bool(smorzato),
        )

    @staticmethod
    def calibra(
        prezzi: np.ndarray, volumi_rel: np.ndarray, frazione: float = 0.2
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (range_ret, range_vol) from the first ``frazione`` days.

        Args:
            prezzi: closing-price series.
            volumi_rel: relative-volume series (volume / mean volume).
            frazione: fraction of the series used to calibrate.
        """
        n_cal = max(20, int(len(prezzi) * frazione))
        ret = np.diff(prezzi[:n_cal]) / prezzi[:n_cal][:-1]
        vol = volumi_rel[:n_cal]
        range_ret = (float(np.percentile(ret, 0.5)), float(np.percentile(ret, 99.5)))
        range_vol = (float(np.percentile(vol, 0.5)), float(np.percentile(vol, 99.5)))
        return range_ret, range_vol
