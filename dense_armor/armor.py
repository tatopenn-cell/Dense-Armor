"""
armor.py — Armor: the wearable instance of the shield for a 1D series.
Uses only the modules in core/ and utility/ of this package.

Quick use (CLI):
    python -m dense_armor 1.2 1.3 9999 1.25 nan 1.3
    python -m dense_armor file_with_numbers.txt

Use from code:
    from dense_armor import Armor
    a = Armor(livello_ia=0.0)        # 0 = newborn AI (FILTERS) ... 1 = mature (only MARKS)
    pulito, K, anomalie = a.analizza(serie)
    pulito, K, anomalie = a.analizza(today_series, riferimento=yesterday_baseline)  # anti-drift
"""

import os
import sys

import numpy as np

from .core.signals.preprocessing.metro import Metro  # anti-underflow scaler, unchanged
from .utility.hybrid_engine import (  # phi_ab/trigger engine, see CHANGELOG 1.1.0
    _local_nan_fill,
    hybrid_shield,
)


class Armor:
    """Wearable shield with a universal DYNAMIC CLIP.

    livello_ia (0..1): how mature the AI that wears it is.
      0.0 -> full clip: the shield FILTERS the data (armor for a newborn AI)
      1.0 -> zero clip: the shield does NOT touch the data, it only MARKS (a lens for a mature AI)
    FLOOR: the marking of anomalies is NEVER clipped — at no level does the
    shield fall silent (the anomaly may BE the answer: never erase it
    silently). The level can be measured, not declared:
    Armor.livello_da_entropia(entropia, vocab_size).

    MEMORY: the shield is a PURE function, with no state (light by design).
    The memory belongs to the AI that wears it: to see SLOW DRIFT
    pass your historical baseline as `riferimento`.
    """

    def __init__(
        self,
        static_threshold: float = 0.15,
        initial_damping: float = 0.85,
        alpha: float = 0.05,
        soglia_anomalia: float | None = None,
        livello_ia: float = 1.0,
    ) -> None:
        """static_threshold/initial_damping/alpha — kept for signature compatibility
        with earlier versions, not used by the hybrid engine (which has no
        state or calibration: it is a pure function, see CHANGELOG 1.1.0).
        soglia_anomalia — optional fixed threshold (None = computed from the data);
        livello_ia — 0=newborn (filters) .. 1=mature (only marks)."""
        self.soglia = soglia_anomalia
        self.livello_ia = float(min(1.0, max(0.0, livello_ia)))

    def set_livello(self, livello: float) -> None:
        """Update the level of the AI (0=weak -> filters; 1=mature -> marks)."""
        self.livello_ia = float(min(1.0, max(0.0, livello)))

    @staticmethod
    def livello_da_entropia(entropia: float, vocab_size: int) -> float:
        """MEASURED level: 1 - entropy/log(V) (same formula as the dynamic vent)."""
        import math

        return max(
            0.0, min(1.0, 1.0 - entropia / (math.log(max(2, vocab_size)) + 1e-12))
        )

    def analizza(
        self, serie: np.ndarray, riferimento: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """serie: 1D (list/array). For N-D tensors pass tensor.ravel()
        and reshape afterwards (the shield is agnostic: the transposition is enough).
        Returns (pulito, K, anomaly indices)."""
        grezza = np.asarray(serie, dtype=np.float64).ravel()

        pulito_pieno, trigger, _meta = hybrid_shield(grezza, riferimento)

        # PERCEPTION at full K (floor: never clipped) — binary by design:
        # the underlying trigger is strictly 0.0/1.0, not a continuous sigmoid.
        K = 1.0 - trigger

        # raw series sanitized (only NaN/Inf -> local median), the same
        # sanitization used internally by hybrid_shield (see
        # _local_nan_fill): it is the real "does not touch the data" at zero
        # clip, not a signal already pre-filtered by an upstream stage. LOCAL
        # median, not the mean of the whole series: otherwise a spike elsewhere
        # in the series distorts the substitute of every NaN, not only the
        # nearby ones (see CHANGELOG 1.1.1).
        radius_fill = max(
            3 if len(grezza) < 3 else min(20, max(3, len(grezza) // 3)), 3
        )
        grezza_pulita = _local_nan_fill(
            np.where(np.isinf(grezza), np.nan, grezza), radius_fill
        )

        # INTERVENTION clipped by the level of the AI
        clip = 1.0 - self.livello_ia
        pulito = grezza_pulita - (K * clip) * (grezza_pulita - pulito_pieno)

        dev = np.abs(grezza_pulita - pulito_pieno)
        soglia = (
            self.soglia if self.soglia is not None else (dev.mean() + 2.0 * dev.std())
        )
        anomalie = {int(i) for i in np.where(dev > max(soglia, 1e-12))[0]}

        # robust detector (median/MAD), independent of the hybrid engine —
        # it also covers the first 2 points of the series, which the trigger
        # loop never evaluates (see known limit in hybrid_engine.hybrid_shield)
        finiti = grezza[np.isfinite(grezza)]
        if finiti.size >= 4:
            med = float(np.median(finiti))
            mad = float(np.median(np.abs(finiti - med))) or 1e-12
            z_rob = np.abs((np.nan_to_num(grezza, nan=med) - med) / (1.4826 * mad))
            anomalie |= {int(i) for i in np.where(z_rob > 6.0)[0]}

        anomalie |= {int(i) for i in np.where(~np.isfinite(grezza))[0]}
        return pulito, K, sorted(anomalie)

    def deriva(self, serie: np.ndarray) -> tuple[float, bool, float]:
        """SLOW DRIFT DETECTOR (uses METRO, as suggested by Salvatore).

        The boiled frog: infinitesimal increments (e.g. +0.004/step) invisible
        to the point-by-point shield. Here we look at the TREND of the
        increments: if their mean is systematic with respect to the noise,
        there is drift — and METRO (logarithmic scaler) resolves its scale
        even when it is below the visible precision. It works WITHOUT a
        historical baseline (but the baseline via `riferimento` in analizza()
        remains the stronger method).

        Returns (rate_per_step, significant: bool, metro_exponent).
        """
        v = np.asarray(serie, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size < 9:
            return 0.0, False, 0.0
        # the signature of drift is the CUMULATIVE SHIFT: median of the first
        # third vs median of the last third (robust to spikes), compared with
        # the noise of the series. The single increments stay in the noise —
        # it is the sum that betrays the frog.
        terzo = v.size // 3
        med_inizio = float(np.median(v[:terzo]))
        med_fine = float(np.median(v[-terzo:]))
        spostamento = med_fine - med_inizio
        tasso = spostamento / max(1, v.size - terzo)
        mad_serie = float(np.median(np.abs(v - np.median(v)))) or 1e-12
        rumore_mediane = 1.4826 * mad_serie / np.sqrt(terzo)
        significativa = abs(spostamento) > 3.0 * rumore_mediane
        # METRO: resolves the scale of the infinitesimal (readable even at 1e-12)
        _v_new, fact = Metro().enc(abs(tasso))
        esponente = (
            float(np.log10(fact)) - (-4.0) if fact > 0 else 0.0
        )  # = -log10(|tasso|)
        return tasso, bool(significativa), esponente

    def referto_json(
        self,
        serie: np.ndarray,
        nome: str = "serie",
        riferimento: np.ndarray | None = None,
    ) -> dict:
        """MACHINE-READABLE report (dict ready for json.dumps).
        Meant for AIs and other programs: no prose, only data.
        Non-finite values become strings ("NaN") to give valid JSON."""
        pulito, K, anomalie = self.analizza(serie, riferimento)
        s = np.asarray(serie, dtype=np.float64).ravel()
        tasso, sig, esp = self.deriva(serie)

        def _num(v: float) -> float | str:
            """Convert to float, or to "NaN" (a string, to stay JSON-valid) if not finite."""
            return float(v) if np.isfinite(v) else "NaN"

        return {
            "nome": nome,
            "punti": int(s.size),
            "livello_ia": self.livello_ia,
            "anomalie": [
                {
                    "indice": int(i),
                    "grezzo": _num(s[i]),
                    "pulito": _num(pulito[i]),
                    "K": _num(K[i]),
                }
                for i in anomalie
            ],
            "K_medio": _num(np.mean(K)),
            "deriva": {
                "tasso_per_passo": _num(tasso),
                "significativa": bool(sig),
                "esponente_metro": _num(esp),
            },
        }

    def referto(
        self,
        serie: np.ndarray,
        nome: str = "serie",
        riferimento: np.ndarray | None = None,
    ) -> list[int]:
        """Analysis + readable report."""
        pulito, K, anomalie = self.analizza(serie, riferimento)
        s = np.asarray(serie, dtype=np.float64).ravel()
        print(
            f"[ARMATURA] {nome}: {len(s)} punti | livello IA {self.livello_ia:.2f} | anomalie: {len(anomalie)}"
        )
        for i in anomalie[:10]:
            print(
                f"   -> indice {i}: valore {s[i]!r} (pulito: {pulito[i]:.4f}, K={K[i]:.3f})"
            )
        if len(anomalie) > 10:
            print(f"   ... e altre {len(anomalie) - 10}")
        tasso, sig, esp = self.deriva(serie)
        if sig:
            print(
                f"   ⚠ DERIVA LENTA rilevata (Metro): {tasso:+.2e} per passo (scala 10^-{esp:.1f})"
            )
        return anomalie


def main() -> None:
    """CLI entry point: `python -m dense_armor [--json] <file_or_numbers...>`."""
    import re as _re

    argv = sys.argv[1:]
    come_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if not argv:
        print(
            "USO: python -m dense_armor [--json] <file_con_numeri>  oppure  python -m dense_armor [--json] n1 n2 n3 ..."
        )
        sys.exit(0)
    if len(argv) == 1 and os.path.exists(argv[0]):
        with open(argv[0], encoding="utf-8", errors="ignore") as fh:
            testo = fh.read()
        numeri = [
            float(x) if x.lower() != "nan" else float("nan")
            for x in _re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|NaN)", testo)
        ]
        nome = os.path.basename(argv[0])
    else:
        numeri = [float(x) if x.lower() != "nan" else float("nan") for x in argv]
        nome = "serie da riga di comando"
    serie = np.asarray(numeri, dtype=np.float64)
    if come_json:
        import json as _json

        print(_json.dumps(Armor().referto_json(serie, nome), ensure_ascii=False))
    else:
        Armor().referto(serie, nome)


if __name__ == "__main__":
    main()
