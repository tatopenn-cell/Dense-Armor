"""False alarms of the CUSUM on a stable stream, against the formula.

For each threshold ``h`` and each reference mode, ``TRIALS`` stable
N(0, 1) series of ``N`` samples go through ``cusum_detector``. The
stream-level false-alarm rate is the fraction of series with at least one
flag. The formula column is ``two_sided_arl(0, k, h)`` and the chance of
at least one alarm in ``N`` samples, ``1 - exp(-N / ARL)``, which takes
the mean and the noise width as known exactly; the detector estimates
them on a window of ``radius * ref_mult`` samples.

Run from the repository root:

    python benchmarks/cusum_false_alarm.py
    python benchmarks/cusum_false_alarm.py --smoke

The full run writes ``benchmarks/cusum_false_alarm.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dense_armor.utility.drift.cusum import cusum_detector, two_sided_arl

K = 0.5
N = 1000
RADIUS = 10
REF_MULT = 3
THRESHOLDS = (5.0, 10.0, 20.0)
MODES = ("adaptive", "fixed")
SEED = 42
OUT = Path(__file__).resolve().parent / "cusum_false_alarm.json"


def false_alarm_rate(h: float, mode: str, trials: int) -> float:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(trials):
        flagged, _ = cusum_detector(
            rng.normal(0.0, 1.0, N), radius=RADIUS, ref_mult=REF_MULT, k=K, h=h, reference=mode
        )
        hits += int(flagged.any())
    return hits / trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    trials = 50 if smoke else 1000
    rows = []
    print(f"{trials} stable series of {N} samples, k={K}, radius={RADIUS}, ref_mult={REF_MULT}, seed {SEED}")
    print(f"{'h':>5} {'formula ARL':>14} {'formula P(alarm)':>17} {'adaptive':>10} {'fixed':>8}")
    for h in THRESHOLDS:
        arl = two_sided_arl(0.0, K, h)
        row = {
            "h": h,
            "formula_arl": arl,
            "formula_alarm_in_n": 1.0 - math.exp(-N / arl),
            "measured": {m: false_alarm_rate(h, m, trials) for m in MODES},
        }
        rows.append(row)
        print(f"{h:>5g} {arl:>14.1f} {row['formula_alarm_in_n']:>17.3f} "
              f"{row['measured']['adaptive']:>10.3f} {row['measured']['fixed']:>8.3f}")
    if not smoke:
        payload = {"n_samples": N, "trials": trials, "k": K, "radius": RADIUS, "ref_mult": REF_MULT,
                   "seed": SEED, "rows": rows}
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
