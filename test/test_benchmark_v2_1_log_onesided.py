# -*- coding: utf-8 -*-
"""
test/test_benchmark_v2_1_log_onesided.py
===========================================
Tests a real fix for Benchmark v2's own most important finding: baseline
false-positive rate on REAL agent latency (dense_armor 10.0%, cusum
12.5%) was much higher than on synthetic gaussian noise (~1.3-1.4%).
Root cause hypothesis (declared here, before looking at this file's own
results): latency is a positive, right-skewed quantity with a heavy
tail (occasional slow calls), not gaussian noise -- two standard,
well-established fixes for exactly that kind of signal:

  1. LOG-TRANSFORM: analyze log(latency_s) instead of raw latency_s.
     Standard practice for positive/skewed metrics (also applies to
     token counts, response sizes, etc.) -- symmetrizes the
     distribution so a robust median/MAD-based z-score isn't dominated
     by the right tail.
  2. ONE-SIDED (upper-tail only): for latency specifically, only an
     INCREASE matters (a slow/corrupted call) -- a fast response is
     never itself evidence of an anomaly. Implemented here as a
     post-filter on top of classify_segments/cusum_detector's own
     generic two-sided output (a flag is kept only if x[i] is above a
     causal reference median), NOT a change to those library functions
     themselves -- they stay general-purpose two-sided detectors for
     signals where both directions matter.

METHODOLOGICAL DISCIPLINE: this is evaluated on telemetry_v2_1_fresh.
jsonl, a SEPARATE, NEW real Qwen run (test/agent_v2/run_benchmark_v2.py
telemetry_v2_1_fresh.jsonl) -- NOT a re-evaluation of the original
telemetry_v2_frozen.jsonl that motivated this fix. Retuning and
re-testing on the same frozen data that revealed a problem is exactly
the post-hoc p-hacking this project's own benchmarks have deliberately
avoided throughout; the fix has to prove itself on data it never saw.

This file reports RAW (two-sided, linear latency) vs FIXED
(log-transformed, one-sided) side by side, honestly -- if the fix does
not actually help on this new real data, that is what gets reported.
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

_DATA_PATH = pathlib.Path(__file__).resolve().parent / "agent_v2" / "telemetry_v2_1_fresh.jsonl"

ARBITER_KW = dict(radius=5, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=5, ref_mult=2, k=0.5, h=5.0)
CAUSAL_SPAN = ARBITER_KW["radius"] * ARBITER_KW["ref_mult"]


def _load_records():
    records = []
    with open(_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    by_scenario = defaultdict(list)
    for rec in records:
        by_scenario[rec["scenario"]].append(rec)
    for scen in by_scenario:
        by_scenario[scen].sort(key=lambda r: r["step_id"])
    return by_scenario


def _causal_median(x: np.ndarray, span: int) -> np.ndarray:
    n = len(x)
    meds = np.zeros(n)
    for i in range(n):
        lo = max(0, i - span)
        w = x[lo:i]
        meds[i] = float(np.median(w)) if w.size > 0 else x[i]
    return meds


def _one_sided_upper(x_raw: np.ndarray, flags: np.ndarray, span: int = CAUSAL_SPAN) -> np.ndarray:
    """Keep a flag only if the point is ABOVE its causal reference median
    on the ORIGINAL (untransformed) scale -- upper-tail-only for a signal
    where only an increase is meaningful."""
    meds = _causal_median(x_raw, span)
    return flags & (x_raw > meds)


def _global_3sigma(x: np.ndarray) -> np.ndarray:
    mu, sigma = float(np.mean(x)), float(np.std(x))
    z = np.abs(x - mu) / sigma if sigma > 1e-12 else np.zeros(len(x))
    return z > 3.0


def _dense_armor(x: np.ndarray) -> np.ndarray:
    labels, _, _ = classify_segments(x, **ARBITER_KW)
    return labels != "clean"


def _cusum(x: np.ndarray) -> np.ndarray:
    flagged, _ = cusum_detector(x, **CUSUM_KW)
    return flagged


def _ground_truth_mask(records, label: str) -> np.ndarray:
    return np.array([r["ground_truth"] == label for r in records])


def _rate_in_mask(flags: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(flags[mask]))


def _run_all_arms(x_raw: np.ndarray):
    """Returns (raw_two_sided, fixed_log_one_sided) dicts of arm->flags."""
    x_log = np.log(x_raw)

    raw = {
        "global_3sigma": _global_3sigma(x_raw),
        "dense_armor": _dense_armor(x_raw),
        "cusum": _cusum(x_raw),
    }
    fixed = {
        "global_3sigma": _one_sided_upper(x_raw, _global_3sigma(x_log)),
        "dense_armor": _one_sided_upper(x_raw, _dense_armor(x_log)),
        "cusum": _one_sided_upper(x_raw, _cusum(x_log)),
    }
    return raw, fixed


def test_benchmark_v2_1_log_onesided_report():
    by_scenario = _load_records()
    for scen in ("A_normal", "B_transient", "C_persistent", "D_legit_switch"):
        assert scen in by_scenario, f"missing scenario {scen} -- run run_benchmark_v2.py telemetry_v2_1_fresh.jsonl first"
        assert len(by_scenario[scen]) == 50

    print("\n" + "=" * 78)
    print("  Benchmark v2.1 -- log-transform + one-sided fix, FRESH real Qwen data")
    print("=" * 78)

    all_results = {}
    for scen, records in by_scenario.items():
        x_raw = np.array([r["latency_s"] for r in records])
        raw, fixed = _run_all_arms(x_raw)
        all_results[scen] = dict(records=records, raw=raw, fixed=fixed)

    print("\n[A] Normal -- false-positive rate, RAW (2-sided,linear) vs FIXED (log,1-sided)")
    r = all_results["A_normal"]
    for name in ("global_3sigma", "dense_armor", "cusum"):
        fp_raw = float(np.mean(r["raw"][name][10:]))
        fp_fixed = float(np.mean(r["fixed"][name][10:]))
        print(f"    {name:14s}  raw_FP={fp_raw:.3f}   fixed_FP={fp_fixed:.3f}")

    print("\n[B] Transient (steps 25-26 latency x8) -- detection rate")
    r = all_results["B_transient"]
    mask = _ground_truth_mask(r["records"], "transient_injected")
    for name in ("global_3sigma", "dense_armor", "cusum"):
        det_raw = _rate_in_mask(r["raw"][name], mask)
        det_fixed = _rate_in_mask(r["fixed"][name], mask)
        print(f"    {name:14s}  raw_detect={det_raw:.3f}   fixed_detect={det_fixed:.3f}")

    print("\n[C] Persistent (steps 25+ latency +2.0s) -- detection rate")
    r = all_results["C_persistent"]
    mask = _ground_truth_mask(r["records"], "persistent_shift")
    for name in ("global_3sigma", "dense_armor", "cusum"):
        det_raw = _rate_in_mask(r["raw"][name], mask)
        det_fixed = _rate_in_mask(r["fixed"][name], mask)
        print(f"    {name:14s}  raw_detect={det_raw:.3f}   fixed_detect={det_fixed:.3f}")

    print("\n[D] Legitimate task switch -- false-reject rate (want LOW)")
    r = all_results["D_legit_switch"]
    mask = _ground_truth_mask(r["records"], "legit_switch")
    for name in ("global_3sigma", "dense_armor", "cusum"):
        fr_raw = _rate_in_mask(r["raw"][name], mask)
        fr_fixed = _rate_in_mask(r["fixed"][name], mask)
        print(f"    {name:14s}  raw_false_reject={fr_raw:.3f}   fixed_false_reject={fr_fixed:.3f}")

    print("\n" + "=" * 78)

    # Structural sanity only -- NOT an assertion that the fix "wins":
    # per this project's own preregistration discipline, honest reporting
    # even of a fix that fails to help is the point.
    for scen, r in all_results.items():
        for arm_dict in (r["raw"], r["fixed"]):
            for name, flags in arm_dict.items():
                assert flags.dtype == bool
                assert len(flags) == 50
