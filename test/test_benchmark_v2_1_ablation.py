# -*- coding: utf-8 -*-
"""
test/test_benchmark_v2_1_ablation.py
========================================
Decomposes test_benchmark_v2_1_log_onesided.py's combined fix into its
two components, evaluated separately, to find out which one is actually
responsible for which effect: log-transform alone, one-sided-filter
alone, and both together (the original v2.1 result) vs. raw.

METHODOLOGICAL NOTE: this reuses telemetry_v2_1_fresh.jsonl, the SAME
frozen dataset test_benchmark_v2_1_log_onesided.py already reported on
-- not a new Qwen run. This is a legitimate ablation of an already-
committed result, not the kind of post-hoc threshold retuning this
project's benchmarks have avoided elsewhere: no parameter is being
searched or adjusted to make a number look better here, this file only
decomposes two already-declared, principled structural choices (log
transform; one-sided filtering) into their individual contributions on
data whose combined-fix result is already on the record.

FOUR VARIANTS per arm (dense_armor, cusum -- global_3sigma included for
completeness):
  raw          -- linear latency, two-sided (original v2 behavior)
  log_only     -- log(latency), two-sided
  onesided_only-- linear latency, one-sided upper-tail filter
  both         -- log(latency), one-sided filter (= v2.1's result)
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


_DETECTORS = {"global_3sigma": _global_3sigma, "dense_armor": _dense_armor, "cusum": _cusum}


def _run_variants(x_raw: np.ndarray):
    x_log = np.log(x_raw)
    variants = {}
    for name, det in _DETECTORS.items():
        raw_flags = det(x_raw)
        log_flags = det(x_log)
        variants[name] = {
            "raw": raw_flags,
            "log_only": log_flags,
            "onesided_only": _one_sided_upper(x_raw, raw_flags),
            "both": _one_sided_upper(x_raw, log_flags),
        }
    return variants


def _ground_truth_mask(records, label: str) -> np.ndarray:
    return np.array([r["ground_truth"] == label for r in records])


def _rate_in_mask(flags: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(flags[mask]))


def test_ablation_report():
    by_scenario = _load_records()
    for scen in ("A_normal", "B_transient", "C_persistent", "D_legit_switch"):
        assert scen in by_scenario, f"missing scenario {scen}"
        assert len(by_scenario[scen]) == 50

    print("\n" + "=" * 88)
    print("  Benchmark v2.1 ablation -- log-transform vs one-sided filter, isolated")
    print("=" * 88)

    all_variants = {}
    for scen, records in by_scenario.items():
        x_raw = np.array([r["latency_s"] for r in records])
        all_variants[scen] = dict(records=records, variants=_run_variants(x_raw))

    variant_names = ("raw", "log_only", "onesided_only", "both")

    print("\n[A] Normal -- false-positive rate")
    r = all_variants["A_normal"]
    for arm in ("dense_armor", "cusum"):
        vals = [float(np.mean(r["variants"][arm][v][10:])) for v in variant_names]
        print(f"    {arm:12s}  raw={vals[0]:.3f}  log_only={vals[1]:.3f}"
              f"  onesided_only={vals[2]:.3f}  both={vals[3]:.3f}")

    print("\n[C] Persistent (steps 25+ latency +2.0s) -- detection rate")
    r = all_variants["C_persistent"]
    mask = _ground_truth_mask(r["records"], "persistent_shift")
    for arm in ("dense_armor", "cusum"):
        vals = [_rate_in_mask(r["variants"][arm][v], mask) for v in variant_names]
        print(f"    {arm:12s}  raw={vals[0]:.3f}  log_only={vals[1]:.3f}"
              f"  onesided_only={vals[2]:.3f}  both={vals[3]:.3f}")

    print("\n[D] Legitimate task switch -- false-reject rate (want LOW)")
    r = all_variants["D_legit_switch"]
    mask = _ground_truth_mask(r["records"], "legit_switch")
    for arm in ("dense_armor", "cusum"):
        vals = [_rate_in_mask(r["variants"][arm][v], mask) for v in variant_names]
        print(f"    {arm:12s}  raw={vals[0]:.3f}  log_only={vals[1]:.3f}"
              f"  onesided_only={vals[2]:.3f}  both={vals[3]:.3f}")

    print("\n[B] Transient (steps 25-26 latency x8) -- detection rate (sanity, expect ~1.0 everywhere)")
    r = all_variants["B_transient"]
    mask = _ground_truth_mask(r["records"], "transient_injected")
    for arm in ("dense_armor", "cusum"):
        vals = [_rate_in_mask(r["variants"][arm][v], mask) for v in variant_names]
        print(f"    {arm:12s}  raw={vals[0]:.3f}  log_only={vals[1]:.3f}"
              f"  onesided_only={vals[2]:.3f}  both={vals[3]:.3f}")

    print("\n" + "=" * 88)

    for scen, r in all_variants.items():
        for arm, variants in r["variants"].items():
            for v, flags in variants.items():
                assert flags.dtype == bool and len(flags) == 50
