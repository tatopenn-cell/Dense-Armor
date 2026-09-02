# -*- coding: utf-8 -*-
"""
test/test_benchmark_v2_agent_runtime.py
=========================================
Evaluates the frozen Benchmark v2 dataset (test/agent_v2/telemetry_v2_
frozen.jsonl -- real Qwen2 1.8B tool-use trajectories via Ollama, see
test/agent_v2/run_benchmark_v2.py for how it was generated and the exact
scenario/injection protocol). This file does NO LLM calls -- it is the
"detector" half of the "raw trajectory -> feature extraction -> detector
-> decision" separation the generator script was built around, so the
detector side can be iterated on without re-running the real agent.

FEATURE: per-step latency_s is the monitored signal. It is the single
cleanest, most realistic runtime-observable feature in this telemetry
(a real production monitor plausibly watches exactly this), and it is
where scenarios B/C's controlled injections live (see the generator's
docstring) -- token counts/tool-choice are recorded in the frozen file
too but not analyzed here; a real production system would likely
combine several features, but per this project's own stated restraint
against building an unproven multi-feature fusion score before a single
feature has been shown to work, this file evaluates ONE feature first.

WINDOW PARAMETERS -- declared here BEFORE this file was ever run against
the frozen data, chosen for the real, short series length (50 steps per
scenario, not the 600-point synthetic benchmark): radius=5, ref_mult=2
(span=10), vs. the synthetic benchmark's radius=10/ref_mult=3 -- smaller
because the data itself is shorter, not because these numbers were
picked after looking at detection results. cusum k/h and the global-
3sigma/dense_armor thresholds are otherwise the same library defaults
used throughout this project.

FOUR ARMS: global_3sigma, dense_armor (classify_segments), cusum
(adaptive), and "arbiter_combined" (flags whenever EITHER dense_armor OR
cusum flags) -- the combination GPT's own advice named as the real
open question ("is the advantage really Dense-Armor's specific design,
or does any local/adaptive detector beat a naive global one").
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

_DATA_PATH = pathlib.Path(__file__).resolve().parent / "agent_v2" / "telemetry_v2_frozen.jsonl"

ARBITER_KW = dict(radius=5, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=5, ref_mult=2, k=0.5, h=5.0)


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


def _combined(dense_flags: np.ndarray, cusum_flags: np.ndarray) -> np.ndarray:
    return dense_flags | cusum_flags


def _ground_truth_mask(records, label: str) -> np.ndarray:
    return np.array([r["ground_truth"] == label for r in records])


def _rate_in_mask(flags: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(flags[mask]))


def _first_detection_latency(flags: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return float("nan")
    start = idx[0]
    for i in range(start, len(flags)):
        if flags[i]:
            return float(i - start)
    return float("nan")


def test_benchmark_v2_agent_runtime_report():
    by_scenario = _load_records()
    for scen in ("A_normal", "B_transient", "C_persistent", "D_legit_switch"):
        assert scen in by_scenario, f"missing scenario {scen} in {_DATA_PATH} -- run run_benchmark_v2.py first"
        assert len(by_scenario[scen]) == 50, f"{scen}: expected 50 steps, got {len(by_scenario[scen])}"

    print("\n" + "=" * 78)
    print("  Benchmark v2 -- Real Agent Runtime (Qwen2 1.8B via Ollama, frozen telemetry)")
    print("=" * 78)

    results = {}
    for scen, records in by_scenario.items():
        x = np.array([r["latency_s"] for r in records])
        flags_g3s = _global_3sigma(x)
        flags_da = _dense_armor(x)
        flags_cs = _cusum(x)
        flags_comb = _combined(flags_da, flags_cs)
        results[scen] = dict(x=x, records=records, g3s=flags_g3s, da=flags_da, cs=flags_cs, comb=flags_comb)

    # --- A: false-positive rate (normal region == the whole scenario) ---
    print("\n[A] Normal -- false-positive rate (valid_from=10 to skip window warmup)")
    r = results["A_normal"]
    for name, flags in (("global_3sigma", r["g3s"]), ("dense_armor", r["da"]), ("cusum", r["cs"]), ("arbiter_combined", r["comb"])):
        fp = float(np.mean(flags[10:]))
        print(f"    {name:18s}  FP={fp:.3f}")

    # --- B: transient -- detection rate + latency in the injected region ---
    print("\n[B] Transient (steps 25-26 latency x8) -- detection + latency")
    r = results["B_transient"]
    mask = _ground_truth_mask(r["records"], "transient_injected")
    for name, flags in (("global_3sigma", r["g3s"]), ("dense_armor", r["da"]), ("cusum", r["cs"]), ("arbiter_combined", r["comb"])):
        det = _rate_in_mask(flags, mask)
        lat = _first_detection_latency(flags, mask)
        print(f"    {name:18s}  detect={det:.3f}  latency={lat}")

    # --- C: persistent -- detection rate (full shifted region) + latency to first flag ---
    print("\n[C] Persistent (steps 25+ latency +2.0s) -- detection + latency")
    r = results["C_persistent"]
    mask = _ground_truth_mask(r["records"], "persistent_shift")
    for name, flags in (("global_3sigma", r["g3s"]), ("dense_armor", r["da"]), ("cusum", r["cs"]), ("arbiter_combined", r["comb"])):
        det = _rate_in_mask(flags, mask)
        lat = _first_detection_latency(flags, mask)
        print(f"    {name:18s}  detect={det:.3f}  latency={lat}")

    # --- D: legitimate switch -- false-reject rate (want LOW) ---
    print("\n[D] Legitimate task switch (steps 25+, real behavior change, no injection)")
    print("    -- false_reject here means the detector wrongly treats a genuine, benign")
    print("    task-domain switch as an anomaly. Low is good.")
    r = results["D_legit_switch"]
    mask = _ground_truth_mask(r["records"], "legit_switch")
    for name, flags in (("global_3sigma", r["g3s"]), ("dense_armor", r["da"]), ("cusum", r["cs"]), ("arbiter_combined", r["comb"])):
        fr = _rate_in_mask(flags, mask)
        print(f"    {name:18s}  false_reject={fr:.3f}")

    print("\n" + "=" * 78)

    # Structural sanity only, per this project's own preregistration
    # discipline -- not an assertion that any arm "wins".
    for scen, r in results.items():
        for flags in (r["g3s"], r["da"], r["cs"], r["comb"]):
            assert flags.dtype == bool
            assert len(flags) == 50
