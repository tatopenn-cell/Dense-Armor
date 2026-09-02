# -*- coding: utf-8 -*-
"""
test/agent_v2/run_benchmark_v2.py
===================================
Generates the frozen Benchmark v2 telemetry dataset: real Qwen2 1.8B
(via Ollama, qwen:latest) tool-use trajectories, NOT synthetic gaussian
noise. Run once; writes test/agent_v2/telemetry_v2_frozen.jsonl, which
test/test_benchmark_v2_agent_runtime.py loads and evaluates (fast, no
LLM calls) -- the "raw trajectory -> feature extraction -> detector ->
decision" separation this was built around, so the detector side can be
iterated on without re-running the LLM.

FOUR SCENARIOS (A, B, C, D -- deliberately 4, not the original 6-
scenario proposal, to keep this a tractable first real-agent
iteration):
  A. Normal      -- 50 steps, a fixed mixed pool of math/definition/
                     word-count tasks. Real baseline for FP rate.
  B. Transient    -- SAME 50-step real trajectory as A, but steps 25-26
                     have their RECORDED latency multiplied 8x, an
                     explicit, documented, controlled fault injected at
                     the telemetry layer (not a change to the real LLM
                     run) -- ground truth exact.
  C. Persistent   -- SAME 50-step real trajectory as A, but every step
                     from 25 onward has a fixed +2.0s added to recorded
                     latency, simulating sustained system degradation --
                     ground truth exact.
  D. Legit switch -- a SEPARATE real 50-step trajectory: steps 1-25 the
                     same task pool as A, steps 26-50 switch to a
                     genuinely different task pool (open questions that
                     need no tool at all) -- NO injection, a real
                     behavioral change from the model itself. This is
                     the one scenario that must NOT be treated as
                     equivalent to B/C by a detector deciding reject
                     vs. pass.

B and C deliberately reuse A's real trajectory rather than each running
their own separate real 50 steps: this keeps the "clean" portion
identical across A/B/C (only the controlled injection differs), the
same principle applied when the earlier synthetic benchmark's per-
severity noise-seed confound was fixed -- and it roughly halves total
LLM calls (100 instead of 150).

Telemetry recorded per step (see agent.StepResult): latency_s,
tokens_in, tokens_out, tool_name, tool_arguments_repr/tool_result_repr
(truncated to 200 chars, not full raw prompts), error/error_kind,
raw_content_len. Plus scenario, step_id, ground_truth_label.
"""
import dataclasses
import json
import pathlib
import sys
import time

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from agent import run_step  # noqa: E402
from tools import _KNOWLEDGE  # noqa: E402

MATH_TASKS = [
    "What is 12*7?", "What is 144/12?", "What is 9+16?", "What is 23-8?",
    "What is 6*6?", "What is 100/4?", "What is 17+25?", "What is 81/9?",
    "What is 14*3?", "What is 50-19?",
]
DEFINE_TASKS = [f"Define '{term}'." for term in _KNOWLEDGE]
COUNT_TASKS = [
    "How many words are in: the quick brown fox jumps over?",
    "How many words are in: a small step for one person?",
    "How many words are in: real numbers only, no shortcuts here?",
    "How many words are in: dense armor watches every signal closely?",
    "How many words are in: statistics beats vague intuition every time?",
]
NORMAL_POOL = MATH_TASKS + DEFINE_TASKS + COUNT_TASKS  # 25 tasks, cycled

SWITCH_TASKS = [
    "What color is the sky on a clear day?",
    "Name a primary color.",
    "What is the capital of France?",
    "What sound does a cat make?",
    "Name a season of the year.",
    "What is the opposite of hot?",
    "Name a day of the week.",
    "What shape has three sides?",
    "What do bees produce?",
    "Name a planet in the solar system.",
]

N_STEPS = 50
SWITCH_AT = 25
TRANSIENT_AT = 25
TRANSIENT_WIDTH = 2
PERSISTENT_FROM = 25
PERSISTENT_EXTRA_S = 2.0
TRANSIENT_MULT = 8.0

OUT_PATH = _THIS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "telemetry_v2_frozen.jsonl")


def _pool_cycle(pool, n):
    return [pool[i % len(pool)] for i in range(n)]


def _run_trajectory(tasks, label):
    rows = []
    for step_id, task in enumerate(tasks):
        t0 = time.perf_counter()
        r = run_step(task)
        elapsed_total = time.perf_counter() - t0
        row = dataclasses.asdict(r)
        row["step_id"] = step_id
        row["label"] = label
        row["wall_elapsed_s"] = elapsed_total
        rows.append(row)
        print(f"  [{label}] step {step_id:3d}  tool={r.tool_name!s:12s}  "
              f"latency={r.latency_s:6.2f}s  error={r.error}")
    return rows


def main():
    print("=== Benchmark v2 telemetry generation (real Qwen2 1.8B via Ollama) ===")

    print("\n[A/B/C base trajectory] 50 steps, normal task pool")
    base_tasks = _pool_cycle(NORMAL_POOL, N_STEPS)
    base_rows = _run_trajectory(base_tasks, "base")

    print("\n[D trajectory] 50 steps, switches task pool at step 25")
    d_tasks = _pool_cycle(NORMAL_POOL, SWITCH_AT) + _pool_cycle(SWITCH_TASKS, N_STEPS - SWITCH_AT)
    d_rows = _run_trajectory(d_tasks, "d_switch")

    all_records = []

    # --- Scenario A: base trajectory, untouched ---
    for row in base_rows:
        rec = dict(row)
        rec["scenario"] = "A_normal"
        rec["ground_truth"] = "normal"
        all_records.append(rec)

    # --- Scenario B: base trajectory, transient injected at telemetry layer ---
    for row in base_rows:
        rec = dict(row)
        rec["scenario"] = "B_transient"
        if TRANSIENT_AT <= rec["step_id"] < TRANSIENT_AT + TRANSIENT_WIDTH:
            rec["latency_s"] = rec["latency_s"] * TRANSIENT_MULT
            rec["ground_truth"] = "transient_injected"
        else:
            rec["ground_truth"] = "normal"
        all_records.append(rec)

    # --- Scenario C: base trajectory, persistent shift injected at telemetry layer ---
    for row in base_rows:
        rec = dict(row)
        rec["scenario"] = "C_persistent"
        if rec["step_id"] >= PERSISTENT_FROM:
            rec["latency_s"] = rec["latency_s"] + PERSISTENT_EXTRA_S
            rec["ground_truth"] = "persistent_shift"
        else:
            rec["ground_truth"] = "normal"
        all_records.append(rec)

    # --- Scenario D: real, separate trajectory, real task switch, no injection ---
    for row in d_rows:
        rec = dict(row)
        rec["scenario"] = "D_legit_switch"
        rec["ground_truth"] = "legit_switch" if rec["step_id"] >= SWITCH_AT else "normal"
        all_records.append(rec)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(all_records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
