# -*- coding: utf-8 -*-
"""
test/agent_v2/agent.py
========================
Minimal single-turn tool-using agent over Ollama's qwen:latest (Qwen2
1.8B, Q4_K_M). Built for Benchmark v2 (real agent runtime telemetry),
after finding directly (not assumed from the model's "tools" capability
tag) that this model does NOT reliably return Ollama's structured
tool_calls field -- it emits the tool call as free-form JSON inside the
text content instead. This module parses that manually (a small ReAct-
style loop), rather than relying on Ollama's tool-calling API.

Each "step" is ONE LLM call per task (not a full multi-turn tool-use
conversation with a second finalization call) -- a deliberate scope
choice given each call takes ~10-15s on this machine: what this
benchmark needs is real telemetry of the model's tool-use DECISION
(which tool, latency, tokens, whether it errored/retried), not a
polished final natural-language answer.

No raw prompt/response TEXT is persisted by the telemetry recorder in
test/agent_v2/run_benchmark_v2.py -- only structured features (tool
name, argument length, result length, latency, token counts, error
flags), per the "don't save everything indiscriminately" guidance this
was built against.
"""
import json
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import ollama

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from tools import TOOL_SPECS, TOOL_FUNCS  # noqa: E402

MODEL = "qwen:latest"

_TOOL_LIST = "\n".join(
    f"- {t['name']}({', '.join(t['parameters'])}): {t['description']}" for t in TOOL_SPECS
)

SYSTEM_PROMPT = f"""You are a helpful assistant with access to these tools:
{_TOOL_LIST}

If a tool would help answer the user's request, respond with ONLY a JSON object of the form:
{{"tool": "<tool_name>", "arguments": {{...}}}}
Otherwise answer directly in plain text. Never explain the JSON, just emit it alone when using a tool."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class StepResult:
    task: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    tool_name: Optional[str]
    tool_arguments_repr: Optional[str]
    tool_result_repr: Optional[str]
    error: bool
    error_kind: Optional[str]
    raw_content_len: int


def _extract_tool_call(content: str):
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj


def run_step(task: str) -> StepResult:
    t0 = time.perf_counter()
    error, error_kind = False, None
    tool_name = tool_args_repr = tool_result_repr = None
    tokens_in = tokens_out = 0
    content = ""

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            options={"temperature": 0.2},
        )
        content = resp["message"]["content"] or ""
        tokens_in = int(resp.get("prompt_eval_count", 0) or 0)
        tokens_out = int(resp.get("eval_count", 0) or 0)
    except Exception as exc:
        error, error_kind = True, f"llm_call_failed:{type(exc).__name__}"

    if not error:
        call = _extract_tool_call(content)
        if call is not None:
            tool_name = str(call.get("tool"))
            args = call.get("arguments", {})
            tool_args_repr = json.dumps(args, sort_keys=True)[:200]
            func = TOOL_FUNCS.get(tool_name)
            if func is None:
                error, error_kind = True, "unknown_tool"
            else:
                try:
                    result = func(**args) if isinstance(args, dict) else func(str(args))
                    tool_result_repr = str(result)[:200]
                except Exception as exc:
                    error, error_kind = True, f"tool_call_failed:{type(exc).__name__}"

    latency_s = time.perf_counter() - t0
    return StepResult(
        task=task, latency_s=latency_s, tokens_in=tokens_in, tokens_out=tokens_out,
        tool_name=tool_name, tool_arguments_repr=tool_args_repr,
        tool_result_repr=tool_result_repr, error=error, error_kind=error_kind,
        raw_content_len=len(content),
    )
