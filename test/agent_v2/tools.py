# -*- coding: utf-8 -*-
"""
test/agent_v2/tools.py
=======================
Three small, deterministic tools for the minimal agent loop used to
generate real telemetry for Benchmark v2 (agent runtime). Kept tiny and
side-effect-free on purpose -- this is a laboratory for measuring a real
LLM's tool-use behavior over time, not a production agent.
"""
import ast
import operator

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def calc(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+-*/**), no names/calls allowed."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:
        return f"error: {exc}"


_KNOWLEDGE = {
    "entropy": "a measure of disorder or uncertainty in a system",
    "gradient": "a vector of partial derivatives pointing in the direction of steepest increase",
    "qubit": "the basic unit of quantum information, a two-level quantum system",
    "latency": "the time delay between a request and its response",
    "overhead": "the extra resource cost a system incurs beyond the useful work it does",
    "drift": "a gradual change in a signal's statistical properties over time",
    "variance": "the average squared deviation of a random variable from its mean",
    "median": "the middle value of a sorted dataset",
    "outlier": "a data point that differs significantly from other observations",
    "kernel": "a small, reusable function or the core component of a larger system",
}


def lookup(term: str) -> str:
    """Look up a short definition for `term` in a small fixed local knowledge base."""
    key = term.strip().lower().strip(".,?!'\"")
    return _KNOWLEDGE.get(key, f"no definition found for '{term}'")


def word_count(text: str) -> str:
    """Count the whitespace-separated words in `text`."""
    return str(len(text.split()))


TOOL_SPECS = [
    {
        "name": "calc",
        "description": "Evaluate a basic arithmetic expression, e.g. '12*7' or '144/12'.",
        "parameters": {"expression": "string, the arithmetic expression"},
    },
    {
        "name": "lookup",
        "description": "Look up the definition of a technical term (entropy, gradient, qubit, latency, overhead, drift, variance, median, outlier, kernel).",
        "parameters": {"term": "string, the term to define"},
    },
    {
        "name": "word_count",
        "description": "Count the number of words in a piece of text.",
        "parameters": {"text": "string, the text to count"},
    },
]

TOOL_FUNCS = {"calc": calc, "lookup": lookup, "word_count": word_count}
