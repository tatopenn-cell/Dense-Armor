"""Apply a synthesized rule list to the test inputs of a task."""

from __future__ import annotations

from typing import Any

from dense_armor.core.graph.graph_diff import level2_nodes
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects
from dense_armor.core.graph.rule_synthesis import Rule, render


def apply_to_test(rules: list[Rule], task: dict[str, Any]) -> dict[str, Any]:
    """Return the ARC submission dict with the test outputs filled."""
    test_out = []
    for ex in task.get("test", []):
        g = Grid.from_list(ex["input"])
        L2 = level2_nodes(g, extract_objects(g))
        out = render(g, rules, L2)
        test_out.append({"input": ex["input"], "output": out.to_list()})
    return {"train": task.get("train", []), "test": test_out}
