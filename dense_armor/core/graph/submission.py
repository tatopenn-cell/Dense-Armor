"""Convert a Program and a task into the ARC submission format.

The output is ``{"train": [...unchanged...],
"test": [{"input": [...], "output": [...]}, ...]}``: the test inputs
are kept, the test outputs are the result of applying the program to
them.
"""

from __future__ import annotations

from typing import Any

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.program import Program


def to_submission(program: Program, task: dict[str, Any]) -> dict[str, Any]:
    """Return the ARC submission dict for ``task`` with test outputs filled.

    Args:
        program: the program that solves the task.
        task: the task dict with a ``"test"`` list of ``{"input": ...}``.

    Returns:
        a dict with ``"train"`` unchanged and ``"test"`` with each
        ``{"input": ..., "output": ...}``, where ``output`` is the
        result of applying ``program`` to the test input.
    """
    test_out: list[dict[str, Any]] = []
    for ex in task.get("test", []):
        g = Grid.from_list(ex["input"])
        test_out.append({"input": ex["input"], "output": program.apply(g).to_list()})
    return {"train": task.get("train", []), "test": test_out}
