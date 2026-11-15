"""Tests for dense_armor.core.graph.submission."""

from dense_armor.core.graph import dsl
from dense_armor.core.graph.program import Program
from dense_armor.core.graph.submission import to_submission


def test_to_submission_fills_test_outputs():
    task = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }
    program = Program([dsl.hmirror])
    sub = to_submission(program, task)
    assert sub["train"] == task["train"]
    assert sub["test"][0]["input"] == [[3, 4]]
    assert sub["test"][0]["output"] == [[4, 3]]


def test_to_submission_multiple_test():
    task = {"train": [], "test": [{"input": [[1]]}, {"input": [[2]]}]}
    sub = to_submission(Program([dsl.rot90]), task)
    assert len(sub["test"]) == 2


def test_to_submission_preserves_train():
    task = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}
    sub = to_submission(Program([]), task)
    assert sub["train"] == task["train"]
    assert sub["test"] == []
