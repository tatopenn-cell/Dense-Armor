"""Tests for GateNode with a fake detector."""

from dense_armor.core.graph import GateNode
from dense_armor.roles import Signal


class _FakeDetector:
    """A detector whose score is the value of the 'x' key."""

    budget_s = 1e-3
    memory_class = "O(1)"
    threshold = 5.0

    def __init__(self):
        self.n = 0

    def score_one(self, x, t=None):
        return float(x["x"])

    def learn_one(self, x, t=None):
        self.n += 1
        return self


def _sig(v):
    return Signal([float(v)], names=["x"], units=[""])


def test_gate_blocks_anomaly():
    det = _FakeDetector()
    gate = GateNode(det, threshold=5.0)
    assert gate(_sig(1.0)) is not None
    assert gate(_sig(10.0)) is None
    assert gate.blocked == 1
    assert gate.passed == 1


def test_gate_uses_detector_threshold():
    det = _FakeDetector()
    gate = GateNode(det)
    assert gate(_sig(1.0)) is not None
    assert gate(_sig(10.0)) is None


def test_gate_learns_only_from_passed_samples():
    det = _FakeDetector()
    gate = GateNode(det)
    gate(_sig(1.0))
    gate(_sig(10.0))
    assert det.n == 1


class _Counter:
    """Conta le chiamate a learn_one."""

    def __init__(self):
        self.n = 0

    def learn_one(self, x, y=None):
        self.n += 1
        return self

    def transform_one(self, x):
        return dict(x)


def test_gate_inside_dag():
    from dense_armor.core.graph import ExecutionDAG, GateNode

    counter = _Counter()
    gate = GateNode(_FakeDetector(), threshold=5.0)
    dag = ExecutionDAG(
        nodes={"a": gate, "b": counter},
        edges=[("a", "b")],
    )
    # anomalia: b non riceve learn_one
    dag.run({"a": _sig(10.0)})
    assert gate.blocked == 1
    assert counter.n == 0
    # normale: b riceve learn_one
    dag.run({"a": _sig(1.0)})
    assert gate.passed == 1
    assert counter.n == 1


def test_gate_scores_each_sample_once_and_declares_budget():
    class _Counting(_FakeDetector):
        def __init__(self):
            super().__init__()
            self.scored = 0

        def score_one(self, x, t=None):
            self.scored += 1
            return super().score_one(x, t)

    det = _Counting()
    gate = GateNode(det)
    gate(_sig(1.0))
    gate(_sig(10.0))
    assert det.scored == 2
    assert gate.budget_s == 2.0 * det.budget_s
    assert gate.memory_class == "O(1)"
