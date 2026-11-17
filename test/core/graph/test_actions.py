"""Tests for Action and apply_action (STRIPS)."""

import numpy as np
import pytest

from dense_armor.core.graph import (
    Action,
    ActionRejected,
    TypedGraph,
    apply_action,
)

SCHEMA = {
    "sensor": {"battery": float, "status": str, "alarms": list},
    "system": {"mode": str},
}


def _graph():
    g = TypedGraph(schema=SCHEMA)
    g.add_node("s1", "sensor", battery=80.0, status="active", alarms=[])
    g.add_node("sys", "system", mode="maintenance")
    return g


def test_rejected_when_precondition_false():
    g = _graph()
    a = Action(
        name="replace_sensor",
        preconditions=[
            ("s1", "battery", ">", 20.0),
            ("s1", "status", "==", "active"),
            ("sys", "mode", "==", "maintenance"),
        ],
        effects=[("s1", "status", "set", "maintenance")],
    )
    g2 = TypedGraph(schema=SCHEMA)
    g2.graph = g.graph.copy()
    g2.graph.nodes["s1"]["state"]["battery"] = 10.0
    with pytest.raises(ActionRejected) as exc:
        a.apply(g2)
    assert len(exc.value.failed) == 1
    assert exc.value.failed[0][0] == "s1"


def test_applied_when_all_true():
    g = _graph()
    a = Action(
        name="replace_sensor",
        preconditions=[
            ("s1", "battery", ">", 20.0),
            ("s1", "status", "==", "active"),
            ("sys", "mode", "==", "maintenance"),
        ],
        effects=[("s1", "status", "set", "maintenance")],
    )
    g2 = apply_action(g, a)
    assert g2.graph.nodes["s1"]["state"]["status"] == "maintenance"
    assert g.graph.nodes["s1"]["state"]["status"] == "active"


def test_effect_add_and_delete():
    g = _graph()
    a = Action(
        name="add_alarm",
        preconditions=[],
        effects=[("s1", "alarms", "add", "high_temp")],
    )
    g2 = a.apply(g)
    assert g2.graph.nodes["s1"]["state"]["alarms"] == ["high_temp"]
    a2 = Action(
        name="clear_alarm",
        preconditions=[],
        effects=[("s1", "alarms", "delete", "high_temp")],
    )
    g3 = a2.apply(g2)
    assert g3.graph.nodes["s1"]["state"]["alarms"] == []


def test_params_substitution():
    g = _graph()
    a = Action(
        name="set_battery",
        preconditions=[("s1", "battery", ">=", 0.0)],
        effects=[("s1", "battery", "set", 42.0)],
    )
    g2 = a.apply(g, params={"s1": "s1"})
    assert g2.graph.nodes["s1"]["state"]["battery"] == 42.0


def test_missing_node_in_precondition():
    g = _graph()
    a = Action(
        name="bad",
        preconditions=[("nope", "x", "==", 1)],
        effects=[],
    )
    with pytest.raises(ActionRejected):
        a.apply(g)


def test_deepcopy_other_nodes():
    """L'azione non modifica gli altri nodi (deepcopy, non shallow)."""
    g = _graph()
    g.add_node("s2", "sensor", battery=50.0, status="active", alarms=[])
    a = Action(
        name="set_s1",
        preconditions=[],
        effects=[("s1", "status", "set", "maintenance")],
    )
    g2 = a.apply(g)
    assert g2.graph.nodes["s2"]["state"]["battery"] == 50.0
    # modificare g2 non deve toccare g
    g2.graph.nodes["s2"]["state"]["battery"] = 0.0
    assert g.graph.nodes["s2"]["state"]["battery"] == 50.0


def test_int_is_a_float_and_bool_is_not_an_int():
    g = _graph()
    g.add_node("s3", "sensor", battery=80, status="active", alarms=[])
    with pytest.raises(ValueError):
        g.update("s3", battery=True)
    with pytest.raises(ValueError):
        g.add_node("s1", "sensor", battery=1.0)


def test_set_accepts_int_for_float():
    a = Action(name="b", effects=[("s1", "battery", "set", 42)])
    assert a.apply(_graph()).graph.nodes["s1"]["state"]["battery"] == 42


def test_rejection_says_whether_a_constraint_or_a_precondition_failed():
    g = _graph()
    g.add_edge("sys", "s1", "monitors", constraint=lambda a, b: b["status"] == "active")
    a = Action(name="off", effects=[("s1", "status", "set", "off")])
    with pytest.raises(ActionRejected, match="new constraint.*broken by the action"):
        apply_action(g, a)
    b = Action(name="b", preconditions=[("s1", "battery", "<", 1.0)])
    with pytest.raises(ActionRejected, match="precondition"):
        apply_action(g, b)


def test_a_graph_that_already_breaks_a_rule_can_still_be_changed_safely():
    g = _graph()
    g.add_node("s2", "sensor", battery=50.0, status="off", alarms=[])
    g.add_edge("sys", "s1", "monitors", constraint=lambda a, b: b["status"] == "active")
    g.add_edge("sys", "s2", "monitors", constraint=lambda a, b: b["status"] == "active")
    assert g.validate() == [("sys", "s2", "monitors")]
    ok = Action(name="charge", effects=[("s1", "battery", "set", 90.0)])
    assert apply_action(g, ok).graph.nodes["s1"]["state"]["battery"] == 90.0
    bad = Action(name="off", effects=[("s1", "status", "set", "off")])
    with pytest.raises(ActionRejected) as exc:
        apply_action(g, bad)
    assert exc.value.failed == [("sys", "s1", "monitors")]


def test_constraint_is_named_and_validate_can_raise():
    from dense_armor.core.graph import Constraint, ConstraintViolation

    rule = Constraint("active only", lambda a, b: b["status"] == "active")
    g = _graph()
    g.add_edge("sys", "s1", "monitors", constraint=rule)
    assert repr(rule) == "Constraint('active only')"
    assert g.validate(raise_on_violation=True) == []
    g.update("s1", status="off")
    with pytest.raises(ConstraintViolation) as exc:
        g.validate(raise_on_violation=True)
    assert exc.value.failed == [("sys", "s1", "monitors")]
    assert g.validate() == [("sys", "s1", "monitors")]


def _facts(graph):
    return {
        (n, f, repr(v))
        for n, d in graph.graph.nodes(data=True)
        for f, v in d["state"].items()
    }


def test_action_matches_the_classical_strips_semantics():
    g = _graph()
    pre = {("s1", "status", repr("active")), ("s1", "battery", repr(80.0))}
    delete = {("s1", "status", repr("active"))}
    add = {("s1", "status", repr("maintenance"))}
    state = _facts(g)
    assert pre <= state
    expected = (state - delete) | add
    a = Action(
        name="maintain",
        preconditions=[("s1", "status", "==", "active"), ("s1", "battery", "==", 80.0)],
        effects=[("s1", "status", "set", "maintenance")],
    )
    assert _facts(a.apply(g)) == expected


def test_action_with_a_false_precondition_leaves_the_state_alone():
    g = _graph()
    before = _facts(g)
    a = Action(
        name="maintain",
        preconditions=[("s1", "status", "==", "off")],
        effects=[("s1", "status", "set", "maintenance")],
    )
    with pytest.raises(ActionRejected):
        a.apply(g)
    assert _facts(g) == before


def _random_system(seed, n=30):
    rng = np.random.default_rng(seed)
    g = TypedGraph({"part": {"level": int}})
    for i in range(n):
        g.add_node(f"p{i}", "part", level=int(rng.integers(0, 4)))
    for _ in range(3 * n):
        a, b = rng.integers(0, n, 2)
        if a != b:
            g.add_edge(f"p{a}", f"p{b}", "feeds", lambda s, d: s["level"] <= d["level"])
    return g, rng


def _full_check(g, action, params):
    before = set(g.validate())
    new = action.apply(g, params)
    return [v for v in new.validate() if v not in before]


def test_checking_only_the_changed_nodes_equals_checking_the_whole_graph():
    for seed in range(25):
        g, rng = _random_system(seed)
        node = f"p{int(rng.integers(0, 30))}"
        action = Action(
            name="set",
            effects=[("x", "level", "set", int(rng.integers(0, 4)))],
        )
        expected = _full_check(g, action, {"x": node})
        if expected:
            with pytest.raises(ActionRejected) as exc:
                apply_action(g, action, {"x": node})
            assert sorted(exc.value.failed) == sorted(expected)
        else:
            assert apply_action(g, action, {"x": node}) is not None


def test_validate_of_some_nodes_lists_only_their_relations():
    g = TypedGraph({"part": {"level": int}})
    for name, level in (("a", 3), ("b", 1), ("c", 1), ("d", 3)):
        g.add_node(name, "part", level=level)
    g.add_edge("a", "b", "feeds", lambda s, d: s["level"] <= d["level"])
    g.add_edge("c", "d", "feeds", lambda s, d: s["level"] <= d["level"])
    g.add_edge("d", "a", "feeds", lambda s, d: s["level"] <= d["level"])
    assert g.validate() == [("a", "b", "feeds")]
    assert g.validate(nodes=["c"]) == []
    assert g.validate(nodes=["b"]) == [("a", "b", "feeds")]
    assert g.validate(nodes=["zzz"]) == []
