"""Tests for Action and apply_action (STRIPS)."""

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
