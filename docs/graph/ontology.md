# A graph with rules

A plain graph says which nodes are joined. A typed graph also says what each node is, what it may hold and which condition must be true across each edge. This page builds a small plant, breaks one rule, and then changes the plant only through actions that check themselves first.

## 1. Typed nodes and constraints

`TypedGraph(schema)` takes a dict `{type: {field: python_type}}`. `add_node(name, type, **state)` adds a node whose fields must match the schema, and `add_edge(src, dst, relation, constraint)` adds an edge with an optional function `constraint(state_src, state_dst) -> bool`. `validate()` returns the edges whose constraint is false.

```python
from dense_armor.core.graph import TypedGraph

sch = {"sensor": {"battery": float, "status": str}, "system": {"mode": str}}
t = TypedGraph(sch)
t.add_node("s1", "sensor", battery=80.0, status="active")
t.add_node("sys", "system", mode="maintenance")
t.add_edge("sys", "s1", "monitors", constraint=lambda a, b: b["battery"] >= 0.0)
print(t.validate())
try:
    t.update("s1", battery="low")
except ValueError as e:
    print(e)
t.update("s1", battery=-5)
print(t.validate())
```

```text
[]
field 'battery' must be float, got str
[('sys', 's1', 'monitors')]
```

The plant is valid: its one constraint holds. A text where the schema wants a number is refused, and a battery of -5 is accepted by the type but breaks the constraint of the edge, which `validate()` reports as `(src, dst, relation)`. An unknown type, an unknown field, a name already used and a value of the wrong type raise `ValueError`. An `int` is accepted for a `float` field; a `bool` is accepted only for a `bool` field.

## 2. Actions with preconditions

`Action(name, preconditions, effects)` changes the graph, but only if its preconditions are true. A precondition is `(node, field, op, value)` with `op` one of `==`, `!=`, `<`, `>`, `<=`, `>=`. An effect is `(node, field, kind, value)` with `kind` `"set"`, `"add"` or `"delete"` (`"add"` and `"delete"` work on a list field). `apply_action(graph, action)` checks the constraints of the graph, then the preconditions, applies the effects on a copy and checks the constraints again.

```python
from dense_armor.core.graph import Action, TypedGraph, apply_action

sch = {"sensor": {"battery": float, "status": str}, "system": {"mode": str}}
t = TypedGraph(sch)
t.add_node("s1", "sensor", battery=80.0, status="active")
t.add_node("sys", "system", mode="maintenance")
a = Action(
    "swap",
    [("s1", "battery", ">", 20.0), ("sys", "mode", "==", "maintenance")],
    [("s1", "status", "set", "maintenance")],
)
g2 = apply_action(t, a)
print(g2.graph.nodes["s1"]["state"], t.graph.nodes["s1"]["state"])
```

```text
{'battery': 80.0, 'status': 'maintenance'} {'battery': 80.0, 'status': 'active'}
```

The new graph has the status changed and the first one is untouched: the copy is a deep copy, so no node shares its state with the original.

## 3. An action that is refused

When a precondition is false, nothing is applied and `ActionRejected` is raised. Its `failed` list says which condition failed and the value that was found.

```python
from dense_armor.core.graph import Action, ActionRejected, TypedGraph, apply_action

sch = {"sensor": {"battery": float, "status": str}, "system": {"mode": str}}
t = TypedGraph(sch)
t.add_node("s1", "sensor", battery=10.0, status="active")
t.add_node("sys", "system", mode="maintenance")
a = Action("swap", [("s1", "battery", ">", 20.0)], [("s1", "status", "set", "maintenance")])
try:
    apply_action(t, a)
except ActionRejected as e:
    print(e, e.failed)
```

```text
action 'swap' rejected: 1 precondition(s) failed [('s1', 'battery', '>', 20.0, 10.0)]
```

A node that does not exist in a precondition is a failed precondition with the value `"no such node"`. In `params={"s1": "s7"}` a name in the action is replaced by another node of the graph.

## Details

The schema is a plain dict of Python types, and the effects are the three kinds above: this is a choice of this module. The cited paper (LLMs4OL, Babaei Giglou and others 2023) is about learning ontologies with language models and not about a typed graph; a full OWL or SHACL implementation would be a different and heavier choice. The actions follow the idea of STRIPS (Fikes and Nilsson 1971): preconditions, then effects. `apply_action` raises `ActionRejected` also when the graph already breaks a constraint before the action, or breaks it after: in those two cases `failed` lists `(src, dst, relation)` of the edges, and the message still says "precondition(s)".

Two points changed in the second review. A node name used twice is now refused, where before the second node replaced the first. A number for a `float` field was refused when it was an `int`, and `True` was accepted for an `int` field.
