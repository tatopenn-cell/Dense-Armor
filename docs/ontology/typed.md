# Types, rules and actions

## What it is

`TypedGraph` is a list of the parts of your system and of the links between them, kept in a way a program can check.

- A **part** has a **kind** (`service`, `database`) and some **values** (a service has a `tier`, a database has an `env`).
- A **link** joins two parts (`billing` depends on `billing-db`).
- A **rule** says when a link is wrong. You write it once. The library finds every link that breaks it.
- An **action** is a change you allow only when its conditions hold and no rule would break.

Why use it: a system with hundreds of parts cannot be checked by looking at it. With a `TypedGraph` you ask "which links are wrong?" and get a short list.

The example on this page is the software of a company. The rule: **a service in production must not use a test database.**

## 1. Add a part

First say which kinds exist and which values each kind holds, with their types. Then add parts one by one.

```python
from dense_armor.core.graph import TypedGraph

g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
g.add_node("billing", "service", tier="production")
g.add_node("billing-db", "database", env="test")
print(g.graph.nodes["billing"])
```

```text
{'type': 'service', 'state': {'tier': 'production'}}
```

`TypedGraph(...)` takes a dict: the name of each kind, and for each kind its values with their types. `add_node(name, kind, **values)` adds a part. The part remembers its kind (`type`) and its values (`state`).

## 2. A wrong part is refused

If a part does not fit its kind, `add_node` stops it at once.

```python
from dense_armor.core.graph import TypedGraph

g = TypedGraph({"service": {"tier": str}})
try:
    g.add_node("billing", "service", tier=3)
except ValueError as e:
    print(e)
```

```text
field 'tier' must be str, got int
```

The kind said `tier` is text, and `3` is a number. A value of the wrong type, a value the kind does not have, a kind that does not exist and a name already used all raise `ValueError`. So wrong data cannot get in, and you wrote the check only once, in the kinds.

## 3. Link two parts and write a rule

A rule is a small function. It receives the values of the two ends of a link and answers `True` if the link is fine, `False` if it is wrong.

```python
from dense_armor.core.graph import TypedGraph


def prod_needs_prod_db(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
g.add_node("billing", "service", tier="production")
g.add_node("billing-db", "database", env="test")
g.add_edge("billing", "billing-db", "depends_on", prod_needs_prod_db)
print(g.validate())
g.update("billing-db", env="production")
print(g.validate())
```

```text
[('billing', 'billing-db', 'depends_on')]
[]
```

`add_edge(from, to, name_of_the_link, rule)` creates the link and attaches the rule. `validate()` checks every link and returns the wrong ones, each as `(from, to, name_of_the_link)`. Here `billing` is in production and its database is a test one, so the link is listed. `update(name, **values)` changes the values of a part, with the same type check as `add_node`: after the database is moved to production, `validate()` returns an empty list.

## 4. Make a broken rule an error

Sometimes you do not want a list: you want the program to stop. Give the rule a name with `Constraint`, and ask `validate` to raise.

```python
from dense_armor.core.graph import Constraint, ConstraintViolation, TypedGraph

rule = Constraint("production needs a production database", lambda s, d: not (s["tier"] == "production" and d["env"] == "test"))
g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
g.add_node("billing", "service", tier="production")
g.add_node("billing-db", "database", env="test")
g.add_edge("billing", "billing-db", "depends_on", rule)
try:
    g.validate(raise_on_violation=True)
except ConstraintViolation as e:
    print(e.failed)
```

```text
[('billing', 'billing-db', 'depends_on')]
```

`Constraint(name, check)` is the same rule with a name, and `add_edge` accepts it as it accepts a plain function. With `raise_on_violation=True`, `validate` raises `ConstraintViolation`, and the wrong links are in its `failed` attribute.

## 5. Change the system with an action

An action is a change with conditions. It has two lists:

- **preconditions**, what must be true before: `(part, value, test, expected)`, with `test` one of `==`, `!=`, `<`, `>`, `<=`, `>=`;
- **effects**, what changes: `(part, value, how, new_value)`, with `how` equal to `"set"`, `"add"` or `"delete"` (the last two are for a list).

The names inside an action are placeholders. `params` says which part each one stands for, so one action works for every service. The idea comes from STRIPS (Fikes and Nilsson 1971).

`apply_action(graph, action, params)` makes the change on a **copy** and refuses it if it would break a rule that held.

```python
from dense_armor.core.graph import Action, ActionRejected, TypedGraph, apply_action


def prod_needs_prod_db(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
for name, env in (("search", "production"), ("reports", "test")):
    g.add_node(name, "service", tier="staging")
    g.add_node(f"{name}-db", "database", env=env)
    g.add_edge(name, f"{name}-db", "depends_on", prod_needs_prod_db)
promote = Action("promote", [("s", "tier", "==", "staging")], [("s", "tier", "set", "production")])
for s in ("search", "reports"):
    try:
        print(s, "promoted:", apply_action(g, promote, {"s": s}).graph.nodes[s]["state"]["tier"])
    except ActionRejected as e:
        print(s, e, e.failed)
```

```text
search promoted: production
reports action 'promote' rejected: 1 new constraint(s) broken by the action [('reports', 'reports-db', 'depends_on')]
```

The action `promote` moves a service from staging to production. `search` uses a production database, so it is promoted. `reports` uses a test database, so promoting it would break the rule: it is refused, and the message names the link. The graph `g` is not changed by either call, because the action works on a copy. If a precondition is false, `ActionRejected` is raised too.

## 6. A large system

The same kinds and the same rule work for any size. Here 200 services use 200 databases, and every tenth database is a test one.

```python
from collections import Counter
from dense_armor.core.graph import TypedGraph


def prod_needs_prod_db(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


g = TypedGraph({"service": {"tier": str, "owner": str}, "database": {"env": str}})
for k in range(200):
    g.add_node(f"db{k}", "database", env="test" if k % 10 == 0 else "production")
    g.add_node(f"svc{k}", "service", tier="production" if k % 4 else "staging", owner=f"team{k % 5}")
    g.add_edge(f"svc{k}", f"db{k}", "depends_on", prod_needs_prod_db)
nodes = g.to_networkx().nodes(data=True)
print(Counter(d["type"] for _, d in nodes), Counter(d["state"]["tier"] for _, d in nodes if d["type"] == "service"))
bad = g.validate()
print(len(bad), bad[:3])
```

```text
Counter({'database': 200, 'service': 200}) Counter({'production': 150, 'staging': 50})
10 [('svc10', 'db10', 'depends_on'), ('svc30', 'db30', 'depends_on'), ('svc50', 'db50', 'depends_on')]
```

`to_networkx()` gives the graph to NetworkX, where counting the parts by kind or by any value is one line: 200 services and 200 databases, 150 services in production and 50 in staging. Out of 200 links, 10 break the rule, and the list says which ones. That short list is what you read, not the whole system.

A system this size always has some wrong links. `apply_action` starts from the wrong links that already exist and refuses an action only if it makes a new one, so the ten that are wrong do not stop you from promoting `svc4`.

## Details

A kind holds values, and no kind inherits from another. The kinds are a plain dict of Python types, and a rule is a function of the values at the two ends of a link. An `int` is accepted where the kind says `float`, and a `bool` only where it says `bool`. The actions are written in this module and do not use a planning library; a full OWL or SHACL implementation would be a different and heavier choice. `apply_action` checks only the relations of the parts that its effects change, as the validation of the affected subgraph in G-SPEC (Vijay and Ethiraj 2025, arXiv:2512.20275, Section 3.2): a rule reads the two ends of a link, so no other link can change, and the result is the one of checking the whole system, at a cost that follows the size of the change.

`Action.apply` checks only the preconditions and applies the effects on a copy; `apply_action` also compares the rules before and after. A link whose rule is `None` is never reported by `validate()`.
