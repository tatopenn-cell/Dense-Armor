"""Selection by ontology: which primitives may be used on this file, and why the others may not.

The file and every primitive are nodes of a TypedGraph (the one of Dense-Armor, loaded from its source
so that importing it costs nothing). Each rule is an edge from the file to a primitive with a Constraint;
validate() returns the broken rules. A primitive is accepted when no rule is broken.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path


class _MultiDiGraph:
    """The few methods of networkx.MultiDiGraph that TypedGraph.validate() and add_node/add_edge use.

    Importing networkx costs 0.6 s at every start; this stand-in costs nothing. Set ONTO_REAL_NX=1 to use
    networkx itself.
    """

    class _Nodes(dict):
        pass

    def __init__(self):
        self.nodes = self._Nodes()
        self._edges = []
        self._out = {}
        self._in = {}

    def __contains__(self, name):
        return name in self.nodes

    def add_node(self, name, **attrs):
        self.nodes[name] = dict(attrs)

    def add_edge(self, u, v, **attrs):
        key = len(self._edges)
        self._edges.append((u, v, key, attrs))
        self._out.setdefault(u, []).append((u, v, key, attrs))
        self._in.setdefault(v, []).append((u, v, key, attrs))
        return key

    def edges(self, keys=False, data=False):
        return list(self._edges)

    def in_edges(self, n, keys=False, data=False):
        return list(self._in.get(n, []))

    def out_edges(self, n, keys=False, data=False):
        return list(self._out.get(n, []))

    def copy(self):
        return self


def _use_stand_in():
    if os.environ.get("ONTO_REAL_NX") or "networkx" in sys.modules:
        return
    fake = types.ModuleType("networkx")
    fake.MultiDiGraph = _MultiDiGraph
    sys.modules["networkx"] = fake


SCHEMA = {
    "dataset": {
        "kind": str,
        "has_label": bool,
        "rows": int,
        "budget_s": float,
        "wanted_roles": str,
        "wanted_domains": str,
        "wanted_modules": str,
    },
    "primitive": {
        "role": str,
        "domain": str,
        "module": str,
        "budget_s": float,
        "supervised": bool,
        "consumes": str,
    },
}


# what a primitive can read, for each kind of file: a sound, a text or a picture reaches the others through
# the adapters of the library (AudioInput, TextInput, VideoInput), which give rows or numbers.
READS = {
    "table": ("rows", "scalar"),
    "sound": ("sound", "rows", "scalar"),
    "text": ("text", "rows", "scalar"),
    "vision": ("frames", "rows", "scalar"),
    "code": ("rows", "scalar", "sound", "frames", "text"),
}


def load_ontology(repo: Path):
    """The ontology of the library, loaded from its source.

    Order: (1) ``<repo>/dense_armor/core/graph/_light.py`` (Dense-Armor
    >= 1.6.0, no networkx needed); (2) ``<repo>/dense_armor/core/graph/ontology.py``
    with a stand-in for networkx. Never fails in a cryptic way: a missing
    file names the path looked for and what to do; a file that lacks
    ``TypedGraph`` or ``Constraint`` says so explicitly (library too old).
    """
    graph_dir = repo / "dense_armor" / "core" / "graph"
    if not graph_dir.is_dir():
        raise RuntimeError(
            f"Dense-Armor not found at {repo}: {graph_dir} is not a directory. "
            f"Set USABLE_REPO to the Dense-Armor tree, or install the package "
            f"(see ONTOLOGY_API.md)."
        )
    onto_py = graph_dir / "ontology.py"
    if not onto_py.is_file():
        raise RuntimeError(
            f"the ontology file was not found: {onto_py}. "
            f"Set USABLE_REPO to the Dense-Armor tree, or update the library "
            f"(>= 1.6.0, see ONTOLOGY_API.md)."
        )
    light = graph_dir / "_light.py"
    if light.exists():
        for name in ("dense_armor", "dense_armor.core", "dense_armor.core.graph"):
            if name not in sys.modules:
                stub = types.ModuleType(name)
                stub.__path__ = []
                sys.modules[name] = stub
        spec = importlib.util.spec_from_file_location(
            "dense_armor.core.graph._light", light
        )
        assert spec is not None, spec
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        assert spec.loader is not None, spec
        spec.loader.exec_module(mod)
    else:
        _use_stand_in()
    spec = importlib.util.spec_from_file_location("dense_armor_ontology", onto_py)
    assert spec is not None, spec
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None, spec
    spec.loader.exec_module(mod)
    for name in ("TypedGraph", "Constraint"):
        if not hasattr(mod, name):
            raise RuntimeError(
                f"{onto_py} does not define {name}: the library is too old. "
                f"Update Dense-Armor to a version with {name} in the graph module."
            )
    return mod


def _rules(Constraint):
    return [
        Constraint(
            "budget: the primitive fits the time per row",
            lambda f, p: (
                p["budget_s"] < 0 or f["budget_s"] < 0 or p["budget_s"] <= f["budget_s"]
            ),
        ),
        Constraint(
            "label: the primitive needs a label and the file has none",
            lambda f, p: (not p["supervised"]) or f["has_label"],
        ),
        Constraint(
            "input: the primitive does not read this kind of file",
            lambda f, p: (
                p["consumes"] == "any" or p["consumes"] in READS.get(f["kind"], ())
            ),
        ),
        Constraint(
            "need: nothing the file asks for is done by this primitive",
            lambda f, p: (
                (p["role"] != "" and p["role"] in f["wanted_roles"].split(","))
                or p["domain"] in [d for d in f["wanted_domains"].split(",") if d]
                or any(m and m in p["module"] for m in f["wanted_modules"].split(","))
            ),
        ),
    ]


def select(
    repo: Path,
    catalog: list[dict],
    profile: dict,
    needs: list[str],
    budget: float | None,
    need_role: dict,
    need_domain: dict,
    need_module: dict | None = None,
):
    """Return (accepted rows, {name: [broken rules]})."""
    onto = load_ontology(repo)
    g = onto.TypedGraph(SCHEMA)
    roles = sorted({r for n in needs for r in need_role.get(n, [])})
    domains = sorted({d for n in needs for d in need_domain.get(n, [])})
    modules = sorted({m for n in needs for m in (need_module or {}).get(n, [])})
    has_label = profile.get("has_label", "classify" in needs or "regress" in needs)
    g.add_node(
        "FILE",
        "dataset",
        kind=profile["kind"],
        has_label=has_label,
        rows=int(profile.get("rows", 0)),
        budget_s=float(budget) if budget else -1.0,
        wanted_roles=",".join(roles),
        wanted_domains=",".join(domains),
        wanted_modules=",".join(modules),
    )
    rules = _rules(onto.Constraint)
    by = {}
    for r in catalog:
        by[r["name"]] = r
        g.add_node(
            r["name"],
            "primitive",
            role=r["role"] or "",
            domain=r["domain"],
            module=r["module"],
            budget_s=float(r["budget_s"]) if r["budget_s"] is not None else -1.0,
            supervised=bool(r["supervised"]),
            consumes=r["consumes"],
        )
        for rule in rules:
            g.add_edge("FILE", r["name"], rule.name, rule)
    broken: dict[str, list[str]] = {}
    for _src, dst, relation in g.validate():
        broken.setdefault(dst, []).append(relation)
    accepted = [r for r in catalog if r["name"] not in broken]
    return accepted, broken
