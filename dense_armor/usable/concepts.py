"""The concept layer of the selection: terms, concepts and jobs in a TypedGraph, and the spread of the words of a file.

Nodes:  term (a written word, with its language and weight), concept (an idea: filter, bass, anomaly), job.
Edges:  term -expresses-> concept         the glossary takes a word to its English concept
        concept -synonym-> concept         same idea (weight +0.7)
        concept -antonym-> concept         opposite idea (weight -0.5: it lowers what it reaches)
        concept -evokes-> job              a concept that asks for a job of the library
The mass of every term is spread along these edges; a primitive is lifted by the concepts its own words
(name and first line of the description) meet, and lowered by the opposite ones. The graph is the one of
Dense-Armor (TypedGraph), the data is ontology.json and glossary_it.json: both are yours to edit.
"""

import json
from pathlib import Path

from . import keywords as K
from .onto_select import load_ontology

HERE = Path(__file__).resolve().parent
WEIGHT = {
    "expresses": 1.0,
    "synonym": 0.7,
    "antonym": -0.5,
    "relates_primary": 1.0,
    "relates_secondary": 0.5,
}
SCHEMA = {
    "term": {"word": str, "lang": str, "weight": float},
    "concept": {"label": str},
    "job": {"label": str},
    "primitive": {"label": str},
}
JOB_THRESHOLD = 0.25


def _data() -> dict:
    return json.loads((HERE / "ontology.json").read_text(encoding="utf-8"))


def build(repo: Path, keywords: list, lang: str):
    """The graph of this file: the ontology plus one term node for every keyword."""
    onto = load_ontology(repo)
    data = _data()
    g = onto.TypedGraph(SCHEMA)

    def concept(name: str) -> str:
        node = "c:" + name
        if node not in g.graph:
            g.add_node(node, "concept", label=name)
        return node

    for a, b in data["synonyms"]:
        g.add_edge(concept(a), concept(b), "synonym")
        g.add_edge(concept(b), concept(a), "synonym")
    for a, b in data["antonyms"]:
        g.add_edge(concept(a), concept(b), "antonym")
        g.add_edge(concept(b), concept(a), "antonym")
    for job, names in data["evokes"].items():
        g.add_node("j:" + job, "job", label=job)
        for name in names:
            g.add_edge(concept(name), "j:" + job, "evokes")
    for name, targets in data.get("links", {}).items():
        for cls, rank in targets.items():
            node = "p:" + cls
            if node not in g.graph:
                g.add_node(node, "primitive", label=cls)
            g.add_edge(
                concept(name),
                node,
                "relates_primary" if rank >= 0.75 else "relates_secondary",
            )
    glossary = K._glossary() if lang == "it" else {}
    top = max((c for _w, c in keywords), default=1) or 1
    for word, count in keywords:
        if word in K.COMMON:
            continue
        english = K._to_english(word, glossary) if lang == "it" else word
        node = "t:" + word
        g.add_node(node, "term", word=word, lang=lang, weight=count / top)
        g.add_edge(node, concept(english or word), "expresses")
    return g


def spread(graph) -> dict:
    """Mass of every concept, the chain that brought it, and the jobs the positive mass evokes."""
    mass: dict[str, float] = {}
    chain: dict[str, str] = {}

    def add(node, value, path):
        name = graph.graph.nodes[node]["state"]["label"]
        mass[name] = mass.get(name, 0.0) + value
        if name not in chain or abs(value) > 0.0 and len(path) < len(chain[name]):
            chain[name] = path

    for node, attrs in list(graph.graph.nodes.items()):
        if attrs["type"] != "term":
            continue
        state = attrs["state"]
        for _u, c, _k, d in graph.graph.out_edges(node, keys=True, data=True):
            if d["relation"] != "expresses":
                continue
            name = graph.graph.nodes[c]["state"]["label"]
            first = (
                state["word"] if state["word"] == name else f"{state['word']} -> {name}"
            )
            add(c, state["weight"], first)
            for _u2, c2, _k2, d2 in graph.graph.out_edges(c, keys=True, data=True):
                if d2["relation"] in ("synonym", "antonym"):
                    sign = "~" if d2["relation"] == "synonym" else "!="
                    add(
                        c2,
                        WEIGHT[d2["relation"]] * state["weight"],
                        f"{first} {sign} {graph.graph.nodes[c2]['state']['label']}",
                    )
    jobs: dict[str, tuple[float, str]] = {}
    for name, value in mass.items():
        node = "c:" + name
        if value <= 0 or node not in graph.graph:
            continue
        for _u, j, _k, d in graph.graph.out_edges(node, keys=True, data=True):
            if d["relation"] == "evokes":
                job = graph.graph.nodes[j]["state"]["label"]
                total = jobs.get(job, (0.0, ""))[0] + value
                best = (
                    jobs[job][1]
                    if job in jobs and jobs[job][0] >= value
                    else f"{chain[name]} => {job}"
                )
                jobs[job] = (total, best)
    known = {
        name
        for name in mass
        if "c:" + name in graph.graph and graph.graph.out_edges("c:" + name)
    }
    links: dict[str, list] = {}
    for name, value in mass.items():
        node = "c:" + name
        if node not in graph.graph:
            continue
        for _u, pnode, _k, d in graph.graph.out_edges(node, keys=True, data=True):
            if d["relation"].startswith("relates_"):
                cls = graph.graph.nodes[pnode]["state"]["label"]
                links.setdefault(cls, []).append(
                    (value * WEIGHT[d["relation"]], chain[name], name)
                )
    return {"mass": mass, "chain": chain, "jobs": jobs, "known": known, "links": links}


def boost(row: dict, sp: dict) -> tuple[float, list[str]]:
    """How much the concepts of the file lift (or lower) a class: only along the edges concept -> class of the ontology."""
    items = sp["links"].get(row.get("class", row["name"]), [])
    hits = [
        f"{chain}  [{name}: {value:+.2f}]"
        for value, chain, name in sorted(items, key=lambda x: -abs(x[0]))
    ]
    return sum(v for v, _c, _n in items), hits


def explain(repo: Path, words, lang: str = "en") -> list[tuple[float, str, str]]:
    """The classes that the words reach in the graph: (gain, class, the path of the strongest one)."""
    weights = dict(words) if isinstance(words, dict) else {w: 1.0 for w in words}
    for w, v in list(weights.items()):
        weights.setdefault(K._stem(w), v)
    sp = spread(build(repo, [[w, v] for w, v in weights.items()], lang))
    out = []
    for cls, items in sp["links"].items():
        gain = sum(v for v, _c, _n in items)
        if gain > 0:
            best = max(items, key=lambda x: x[0])
            out.append((gain, cls, f"{best[1]} => {cls}"))
    return sorted(out, reverse=True)
