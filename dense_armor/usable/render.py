"""render: the strings and the colours of the output."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SHOW_WEAK = False  # weak matches by words are shown only with --weak

TITLES = {
    "classify": "classify the rows",
    "regress": "predict a number",
    "detect_drift": "notice when the data changes",
    "detect_anomaly": "flag strange rows",
    "forecast": "follow and forecast the signal",
    "audio_features": "read the sound",
    "robot": "model and protect the robot",
    "finance_controls": "check orders and keep the log",
    "summarize_stream": "summarise the stream",
    "text_features": "read the text",
    "vision_features": "read the pictures or the video",
    "resources": "watch memory and hardware",
    "heal": "heal and mitigate the noise of the signal",
}

LABEL_NAMES = {
    "y",
    "label",
    "target",
    "class",
    "is_fraud",
    "fraud",
    "anomaly",
    "outlier",
    "attack",
    "default",
}

TIME_NAMES = re.compile(r"^(t|ts|time|timestamp|date|datetime|seconds|sec)$", re.I)

ROBOT_NAMES = re.compile(
    r"(joint|^q\d|^qd\d|tau|torque|velocity|position|effort)", re.I
)

FINANCE_NAMES = re.compile(
    r"(price|order|qty|quantity|notional|side|symbol|ticker|volume|amount)", re.I
)

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def paint(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def _enable_colors() -> None:
    if _COLOR and os.name == "nt":
        os.system("")


TEXT_TITLES = {
    "detect_anomaly": "flag unusual sentences",
    "detect_drift": "notice when the subject changes",
    "text_features": "read the text",
}


def job_title(need: str, pr: dict) -> str:
    return (
        TEXT_TITLES.get(need, TITLES[need])
        if pr.get("kind") == "text"
        else TITLES[need]
    )


def evidence(need: str, pr: dict, budget) -> str:
    """The fact of the file that asks for this job."""
    kind = pr.get("kind")
    if pr.get("evoked", {}).get(need):
        return "the words say it: " + pr["evoked"][need]
    if kind == "code":
        found = pr["signals"].get(need, [])
        return "the code uses " + ", ".join(found[:3]) if found else ""
    if kind == "text":
        if need == "text_features":
            return f"{pr['words']} words to turn into numbers"
        if need in ("detect_anomaly", "detect_drift"):
            return (
                f"the {pr['sentences']} sentences, one after the other, are the stream"
            )
    if kind == "sound":
        return f"a sound of {pr['seconds']:.1f} s at {pr['rate']} Hz"
    if kind == "vision":
        return "a picture or video: frames are the rows"
    cols = pr.get("columns", [])
    if need == "classify":
        lab = [c for c in cols if c["name"].lower() in LABEL_NAMES or c is cols[-1]]
        return f"label column '{lab[0]['name']}'" if lab else "a label column"
    if need == "regress":
        return "a curve: the second column follows the first"
    if need == "detect_drift":
        return f"{pr['rows']} rows in order" + (
            f", one every {budget * 1e3:g} ms" if budget else ""
        )
    if need == "detect_anomaly":
        feats = [
            c
            for c in cols
            if c["numeric"]
            and c["unique"] > 20
            and not (c["monotonic"] and TIME_NAMES.match(c["name"]))
        ]
        return f"{len(feats)} numeric columns to score row by row"
    if need == "forecast":
        return "a signal in time" if budget else "a single numeric series"
    return ""


def _called_classes(d: dict, pr: dict, names: set[str]) -> list[str]:
    """The classes of the library that a definition calls by name: the evidence is in the code.

    A name defined in the same file, or imported from outside Dense-Armor, is not the library's class.
    """
    own = {x["name"] for x in pr["defs"]}
    foreign = {
        i.rsplit(".", 1)[-1] for i in pr["imports"] if not i.startswith("dense_armor")
    }
    out: list[str] = []
    for c in d["calls"]:
        last = c.rsplit(".", 1)[-1]
        if (
            last in names
            and last not in own
            and last not in foreign
            and last not in out
        ):
            out.append(last)
    return out


def print_code_analysis(pr: dict, cat: list[dict], repo: Path) -> None:
    """1. what is in the file  2. what Dense-Armor has  3. the classes whose words meet each definition."""
    from . import concepts, retrieval

    defs = pr["defs"]
    idx = retrieval.Index(cat)
    names = {r.get("class", r["name"]) for r in cat}
    domain_of = {r.get("class", r["name"]): r["domain"] for r in cat}
    answers: list[tuple[dict, list, list, list]] = []
    for d in defs:
        if d.get("data_only"):
            answers.append((d, [], [], []))
            continue
        terms = retrieval.terms_of(d["name"], d["doc"])
        reach = concepts.explain(repo, terms)[:4]
        found = [] if (reach or not SHOW_WEAK) else idx.search(terms, {}, top=3)
        answers.append((d, _called_classes(d, pr, names), reach, found))
    if not any(called or reach or found for _d, called, reach, found in answers):
        hint = (
            ""
            if SHOW_WEAK
            else " (weak matches by words are hidden: type weak, or use --weak)"
        )
        print(
            f"  none of the {len(defs)} definitions of this file matches a class of the library{hint}"
        )
        return

    print("\n" + paint("1. THE FILE", "1;36"))
    for group, names in pr["import_groups"].items():
        if names:
            print(f"   imports, {group}: {', '.join(names)}")
    print(
        f"   definitions: {sum(d['kind'] == 'function' for d in defs)} functions, {sum(d['kind'] == 'class' for d in defs)} classes"
    )
    for d in defs:
        sig = (
            f"{d['name']}({', '.join(d['args'])})"
            if d["kind"] == "function"
            else f"class {d['name']}"
        )
        info = d["doc"] or ("calls " + ", ".join(d["calls"][:4]) if d["calls"] else "")
        print(f"     {d['line']:>5}  {sig[:60]:60s} {paint(info[:70], '2')}")
    domains: dict[str, int] = {}
    for r in cat:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    top = sorted(domains.items(), key=lambda kv: -kv[1])[:10]
    print("\n" + paint("2. DENSE-ARMOR", "1;36"))
    print(
        f"   {len(cat)} classes in {len(domains)} domains; the largest: "
        + ", ".join(f"{d} {n}" for d, n in top)
        + "   (all of them: usable list)"
    )
    print(
        "\n"
        + paint("3. ASK THE LIBRARY", "1;36")
        + paint(
            "   every definition is a question; the answer is the classes whose name, description and methods match it",
            "2",
        )
    )
    for d, called, reach, found in answers:
        sig = (
            f"{d['name']}({', '.join(d['args'])})"
            if d["kind"] == "function"
            else d["name"]
        )
        if d.get("data_only"):
            print(
                f"   {sig[:44]:44s} {paint('only declares data or checks it: no question', '2')}"
            )
            continue
        if called:
            print(f"   {sig[:44]}   {paint('by the code: it calls', '2')}")
            for cls in called:
                print(f"        {cls:34s} {paint(domain_of.get(cls, ''), '2')}")
        if reach:
            total = sum(g for g, _c, _p in reach) or 1.0
            print(
                f"   {('' if called else sig[:44])}   {paint('by the ontology' if not called else 'and by the ontology', '2')}"
            )
            for g, cls, path in reach:
                print(f"        {g / total * 100:5.1f}%  {cls:34s} {paint(path, '2')}")
        if called or reach:
            continue
        if not found:
            print(f"   {sig[:44]:44s} {paint('no class answers', '2')}")
            continue
        print(f"   {sig[:44]}   {paint('weak match by words, not a probability', '2')}")
        for r in found:
            print(
                f"        {r['name']:34s} {r['domain']:11s} {paint(', '.join(r['matched'][:4]), '2')}"
            )


def print_text_answer(pr: dict, cat: list[dict], repo: Path) -> None:
    """A text is one question: its most frequent words."""
    from . import concepts, retrieval

    terms = retrieval.terms_from_keywords(pr["keywords"], pr.get("lang", "en"))
    links = {cls: g for g, cls, _p in concepts.explain(repo, terms)}
    found = retrieval.Index(cat).search(terms, links, top=6)
    print(
        "\n"
        + paint("ASK THE LIBRARY", "1;36")
        + paint("   the question: " + ", ".join(list(terms)[:10]), "2")
    )
    for r in found:
        via = ", ".join(r["matched"][:4]) + (" + ontology link" if r["link"] else "")
        print(
            f"    {r['p'] * 100:5.1f}%  {r['name']:34s} {r['domain']:11s} {paint(via, '2')}"
        )
    if not found:
        print("    no class answers this question")
