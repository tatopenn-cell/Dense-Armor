"""select: the rules that pick the classes for a job."""

from __future__ import annotations

import json
from pathlib import Path

from .catalog import get_catalog
from .render import (
    TITLES,
    evidence,
    job_title,
    paint,
    print_code_analysis,
    print_text_answer,
)


def _load_jobs() -> dict:
    """The jobs and the domains/roles/modules each one accepts, from ``ontology.json``.

    The data lives in ``ontology.json`` (``jobs`` key), not in this file:
    item E of ``ISTRUZIONI.md`` says the data is in the JSON, never in the
    code. If the key is missing, an empty dict is returned (every job
    accepts nothing, and the caller sees it as a data bug, not a crash).
    """
    import json as _json
    from pathlib import Path as _Path

    here = _Path(__file__).resolve().parent
    data = _json.loads((here / "ontology.json").read_text(encoding="utf-8"))
    return data.get("jobs", {})


_JOBS = _load_jobs()
NEED_ROLE = {k: v.get("roles", []) for k, v in _JOBS.items()}
NEED_DOMAIN = {k: v.get("domains", []) for k, v in _JOBS.items()}
NEED_MODULE = {k: v.get("modules", []) for k, v in _JOBS.items() if v.get("modules")}


def rank(
    catalog: list[dict], needs: list[str], budget: float | None, sp: dict | None = None
) -> dict:
    """For each job, every usable class with its probability p, which comes from the graph only.

    weight = strength of the edge job -> class (role and domain together 1, one of them 0.5)
             x (1 + the mass that the concepts of the file bring to the words of the class), kept in [0.2, 3]
    p      = weight / sum of the weights of the usable classes of the job
    The opposite concepts have a negative mass and lower the classes they name."""

    out = {}
    for need in needs:
        roles, domains = NEED_ROLE.get(need, []), NEED_DOMAIN.get(need, [])
        rows = []
        for r in catalog:
            role_hit = bool(roles) and r["role"] in roles
            dom_hit = r["domain"] in domains
            mod_hit = any(m in r["module"] for m in NEED_MODULE.get(need, []))
            if not (role_hit or dom_hit or mod_hit):
                continue
            if roles and r["role"] and r["role"] not in roles:
                continue  # a class with another role does another job
            margin = budget / r["budget_s"] if budget and r["budget_s"] else None
            strength = (
                1.0 if (role_hit and dom_hit) else 0.5 if (role_hit or dom_hit) else 0.5
            )
            tight = margin is not None and margin < 10
            weight = strength
            hits: list[str] = []
            if sp:
                from . import concepts

                lift, hits = concepts.boost(r, sp)
                weight *= max(0.2, min(3.0, 1.0 + lift))
            facts = []
            if role_hit:
                facts.append(f"role {r['role']}")
            if dom_hit:
                facts.append(f"domain {r['domain']}")
            if mod_hit:
                facts.append("module " + r["module"].removeprefix("dense_armor."))
            if hits:
                facts.append("graph: " + " | ".join(hits[:2]))
            if margin is not None:
                facts.append(
                    f"budget {r['budget_s'] * 1e3:g} ms, fits {margin:.0f}x"
                    + (" (tight)" if tight else "")
                )
            elif r["budget_s"] is None:
                facts.append("budget not declared")
            rows.append({**r, "why": facts, "margin": margin, "weight": weight})
        total = sum(r["weight"] for r in rows) or 1.0
        for r in rows:
            r["p"] = r["weight"] / total
        rows.sort(key=lambda r: (-r["p"], r["name"]))
        out[need] = rows
    return out


def snippet(r: dict, need: str) -> str:
    """Code to start, built from the methods the class has and the arguments its constructor requires."""
    cls = r.get("class", r["name"])
    args = ", ".join(f"{a}=..." for a in r["required"])
    have = set(r.get("methods", []))
    use = []
    if r["role"] == "DriftDetector" and "update" in have:
        use += ["m.update(value)   # one number per step", "m.drift_detected"]
    elif r["supervised"] and "learn_one" in have:
        use.append("m.learn_one(x, y)   # y: the label of the row")
        if "predict_one" in have:
            use.append("y_hat = m.predict_one(x)")
    else:
        if "score_one" in have:
            use.append("s = m.score_one(x)")
        if "learn_one" in have:
            use.append("m.learn_one(x)")
        if "predict_one" in have:
            use.append("y_hat = m.predict_one(x)")
    if "transform_one" in have:
        use.append("z = m.transform_one(x)")
    if not use:
        use = (
            ["# public methods: " + ", ".join(sorted(have)[:8])]
            if have
            else ["# no public method found in the source"]
        )
    return (
        f"from {r['module']} import {cls}\n    m = {cls}({args})\n    "
        + "\n    ".join(use)
    )


NEEDS = list(TITLES)


def cmd_select(
    repo: Path,
    needs: list[str],
    budget: float | None,
    profile: dict,
    top=3,
    as_json=False,
    why_not=None,
    notes=None,
    title="",
) -> int:
    """Phase 2: the primitives the ontology accepts for these needs, best first, with the code to start."""
    from .onto_select import select

    sp = None
    topic_jobs: list[str] = []
    if profile.get("keywords") and profile.get("kind") in ("text", "code", "table"):
        from . import concepts

        graph = concepts.build(repo, profile["keywords"], profile.get("lang", "en"))
        sp = concepts.spread(graph)
        evoked = {
            j: v
            for j, v in sp["jobs"].items()
            if v[0] >= concepts.JOB_THRESHOLD and j in TITLES
        }
        topic_jobs = [j for j in evoked if j not in needs]
        needs = needs + topic_jobs
        profile = {**profile, "evoked": {j: v[1] for j, v in evoked.items()}}
    if not needs:
        if title:
            print(title)
        for n in notes or []:
            print("  - " + n)
        if profile.get("kind") == "code" and profile.get("defs") is not None:
            print_code_analysis(profile, get_catalog(repo), repo)
        print("  no job for the primitives of the library was found in this file")
        return 0
    cat = get_catalog(repo)
    accepted, broken = select(
        repo,
        cat,
        profile,
        [n for n in needs if n not in topic_jobs],
        budget,
        NEED_ROLE,
        NEED_DOMAIN,
        NEED_MODULE,
    )
    if topic_jobs:
        # the file speaks of these jobs: it is not the data the classes would read, so the rule on the kind of input does not apply
        more, _ = select(
            repo,
            cat,
            {**profile, "kind": "code", "has_label": True},
            topic_jobs,
            budget,
            NEED_ROLE,
            NEED_DOMAIN,
            NEED_MODULE,
        )
        have = {r["name"] for r in accepted}
        accepted = accepted + [r for r in more if r["name"] not in have]
        broken = {k: v for k, v in broken.items() if k not in {r["name"] for r in more}}
    if why_not:
        if why_not not in {r["name"] for r in cat}:
            print(f"{why_not}: not in the catalog")
        elif why_not not in broken:
            print(f"{why_not}: accepted, no rule is broken")
        else:
            print(f"{why_not}: rejected")
            for rule in broken[why_not]:
                print("  - " + rule)
        return 0
    ranked = rank(accepted, needs, budget, sp)
    shown = {n: rows[:top] for n, rows in ranked.items()}
    if as_json:
        print(
            json.dumps(
                {
                    "title": title,
                    "needs": needs,
                    "budget_s": budget,
                    "notes": notes or [],
                    "primitives": shown,
                },
                indent=1,
            )
        )
        return 0
    if title:
        print(title)
    for n in notes or []:
        print("  - " + n)
    if budget:
        print(f"time per row available: {budget * 1e3:g} ms")
    kinds: dict[str, int] = {}
    for rules in broken.values():
        for rule in rules:
            kinds[rule.split(":")[0]] = kinds.get(rule.split(":")[0], 0) + 1
    kinds_str = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    print(
        f"ontology: {len(cat)} primitives checked, {len(accepted)} accepted, {len(broken)} rejected "
        f"({kinds_str})"
    )
    print(
        "order: role and domain both match, then the margin on the time per row, then the name; nothing is estimated"
    )
    if sp:
        print(
            "\n"
            + paint("== what the words evoke (concept graph)", "1;36")
            + paint("   term -> concept, ~ synonym, != opposite, => job", "2")
        )
        pos = sorted(
            ((v, c) for c, v in sp["mass"].items() if v > 0 and c in sp["known"]),
            reverse=True,
        )[:6]
        neg = sorted((v, c) for c, v in sp["mass"].items() if v < 0)[:3]
        for v, c in pos:
            print(f"    {v:+.2f}  {sp['chain'][c]}")
        for v, c in neg:
            print(f"    {v:+.2f}  {sp['chain'][c]}   (lowers what is named {c})")
        for job, (v, why) in sorted(sp["jobs"].items(), key=lambda kv: -kv[1][0])[:5]:
            if v >= concepts.JOB_THRESHOLD and job in TITLES:
                print(f"    job {job}: {why}")
    print(
        paint(
            "p comes from the graph only: edge job -> class (role + domain = 1, one = 0.5) x (1 + mass of the concepts of the file "
            "that meet the words of the class; opposites lower it). Not a measured probability.",
            "2",
        )
    )
    for need, rows in ranked.items():
        why_file = evidence(need, profile, budget)
        print(
            "\n"
            + paint(f"== {job_title(need, profile)}", "1;36")
            + paint(f"   {len(rows)} usable classes", "1")
            + (paint(f"   because: {why_file}", "2") if why_file else "")
        )
        likely = [r for r in rows if r["p"] >= 0.01][:20]
        for r in likely:
            pct_str = f"{r['p'] * 100:5.1f}%"
            name_str = f"{r['name']:36s}"
            domain = r["domain"]
            why = "; ".join(r["why"])
            print(
                f"  {paint(pct_str, '32')}  {paint(name_str, '1')} {domain:11s} "
                + paint(why, "2")
            )
        rest_rows = [r for r in rows if r not in likely]

        def _strong(r: dict) -> bool:
            kinds = {w.split(" ")[0] for w in r["why"]}
            if not NEED_ROLE.get(need):
                return "module" in kinds or "domain" in kinds
            return "module" in kinds or {"role", "domain"} <= kinds

        strong = [r["name"] for r in rest_rows if _strong(r)]
        weak = [r["name"] for r in rest_rows if not _strong(r)]
        if strong:
            names = strong if top >= 1000 else strong[:40]
            print(
                paint(
                    "  also usable, under 1% each"
                    + (" (role and domain match)" if NEED_ROLE.get(need) else "")
                    + ": "
                    + ", ".join(names)
                    + (
                        f", and {len(strong) - len(names)} more (usable select {need} --top 1000)"
                        if len(names) < len(strong)
                        else ""
                    ),
                    "2",
                )
            )
        if weak and (top >= 1000 or len(weak) <= 15):
            print(
                paint(
                    "  also usable, one of role or domain only: " + ", ".join(weak),
                    "2",
                )
            )
        elif weak:
            print(
                paint(
                    f"  and {len(weak)} more that match only the role or only the domain "
                    f"(all of them: usable select {need} --top 1000)",
                    "2",
                )
            )
        if rows:
            print(paint("    " + snippet(rows[0], need), "2"))
    if profile.get("kind") == "code" and profile.get("defs") is not None:
        print_code_analysis(profile, cat, repo)
    if profile.get("kind") == "text" and profile.get("keywords"):
        if profile.get("evoked"):
            print_text_answer(profile, cat, repo)
        else:
            print(
                "  the words of this text do not call for a task of the library: no class is proposed from them"
            )
    return 0
