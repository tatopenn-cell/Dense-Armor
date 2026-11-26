"""The interactive map of the Ontology section: the pages of the tab and the tools each one teaches.

Draws docs/assets/maps/ontology.svg with the same drawing as the other section maps (make_tree_map.py). The pages, their
order and their tools are read from the table of docs/ontology/index.md, so the map follows the page. The job is the
one of the ontology of `usable` that reaches the ``graph`` domain.

Run from the repo root:
    python docs/assets/maps/make_ontology_map.py
"""

import re

from make_tree_map import CYAN, DOCS, HERE, ORANGE, PINK, draw

COLORS = [CYAN, ORANGE, PINK]


def short(names, limit=40):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def rows():
    """[(title, file, tools, use)] from the first table of the index, one entry for each file."""
    seen = {}
    for line in (DOCS / "ontology" / "index.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \[([^\]]+)\]\(([\w]+\.md)\) \| (.+?) \| ([^|]+) \|$", line)
        if not m:
            continue
        title, rel, tools, use = m.groups()
        names = re.findall(r"`([^`]+)`", tools)
        if rel in seen:
            for n in names:
                if n not in seen[rel][2]:
                    seen[rel][2].append(n)
        else:
            seen[rel] = [title, rel, names, use.strip()]
    return list(seen.values())


def main():
    pages, summaries = [], {}
    for k, (title, rel, names, use) in enumerate(rows()):
        sub = use if len(use) <= 38 else use[:35].rstrip() + "..."
        pages.append((title, sub, f"ontology/{rel}", COLORS[k % 3], [title]))
        summaries[title] = f"{len(names)} tool{'' if len(names) == 1 else 's'}: {short(names)}"
    draw(
        pages,
        ("graph",),
        "THE MAP OF THE ONTOLOGY",
        "the job, the page of each tool of the graph part and what it teaches; pass over a page to light it, click to open it",
        HERE / "ontology.svg",
        "The map of the ontology",
        {},
        "",
        "docs/assets/maps/make_ontology_map.py",
        partial=True,
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
