"""The interactive map of the Roles section: the pages of the tab and the roles and classes each one teaches.

Draws docs/assets/maps/roles.svg with the same drawing as the other section maps (make_tree_map.py). The pages and their
order are the list "Where to go next" of docs/roles/index.md; the names on each card are the public names of
``dense_armor.roles`` that the page writes between backticks, the most used first. The last card is the page of the robot
layer, which the index links to.

Run from the repo root:
    python docs/assets/maps/make_roles_map.py
"""

import re
from collections import Counter

from make_tree_map import CYAN, DOCS, HERE, ORANGE, PINK, draw

import dense_armor.roles as roles

COLORS = [CYAN, ORANGE, PINK]
NAMES = set(roles.__all__)


def short(names, limit=40):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def pages_of_index():
    out = []
    text = (DOCS / "roles" / "index.md").read_text(encoding="utf-8")
    for m in re.finditer(r"(?m)^\d\. \[([^\]]+)\]\((\w+\.md)\): (.+)$", text):
        title, rel, what = m.groups()
        out.append((title.replace("`", ""), f"roles/{rel}", what.strip().rstrip(".")))
    out.append(("Robot layer", "api/robot_roles.md", "signals, real-time budgets, safety, physical units"))
    return out


def names_in(rel):
    text = (DOCS / rel).read_text(encoding="utf-8")
    code = "\n".join(re.findall(r"```python\n(.*?)```", text, flags=re.S))
    tokens = re.findall(r"`([A-Z][A-Za-z0-9]+)(?:\(|`)", text) + re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", code)
    return [n for n, _ in Counter(t for t in tokens if t in NAMES).most_common()]


def main():
    pages, summaries = [], {}
    for k, (title, rel, what) in enumerate(pages_of_index()):
        sub = what if len(what) <= 38 else what[:35].rstrip() + "..."
        pages.append((title, sub, rel, COLORS[k % 3], [title]))
        names = names_in(rel)
        summaries[title] = f"{len(names)} name{'' if len(names) == 1 else 's'}: {short(names)}" if names else what[:48]
    draw(
        pages,
        ("roles",),
        "THE MAP OF THE ROLES",
        "the job, the page of each part of the roles tab and the roles and classes it teaches; pass over a page to light it, click to open it",
        HERE / "roles.svg",
        "The map of the roles",
        {},
        "",
        "docs/assets/maps/make_roles_map.py",
        partial=True,
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
