"""The interactive maps of Statistics, Preprocessing and Vision, the three big folders of ``core/signals``.

For each folder the pages are those of its group in the menu (``mkdocs.yml``); a class goes to the page that
names it most often. A page with more than four classes shows them on one line. The jobs are those of the
ontology that reach the domain of the folder.

Run from the repo root:
    python docs/assets/maps/make_signals_sections_maps.py
"""

import re

import yaml
from make_tree_map import CYAN, DOCS, HERE, ORANGE, PINK, REPO, draw

from dense_armor.usable.catalog import get_catalog

SECTIONS = {
    "stats": ("Statistics", "THE MAP OF STATISTICS"),
    "preprocessing": ("Preprocessing", "THE MAP OF PREPROCESSING"),
    "vision": ("Video", "THE MAP OF VIDEO"),
}
DEPRECATED = {"Metro"}
FUNCTIONS = {"Stable variance in float32": "stable_var, stable_moments"}


def short(names, limit=36):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def first_page(value):
    """The page of a menu entry: the path itself, or the first page of a nested section."""
    if isinstance(value, str):
        return value
    head = value[0]
    return head if isinstance(head, str) else first_page(next(iter(head.values())))


def group_pages(nav, name):
    """[(title, path)] of the group ``name`` (inside Signals, or a top-level group), without its first page."""
    signals = next(i["Signals"] for i in nav if isinstance(i, dict) and "Signals" in i)
    group = next((i[name] for i in signals if isinstance(i, dict) and name in i), None)
    if group is None:
        group = next(i[name] for i in nav if isinstance(i, dict) and name in i)
    return [(next(iter(i)), first_page(next(iter(i.values())))) for i in group if isinstance(i, dict)]


def main():
    nav = yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))["nav"]
    catalog = [i for i in get_catalog(REPO) if i.get("kind", "class") == "class"]
    colours = [CYAN, ORANGE, PINK]
    for folder, (group, title) in SECTIONS.items():
        classes = [
            i["class"]
            for i in catalog
            if i["module"].startswith(f"dense_armor.core.signals.{folder}.")
        ]
        pages = group_pages(nav, group)
        texts = {p: (DOCS / p).read_text(encoding="utf-8") for _, p in pages}
        owner = {}
        for c in classes:
            module = next(i["module"] for i in catalog if i["class"] == c)
            block = re.compile(r"^::: " + re.escape(module) + r"[ \t]*\r?$", re.M)
            counts = {
                p: len(re.findall(r"\b" + re.escape(c) + r"\b", t))
                + (1 if block.search(t) else 0)
                for p, t in texts.items()
            }
            best = max(counts, key=counts.get)
            if counts[best]:
                owner[c] = best
        lost = [c for c in classes if c not in owner and c not in DEPRECATED]
        print(folder, "classes without a page of the group:", lost)
        rows, summaries = [], {}
        for k, (ptitle, path) in enumerate(pages):
            mine = [c for c in classes if owner.get(c) == path]
            if not mine and ptitle not in FUNCTIONS:
                continue
            if len(mine) > 4:
                summaries[ptitle] = f"{len(mine)} classes: {short(mine)}"
            elif not mine:
                summaries[ptitle] = f"functions: {FUNCTIONS[ptitle]}"
            rows.append(
                (
                    ptitle,
                    (f"{len(mine)} class" + ("" if len(mine) == 1 else "es"))
                    if mine
                    else "functions",
                    path,
                    colours[k % 3],
                    mine,
                )
            )
        if lost:
            title_more = "Classes documented on other pages"
            rows.append(
                (
                    title_more,
                    f"{len(lost)} more",
                    f"{folder}/index.md#classes-documented-on-other-pages",
                    colours[len(rows) % 3],
                    lost,
                )
            )
            if len(lost) > 4:
                summaries[title_more] = f"{len(lost)} classes: {short(lost)}"
        domains = {i["domain"] for i in catalog if i["class"] in classes}
        draw(
            rows,
            tuple(sorted(domains))
            if folder not in domains
            else (folder, *sorted(domains - {folder})),
            title,
            "the job, the page of each tool and its classes; pass over a page to light it, click to open it",
            HERE / f"{folder}.svg",
            title.capitalize(),
            {"Classifier": "C", "Regressor": "R"},
            "C classifier, R regressor; ",
            "docs/assets/maps/make_signals_sections_maps.py",
            partial=True,
            summaries=summaries,
        )


if __name__ == "__main__":
    main()
