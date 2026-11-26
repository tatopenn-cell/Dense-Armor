"""The interactive map of the Applications tab: the six applications, their sections and every page, from the menu.

Draws docs/assets/maps/applications.svg as a clickable SVG with three columns:

- left: the six applications (finance, drones, robot, IoT, biomedical, ARC-AGI), with the number of pages and, where the
  folder ``pipelines/`` has them, the number of scripts;
- centre: the sections of an application (the groups of its menu); an application with no groups has one box, ``pages``;
- right: every page of the section, from ``mkdocs.yml``.

The names and the order are those of the menu, so the map follows it. Every box opens a page of the docs. Run it again
when the menu changes:

    python docs/assets/maps/make_applications_map.py
"""

import re
from html import escape
from pathlib import Path

from make_area_maps import DOCS, REPO, _nav_cfg

HERE = Path(__file__).resolve().parent
OUT = HERE / "applications.svg"
BG, CARD, TEXT, MUTED = "#0d131a", "#101a24", "#e0fbfc", "#94a3b8"
CYAN, ORANGE, PINK = "#4cc9f0", "#ff9e00", "#f72585"
COLORS = [ORANGE, CYAN, PINK, ORANGE, CYAN, PINK]
HOME = "../../index.md#the-library-folder-by-folder"


def url(rel):
    page, _, anchor = rel.partition("#")
    path = page.removesuffix("index.md").removesuffix(".md")
    return (
        "../../"
        + path
        + ("" if page.endswith("index.md") else "/")
        + (f"#{anchor}" if anchor else "")
    )


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def first(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return first(x[0])
    return first(next(iter(x.values())))


def pages_of(items):
    """[(title, rel)] of a list of menu items: strings are an index page, dicts are one page or a group."""
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(("Overview", it))
        else:
            for k, v in it.items():
                if isinstance(v, str):
                    out.append((k, v))
    return out


def structure():
    """[(application, index page, [(section, link, [(page, rel)])])]."""
    for entry in _nav_cfg():
        if isinstance(entry, dict) and "Applications" in entry:
            apps = [it for it in entry["Applications"] if isinstance(it, dict)]
    out = []
    for it in apps:
        (name, body), = it.items()
        if name == "Overview":
            continue
        sections, flat = [], []
        for part in body:
            if isinstance(part, dict) and any(isinstance(v, list) for v in part.values()):
                for sec, items in part.items():
                    pg = pages_of(items)
                    link = f"{first(body)}#{slug(sec)}" if name == "Finance" else pg[0][1]
                    sections.append((sec, link, pg))
            else:
                flat.append(part)
        if flat:
            label = "Overview" if len(flat) == 1 and isinstance(flat[0], str) else "pages"
            sections.insert(0, (label, first(flat), pages_of(flat)))
        out.append((name, first(body), sections))
    return out


def scripts():
    base = REPO / "pipelines"
    if not base.is_dir():
        return {}
    return {
        d.name: sum(1 for f in d.glob("*.py") if f.name != "__init__.py")
        for d in base.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    }


def count(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def main():
    apps = structure()
    sc = scripts()
    row, gap, top = 19.0, 5.0, 112.0
    y = top
    laid = []
    for ai, (name, idx, sections) in enumerate(apps):
        blocks = []
        for sec, link, pages in sections:
            block = max(row * len(pages), 36.0)
            off = (block - row * len(pages)) / 2
            ys = [y + off + row * (k + 0.5) for k in range(len(pages))]
            blocks.append((sec, link, [(t, r, yy) for (t, r), yy in zip(pages, ys)], sum(ys) / len(ys)))
            y += block + gap
        y += 14
        laid.append((name, idx, COLORS[ai % len(COLORS)], blocks))
    H, W = y + 30, 1000.0
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of the applications">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.35;transition:opacity .15s}.net:hover .e{opacity:.07}.net .s:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP OF THE APPLICATIONS</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">the six applications, their sections and every page; pass over a section to light its pages, click to open a page</text>',
        f'<a href="{HOME}" target="_top"><title>back to the map of the library</title><rect class="card" x="20" y="14" width="150" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="95" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← map of the library</text></a>',
        f'<text x="24" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  APPLICATION</text>',
        f'<text x="236" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  SECTION</text>',
        f'<text x="500" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  PAGES</text>',
        '<g class="net">',
    ]
    ay = {name: sum(b[3] for b in blocks) / len(blocks) for name, _, _, blocks in laid}
    for name, idx, col, blocks in laid:
        for sec, link, pages, my in blocks:
            o.append('<g class="s">')
            o.append(f'<path class="e" d="M164,{ay[name]:.1f} C200,{ay[name]:.1f} 204,{my:.1f} 238,{my:.1f}" fill="none" stroke="{col}" stroke-width="1.4"/>')
            for _, _, py in pages:
                o.append(f'<path class="e" d="M388,{my:.1f} C430,{my:.1f} 436,{py:.1f} 490,{py:.1f}" fill="none" stroke="{col}" stroke-width="1.1"/>')
            o.append("</g>")
    o.append("</g>")
    n_pages = 0
    for name, idx, col, blocks in laid:
        total = sum(len(b[2]) for b in blocks)
        n_pages += total
        key = {"Finance": "finance", "Drones": "drones", "IoT": "iot", "Biomedical": "biomedical"}.get(name)
        extra = f" · {count(sc[key], 'script')}" if key in sc else ""
        o.append(
            f'<a href="{url(idx)}" target="_top"><title>{escape(name)}: open {escape(idx)}</title>'
            f'<rect class="card" x="20" y="{ay[name] - 24:.1f}" width="144" height="48" rx="12" fill="{CARD}" stroke="{col}" stroke-width="1.8"/>'
            f'<text class="dn m" x="34" y="{ay[name] - 4:.1f}" fill="{TEXT}" font-size="14" font-weight="700">{escape(name)}</text>'
            f'<text x="34" y="{ay[name] + 14:.1f}" fill="{MUTED}" font-size="10">{count(total, "page")}{extra}</text></a>'
        )
        for sec, link, pages, my in blocks:
            fs = 12.5 if len(sec) <= 17 else 10.5 if len(sec) <= 24 else 9.2
            o.append(
                f'<a href="{url(link)}" target="_top"><title>{escape(sec)}: open {escape(link)}</title>'
                f'<rect class="card" x="238" y="{my - 17:.1f}" width="150" height="34" rx="10" fill="{CARD}" stroke="{col}" stroke-width="1.6"/>'
                f'<text class="dn m" x="250" y="{my - 2:.1f}" fill="{TEXT}" font-size="{fs}" font-weight="700">{escape(sec)}</text>'
                f'<text x="250" y="{my + 11:.1f}" fill="{MUTED}" font-size="9.5">{count(len(pages), "page")}</text></a>'
            )
            for title, rel, py in pages:
                assert (DOCS / rel.partition("#")[0]).exists(), rel
                o.append(
                    f'<a href="{url(rel)}" target="_top"><title>{escape(title)}: open {escape(rel)}</title>'
                    f'<rect class="card" x="492" y="{py - 8.5:.1f}" width="486" height="17" rx="8.5" fill="{CARD}" stroke="{col}" stroke-width="1.1"/>'
                    f'<text class="dn m" x="504" y="{py + 3.6:.1f}" fill="{TEXT}" font-size="10.5" font-weight="700">{escape(title)}</text>'
                    f'<text x="966" y="{py + 3.5:.1f}" fill="{MUTED}" font-size="9.2" class="m" text-anchor="end">{escape(rel)}</text></a>'
                )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">pages: the menu of mkdocs.yml; scripts: the folder pipelines/; drawn by docs/assets/maps/make_applications_map.py</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT.relative_to(REPO), f"{len(laid)} applications, {n_pages} pages, height {H:.0f}")


if __name__ == "__main__":
    main()
