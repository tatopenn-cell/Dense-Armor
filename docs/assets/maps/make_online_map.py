"""The interactive map of the Online learning tab: the groups, the areas and every page, from the menu.

Draws docs/assets/maps/online.svg as a clickable SVG with three columns:

- left: the groups of the page of the tab (predict, detect, decide and suggest, model the stream, and the way in with
  ``fit``), with the number of areas and of classes;
- centre: the areas, which are the sections of the tab; each one opens its index and says how many classes it has in the
  catalog of the library and how many pages it has;
- right: every page of the area, from ``mkdocs.yml``.

The groups and their order are those of docs/assets/online/make_overview_figure.py, the names and the pages are those of
the menu, the counts are those of the catalog (``dense_armor.usable.catalog``). Run it again when the menu changes:

    python docs/assets/maps/make_online_map.py
"""

import sys
from collections import Counter
from html import escape
from pathlib import Path

from make_applications_map import BG, CARD, COLORS, CYAN, HOME, MUTED, ORANGE, PINK, TEXT, count, first, pages_of, url
from make_area_maps import DOCS, REPO, _nav_cfg

sys.path.insert(0, str(DOCS / "assets" / "online"))
from make_overview_figure import GROUPS  # noqa: E402

from dense_armor.usable.catalog import get_catalog  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "online.svg"

# area of the catalog -> the section of the menu that holds it
SECTION = {
    "learn": "Adaptive learners",
    "tree": "Trees",
    "forest": "Forests",
    "ensemble": "Ensembles",
    "linear": "Linear models",
    "bayes": "Classification",
    "neighbors": "Classification",
    "multiclass": "Classification",
    "multioutput": "Classification",
    "anomaly": "Anomaly detection",
    "drift": "Drift detection",
    "protect": "Protection",
    "bandit": "Bandits",
    "active": "Active learning",
    "reco": "Recommenders",
    "facto": "Factorization machines",
    "timeseries": "Time series",
    "proba": "Distributions",
    "optim": "Optimizers",
    "cluster": "Clustering",
    "rules": "Rules",
}
# the way in of the whole dataset is a section of the tab that is not a folder of core/online
EXTRA_GROUP = ("05", "BRING THE DATA", "a whole table, with fit", ORANGE, (("fit", "Fit"),))


def nav_sections():
    """{title: (index page, [(page title, rel)])} of every section of the Online learning tab, flattened."""
    tab = []
    for entry in _nav_cfg():
        if isinstance(entry, dict):
            for name in ("Online learning", "Detection"):
                if name in entry:
                    tab = tab + entry[name]
    out = {}

    def walk(items, title, nested=False):
        pages = []
        for it in items:
            if isinstance(it, str):
                pages.append((title if nested else "Overview", it))
            else:
                for k, v in it.items():
                    if isinstance(v, str):
                        pages.append((k, v))
                    else:
                        pages.extend(walk(v, k, True))
        return pages

    for it in tab:
        if isinstance(it, dict):
            for k, v in it.items():
                if isinstance(v, list):
                    if k == "Detection" or k == "Classification":
                        for sub in v:
                            if isinstance(sub, dict):
                                for kk, vv in sub.items():
                                    out[kk] = (first(vv), walk(vv, kk))
                            elif isinstance(sub, str):
                                out.setdefault(k, (sub, []))[1].append((sub, sub))
                    else:
                        out[k] = (first(v), walk(v, k))
                else:
                    out[k] = (v, [(k, v)])
    return out


def classes():
    c = Counter()
    for i in get_catalog(REPO):
        p = i["module"].split(".")
        if len(p) > 3 and p[1] == "core" and p[2] == "online":
            c[p[3]] += 1
    return c


def main():
    sections = nav_sections()
    n_classes = classes()
    groups = list(GROUPS) + [EXTRA_GROUP]
    laid = []
    row, gap, top = 19.0, 5.0, 112.0
    y = top
    colors = [CYAN, PINK, ORANGE, CYAN, ORANGE]
    for gi, (num, gname, note, color, cards) in enumerate(groups):
        blocks = []
        seen = set()
        for area, label in cards:
            sec = SECTION.get(area, "Fit, the whole dataset at once")
            if area in ("bayes", "neighbors", "multiclass", "multioutput"):
                # four index pages of the menu group "Classification", one block each
                title = {"bayes": "Classification", "neighbors": "Classification", "multiclass": "Classification", "multioutput": "Classification"}[area]
                idx = f"{area}/index.md"
                pages = [(label, idx)]
            else:
                if sec not in sections:
                    raise SystemExit(f"section not in the menu: {sec}")
                idx, pages = sections[sec]
                seen.add(sec)
            block = max(row * len(pages), 36.0)
            off = (block - row * len(pages)) / 2
            ys = [y + off + row * (k + 0.5) for k in range(len(pages))]
            blocks.append((label, idx, area, [(t, r, yy) for (t, r), yy in zip(pages, ys)], sum(ys) / len(ys)))
            y += block + gap
        y += 14
        laid.append((num, gname, color, blocks, note))
    H, W = y + 30, 1000.0
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of online learning">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.35;transition:opacity .15s}.net:hover .e{opacity:.07}.net .s:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP OF ONLINE LEARNING</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">the groups, the areas and every page of the tab; pass over an area to light its pages, click to open</text>',
        f'<a href="{HOME}" target="_top"><title>back to the map of the library</title><rect class="card" x="20" y="14" width="150" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="95" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← map of the library</text></a>',
        f'<text x="24" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  GROUP</text>',
        f'<text x="236" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  AREA</text>',
        f'<text x="500" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  PAGES</text>',
        '<g class="net">',
    ]
    gy = {g[1]: sum(b[4] for b in g[3]) / len(g[3]) for g in laid}
    for num, gname, color, blocks, note in laid:
        for label, idx, area, pages, my in blocks:
            o.append('<g class="s">')
            o.append(f'<path class="e" d="M164,{gy[gname]:.1f} C200,{gy[gname]:.1f} 204,{my:.1f} 238,{my:.1f}" fill="none" stroke="{color}" stroke-width="1.4"/>')
            for _, _, py in pages:
                o.append(f'<path class="e" d="M388,{my:.1f} C430,{my:.1f} 436,{py:.1f} 490,{py:.1f}" fill="none" stroke="{color}" stroke-width="1.1"/>')
            o.append("</g>")
    o.append("</g>")
    n_pages = 0
    n_areas = 0
    for num, gname, color, blocks, note in laid:
        total = sum(len(b[3]) for b in blocks)
        n_pages += total
        n_areas += len(blocks)
        cls = sum(n_classes.get(b[2], 0) for b in blocks)
        gidx = blocks[0][1]
        o.append(
            f'<a href="{url(gidx)}" target="_top"><title>{escape(gname.title())}: {escape(note)}</title>'
            f'<rect class="card" x="20" y="{gy[gname] - 24:.1f}" width="144" height="48" rx="12" fill="{CARD}" stroke="{color}" stroke-width="1.8"/>'
            f'<text class="dn m" x="32" y="{gy[gname] - 4:.1f}" fill="{TEXT}" font-size="{12 if len(gname) <= 16 else 9.4}" font-weight="700">{escape(gname)}</text>'
            f'<text x="32" y="{gy[gname] + 14:.1f}" fill="{MUTED}" font-size="9.5">{count(len(blocks), "area")} · {cls} class{'' if cls == 1 else 'es'}</text></a>'
        )
        for label, idx, area, pages, my in blocks:
            nc = n_classes.get(area, 0)
            fs = 12.5 if len(label) <= 17 else 10.5 if len(label) <= 24 else 9.2
            sub = f"{count(len(pages), 'page')}" + (f" · {nc} class" + ("" if nc == 1 else "es") if nc else "")
            o.append(
                f'<a href="{url(idx)}" target="_top"><title>{escape(label)}: open {escape(idx)}</title>'
                f'<rect class="card" x="238" y="{my - 17:.1f}" width="150" height="34" rx="10" fill="{CARD}" stroke="{color}" stroke-width="1.6"/>'
                f'<text class="dn m" x="250" y="{my - 2:.1f}" fill="{TEXT}" font-size="{fs}" font-weight="700">{escape(label)}</text>'
                f'<text x="250" y="{my + 11:.1f}" fill="{MUTED}" font-size="9.5">{escape(sub)}</text></a>'
            )
            for title, rel, py in pages:
                assert (DOCS / rel.partition("#")[0]).exists(), rel
                o.append(
                    f'<a href="{url(rel)}" target="_top"><title>{escape(title)}: open {escape(rel)}</title>'
                    f'<rect class="card" x="492" y="{py - 8.5:.1f}" width="486" height="17" rx="8.5" fill="{CARD}" stroke="{color}" stroke-width="1.1"/>'
                    f'<text class="dn m" x="504" y="{py + 3.6:.1f}" fill="{TEXT}" font-size="10.5" font-weight="700">{escape(title)}</text>'
                    f'<text x="966" y="{py + 3.5:.1f}" fill="{MUTED}" font-size="9.2" class="m" text-anchor="end">{escape(rel)}</text></a>'
                )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">pages: the menu of mkdocs.yml; classes: the catalog of the library (dense_armor.usable.catalog); drawn by docs/assets/maps/make_online_map.py</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT.relative_to(REPO), f"{n_areas} areas, {n_pages} pages, height {H:.0f}")


if __name__ == "__main__":
    main()
