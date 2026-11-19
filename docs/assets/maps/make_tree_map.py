"""The interactive map of the trees: jobs, pages and classes, with the exact names of `usable`.

Draws docs/assets/maps/trees.svg for the first page of the Trees section. The jobs on the left
(from dense_armor/usable/ontology.json) and the button at the top lead back to the map of the library on
the home page; every page card and every class opens the page that documents it.

Run from the repo root:
    python docs/assets/maps/make_tree_map.py
"""

import json
from html import escape
from pathlib import Path

from dense_armor.usable.catalog import get_catalog

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
REPO = DOCS.parent
OUT = HERE / "trees.svg"

BG, CARD, TEXT, MUTED = "#0d131a", "#101a24", "#e0fbfc", "#94a3b8"
CYAN, ORANGE, PINK = "#4cc9f0", "#ff9e00", "#f72585"
HOME = "../../#the-library-folder-by-folder"

# page card: (title, subtitle, page, colour, classes in the order shown)
PAGES = [
    (
        "Hoeffding tree",
        "waits for proof",
        "tree/hoeffding.md",
        CYAN,
        [
            "HoeffdingTreeClassifier",
            "HoeffdingTreeRegressor",
            "NumericObserver",
            "NominalObserver",
            "TargetNumericObserver",
            "TargetNominalObserver",
        ],
    ),
    (
        "Anytime tree",
        "changes a split",
        "tree/efdt.md",
        ORANGE,
        ["HoeffdingAnytimeTreeClassifier"],
    ),
    (
        "Adaptive tree",
        "follows a change",
        "tree/adaptive.md",
        PINK,
        ["HoeffdingAdaptiveTreeClassifier"],
    ),
    (
        "Stochastic gradient trees",
        "one rule, any loss",
        "tree/sgt.md",
        CYAN,
        ["SGTRegressor", "SGTClassifier"],
    ),
    (
        "Mondrian forests",
        "a number and its doubt",
        "tree/mondrian.md",
        ORANGE,
        ["MondrianForestRegressor", "MondrianForestClassifier"],
    ),
    (
        "Whole table",
        "trees on all the data",
        "ml/tree.md",
        "#64748b",
        ["BatchDecisionTreeClassifier"],
    ),
    (
        "Model tree",
        "FIMT-DD, in the API",
        "api/general.md#fimtddregressor",
        "#64748b",
        ["FIMTDDRegressor"],
    ),
    (
        "Forests",
        "built from these trees",
        "forest/index.md",
        PINK,
        [
            "AdaptiveRandomForestClassifier",
            "SubspaceHoeffdingTree",
            "OnlineExtraTreesRegressor",
            "AggregatedMondrianForestClassifier",
            "AggregatedMondrianForestRegressor",
        ],
    ),
]


def url(rel):
    page, _, anchor = rel.partition("#")
    path = page.removesuffix("index.md").removesuffix(".md")
    return (
        "../../"
        + path
        + ("" if page.endswith("index.md") else "/")
        + (f"#{anchor}" if anchor else "")
    )


def draw(
    pages, domains, title, subtitle, out, aria, tags, legend, script, partial=False
):
    """Draw one map: the jobs of the ontology that reach domains[0], the pages and their classes.

    With partial=True the map is of one section only, so the classes of the domain that are not
    on its pages are not an error.
    """
    hub_label = domains[0]
    catalog = {
        i["class"]: i
        for i in get_catalog(REPO)
        if i.get("kind", "class") == "class" and i["domain"] in domains
    }
    placed = [c for p in pages for c in p[4]]
    missing = [c for c in placed if c not in catalog]
    assert not missing, f"not in the catalog of usable: {missing}"
    main_domain = {c for c, i in catalog.items() if i["domain"] == hub_label}
    assert partial or main_domain <= set(placed), (
        f"classes of the {hub_label} domain left out: {main_domain - set(placed)}"
    )
    for _, _, rel, _, _ in pages:
        assert (DOCS / rel.partition("#")[0]).exists(), rel

    ontology = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = [j for j, v in ontology["jobs"].items() if hub_label in v["domains"]]
    assert jobs, f"no job of the ontology reaches the {hub_label} domain"

    row, gap, top = 25.0, 14.0, 112.0
    first_mid = top + row * len(pages[0][4]) / 2
    last_start = top + sum(row * len(q[4]) + gap for q in pages[:-1])
    mid0 = (first_mid + last_start + row * len(pages[-1][4]) / 2) / 2
    top += max(0.0, 100.0 + (len(jobs) - 1) * 28 + 24 - mid0)
    y = top
    groups = []
    for ptitle, sub, rel, col, classes in pages:
        ys = [y + row * (k + 0.5) for k in range(len(classes))]
        groups.append((ptitle, sub, rel, col, classes, ys, sum(ys) / len(ys)))
        y += row * len(classes) + gap
    H = y + 44
    W = 1000.0
    mid = (groups[0][6] + groups[-1][6]) / 2
    H = max(H, mid + (len(jobs) - 1) * 28 + 64)
    hub = (330.0, mid)
    jy = {j: mid + (k - (len(jobs) - 1) / 2) * 56 for k, j in enumerate(jobs)}

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="{aria}">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.32;transition:opacity .15s}.net:hover .e{opacity:.06}.net .pg:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}.back:hover .card{fill:#17303f}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">{escape(title)}</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">{escape(subtitle)}</text>',
        f'<a class="back" href="{HOME}" target="_top"><title>back to the map of the library</title>'
        f'<rect class="card" x="20" y="14" width="150" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="95" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← map of the library</text></a>',
        f'<text x="24" y="{98:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  JOB</text>',
        f'<text x="410" y="{98:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  PAGE</text>',
        f'<text x="680" y="{98:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  CLASSES OF THE CATALOG</text>',
        '<g class="net">',
    ]
    for j in jobs:
        o.append(
            f'<path class="e" d="M214,{jy[j]:.1f} C262,{jy[j]:.1f} 282,{hub[1]:.1f} {hub[0] - 12:.0f},{hub[1]:.1f}" fill="none" stroke="{CYAN}" stroke-width="1.4"/>'
        )
    for ptitle, sub, rel, col, classes, ys, cy in groups:
        o.append('<g class="pg">')
        o.append(
            f'<path class="e" d="M{hub[0] + 12:.0f},{hub[1]:.1f} C{hub[0] + 52:.0f},{hub[1]:.1f} 366,{cy:.1f} 408,{cy:.1f}" fill="none" stroke="{col}" stroke-width="1.4"/>'
        )
        for cy2 in ys:
            o.append(
                f'<path class="e" d="M626,{cy:.1f} C650,{cy:.1f} 650,{cy2:.1f} 676,{cy2:.1f}" fill="none" stroke="{col}" stroke-width="1.2"/>'
            )
        o.append("</g>")
    o.append("</g>")

    for j in jobs:
        o.append(
            f'<a href="{HOME}" target="_top"><title>{escape(j)}: back to the map of the library</title>'
            f'<rect class="card" x="20" y="{jy[j] - 19:.1f}" width="190" height="38" rx="10" fill="{CARD}" stroke="{CYAN}" stroke-width="1.6"/>'
            f'<text class="dn m" x="34" y="{jy[j] - 2:.1f}" fill="{TEXT}" font-size="13" font-weight="700">{escape(j)}</text>'
            f'<text x="34" y="{jy[j] + 12:.1f}" fill="{MUTED}" font-size="9.5">job of the ontology</text></a>'
        )
    o.append(
        f'<circle cx="{hub[0]:.0f}" cy="{hub[1]:.1f}" r="12" fill="{CARD}" stroke="{CYAN}" stroke-width="2"/>'
        f'<text class="m" x="{hub[0]:.0f}" y="{hub[1] + 30:.1f}" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">{escape(hub_label)}</text>'
    )
    for ptitle, sub, rel, col, classes, ys, cy in groups:
        o.append(
            f'<a href="{url(rel)}" target="_top"><title>{escape(ptitle)}: open {escape(rel)}</title>'
            f'<rect class="card" x="410" y="{cy - 19:.1f}" width="216" height="38" rx="10" fill="{CARD}" stroke="{col}" stroke-width="1.6"/>'
            f'<text class="dn" x="424" y="{cy - 3:.1f}" fill="{TEXT}" font-size="13" font-weight="700">{escape(ptitle)}</text>'
            f'<text x="424" y="{cy + 12:.1f}" fill="{MUTED}" font-size="9.5">{escape(sub)}</text></a>'
        )
        for name, cy2 in zip(classes, ys):
            tag = tags.get(catalog[name]["role"], "")
            o.append(
                f'<a href="{url(rel)}" target="_top"><title>{escape(name)}: open {escape(rel)}</title>'
                f'<rect class="card" x="678" y="{cy2 - 10:.1f}" width="300" height="20" rx="10" fill="{CARD}" stroke="{col}" stroke-width="1.2"/>'
                f'<text class="dn m" x="692" y="{cy2 + 3.8:.1f}" fill="{TEXT}" font-size="11">{escape(name)}</text>'
                + (
                    f'<text x="966" y="{cy2 + 3.8:.1f}" fill="{CYAN if tag == "C" else ORANGE}" font-size="10" font-weight="700" text-anchor="end">{tag}</text>'
                    if tag
                    else ""
                )
                + "</a>"
            )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 16:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">{legend}jobs: dense_armor/usable/ontology.json; classes: dense_armor.usable.catalog; drawn by {script}</text>'
    )
    o.append("</svg>")
    out.write_text("\n".join(o), encoding="utf-8")
    print(
        "wrote",
        out.relative_to(REPO),
        f"{len(placed)} classes, {len(pages)} pages, jobs {jobs}",
    )


def main():
    draw(
        PAGES,
        ("tree", "forest"),
        "THE MAP OF THE TREES",
        "the job, the page of each tree and its classes; pass over a page to light its classes, click to open it",
        OUT,
        "The map of the trees",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_tree_map.py",
    )


if __name__ == "__main__":
    main()
