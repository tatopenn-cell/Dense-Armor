"""The map of Dense-Armor by job, drawn from `usable` as a clickable SVG.

The jobs, the domains each job accepts and the classes of each domain are those of
``dense_armor.usable`` (ontology.json and the catalog). A domain is a link to its page in the docs
when the page exists; passing the mouse over a job lights its links. Run it again when the catalog
or the ontology changes:

    python docs/assets/maps/make_usable_map.py
"""

import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path

from dense_armor.usable.catalog import get_catalog

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
REPO = DOCS.parent
OUT = HERE / "jobs.svg"

BG = "#0d131a"
CARD = "#101a24"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
PINK = "#f72585"
COLORS = ("#4cc9f0", "#ff9e00", "#f72585")
PAGES = {
    "datasets": "stream/index.md",
    "linear": "linear/index.md",
    "graph": "ontology/index.md",
    "evaluate": "metrics/index.md",
    "neural": "neural/overview.md",
    "kalman": "timeseries/kalman.md",
    "ml": "ml/index.md",
    "roles": "roles/index.md",
    "stats": "stats/index.md",
    "preprocessing": "preprocessing/index.md",
    "metrics": "metrics/index.md",
    "audio": "audio/index.md",
    "vision": "vision/index.md",
    "finance": "finance/index.md",
    "dynamics": "dynamics/index.md",
    "optim": "optim/index.md",
}


# jobs that `usable` still lists but the library no longer recommends
DEPRECATED_JOBS = {"heal"}
# the catalog calls this domain after its folder; its class is Armor
DISPLAY = {"core": "armor"}

_TEXTS = {}
_CLASSES = {}


def jobs_of(onto):
    """The jobs of the ontology without the deprecated ones; a job that lists only modules gets the
    domains of the classes in those modules."""
    jobs = {}
    rows = get_catalog(REPO)
    for name, job in onto["jobs"].items():
        if name in DEPRECATED_JOBS:
            continue
        job = dict(job)
        if not job["domains"] and job["modules"]:
            job["domains"] = sorted(
                {
                    r["domain"]
                    for r in rows
                    for m in job["modules"]
                    if r["module"].endswith("." + m) or r["module"] == m
                }
            )
        jobs[name] = job
    return jobs


def _best_page(domain):
    """The page of the docs that names the classes of the domain most often, written in backticks."""
    if not _TEXTS:
        for f in DOCS.rglob("*.md"):
            if f.name in ("changelog.md", "general.md") or "assets" in f.parts:
                continue
            _TEXTS[f.relative_to(DOCS).as_posix()] = f.read_text(encoding="utf-8")
        for i in get_catalog(REPO):
            _CLASSES.setdefault(i["domain"], []).append(i["class"])
    scores = {}
    for name in _CLASSES.get(domain, []):
        pattern = re.compile("`" + re.escape(name) + "[`(]")
        for rel, text in _TEXTS.items():
            scores[rel] = scores.get(rel, 0) + len(pattern.findall(text))
    best = max(scores.items(), key=lambda kv: kv[1], default=(None, 0))
    return best[0] if best[1] else None


def page_of(domain):
    """The page of a domain: a chosen one, the index of its folder, or the page that names its classes."""
    for rel in (PAGES.get(domain), f"{domain}/index.md"):
        if rel and (DOCS / rel).exists():
            return rel
    return _best_page(domain)


def url_of(rel):
    """The address of a page, relative to docs/assets/maps/jobs.svg (two folders below the docs root)."""
    return "../../" + (
        rel.removesuffix("index.md").removesuffix(".md")
        + ("" if rel.endswith("index.md") else "/")
    )


def short(names, limit=46):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def order_nodes(jobs):
    """Barycentre sweeps: fewer crossings between the jobs and the domains."""
    names = list(jobs)
    domains = sorted({d for j in names for d in jobs[j]["domains"]})
    jpos = {j: k for k, j in enumerate(names)}
    dpos = {}
    for _ in range(12):
        dpos = {
            d: sum(jpos[j] for j in names if d in jobs[j]["domains"])
            / sum(1 for j in names if d in jobs[j]["domains"])
            for d in domains
        }
        domains.sort(key=lambda d: (dpos[d], d))
        rank = {d: k for k, d in enumerate(domains)}
        names.sort(
            key=lambda j: (
                sum(rank[d] for d in jobs[j]["domains"])
                / max(1, len(jobs[j]["domains"])),
                j,
            )
        )
        jpos = {j: k for k, j in enumerate(names)}
    return names, domains


def draw():
    onto = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = jobs_of(onto)
    by_domain = defaultdict(list)
    functions = defaultdict(int)
    for i in get_catalog(REPO):
        if i.get("kind", "class") == "function":
            functions[i["domain"]] += 1
        elif i["class"] not in by_domain[i["domain"]]:
            by_domain[i["domain"]].append(i["class"])
    order, domains = order_nodes(jobs)
    jh, dh = 44.0, 21.0
    n_rows = max(len(order) * jh, len(domains) * dh)
    top = 96.0
    H = top + n_rows + 50
    W = 1000.0
    jy = {
        j: top + jh * (k + 0.5) + (n_rows - len(order) * jh) / 2
        for k, j in enumerate(order)
    }
    dy = {
        d: top + dh * (k + 0.5) + (n_rows - len(domains) * dh) / 2
        for k, d in enumerate(domains)
    }
    color = {j: COLORS[k % 3] for k, j in enumerate(order)}
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of Dense-Armor by job">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.38;transition:opacity .15s}.net:hover .e{opacity:.07}.net .j:hover .e{opacity:1;stroke-width:2.4}"
        ".j{cursor:default}a:hover .pill{fill:#17303f;stroke-width:2.2}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP BY JOB</text>',
        f'<text x="{W / 2:.0f}" y="64" fill="#4cc9f0" font-size="12.5" text-anchor="middle">what you want to do, the domains that serve it and their classes; pass over a job to light its links, click a domain to open its page</text>',
        f'<text x="24" y="{top - 12:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  JOB</text>',
        f'<text x="410" y="{top - 12:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  DOMAIN</text>',
        f'<text x="640" y="{top - 12:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  CLASSES OF THE CATALOG</text>',
        '<g class="net">',
    ]
    for j in order:
        roles = (
            ", ".join(jobs[j]["roles"])
            or ", ".join(jobs[j]["modules"][:1])
            or "by domain"
        )
        o.append('<g class="j">')
        for d in jobs[j]["domains"]:
            x1, y1, x2, y2 = 214, jy[j], 408, dy[d]
            o.append(
                f'<path class="e" d="M{x1},{y1:.1f} C{x1 + 96},{y1:.1f} {x2 - 96},{y2:.1f} {x2},{y2:.1f}" fill="none" stroke="{color[j]}" stroke-width="1.2"/>'
            )
        o.append(f'<circle cx="214" cy="{jy[j]:.1f}" r="4" fill="{color[j]}"/>')
        o.append(
            f'<rect x="20" y="{jy[j] - 17:.1f}" width="190" height="34" rx="9" fill="{CARD}" stroke="{color[j]}" stroke-width="1.6"/>'
        )
        o.append(
            f'<text x="32" y="{jy[j] - 2:.1f}" fill="{TEXT}" font-size="13" font-weight="700" class="m">{escape(j)}</text>'
        )
        o.append(
            f'<text x="32" y="{jy[j] + 11:.1f}" fill="{MUTED}" font-size="9.5">{escape(roles[:34])}</text>'
        )
        o.append("</g>")
    o.append("</g>")
    linked = 0
    for d in domains:
        classes = by_domain.get(d, [])
        rel = page_of(d)
        y = dy[d]
        pill = f'<rect class="pill" x="410" y="{y - 9:.1f}" width="214" height="18" rx="9" fill="{CARD}" stroke="{"#4cc9f0" if rel else "#334155"}" stroke-width="1.3"/>'
        body = (
            f'<circle cx="410" cy="{y:.1f}" r="3.2" fill="#4cc9f0"/>{pill}'
            f'<text class="dn m" x="422" y="{y + 3.6:.1f}" fill="{TEXT if rel else MUTED}" font-size="11" font-weight="700">{escape(DISPLAY.get(d, d))}</text>'
            f'<text x="614" y="{y + 3.6:.1f}" fill="{"#4cc9f0" if rel else MUTED}" font-size="10" text-anchor="end">{len(classes)}{f" + {functions[d]} fn" if functions.get(d) else ""}</text>'
        )
        if rel:
            linked += 1
            o.append(
                f'<a href="{url_of(rel)}" target="_top"><title>{escape(d)}: open {escape(rel)}</title>{body}</a>'
            )
        else:
            o.append(f"<g>{body}</g>")
        o.append(
            f'<text x="640" y="{y + 3.6:.1f}" fill="{MUTED}" font-size="9.4" class="m">{escape(short(classes)) if classes else "no class of this domain in the catalog"}</text>'
        )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 18:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">jobs and domains: dense_armor/usable/ontology.json; classes: dense_armor.usable.catalog; drawn by docs/assets/maps/make_usable_map.py</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print(
        "wrote",
        OUT.relative_to(REPO),
        f"{len(order)} jobs, {len(domains)} domains, {linked} linked",
    )


SITE = "https://tatopenn-cell.github.io/Dense-Armor/"
START, END = "<!-- usable-map:start -->", "<!-- usable-map:end -->"


def site_url(rel):
    return (
        SITE
        + rel.removesuffix("index.md").removesuffix(".md")
        + ("" if rel.endswith("index.md") else "/")
    )


def readme_block(jobs, order):
    """The picture, which opens the interactive map, and a table whose domains are links."""
    rows = []
    for j in order:
        cells = []
        for d in jobs[j]["domains"]:
            rel = page_of(d)
            cells.append(f"[{DISPLAY.get(d, d)}]({site_url(rel)})" if rel else d)
        if not cells:
            cells = [f"`{m}`" for m in jobs[j]["modules"]]
        rows.append(f"| `{j}` | {' &middot; '.join(cells)} |")
    return chr(10).join(
        [
            START,
            f'<p align="center"><a href="{SITE}api/general/"><img src="docs/assets/maps/jobs.png" alt="The map of Dense-Armor by job: what you want to do, and the domains that serve it" width="860"></a></p>',
            "",
            f"*What you want to do, and the domains of Dense-Armor that serve it, as `usable` knows them ({len(order)} jobs). Click the picture for the interactive map, where a domain opens its page and its classes are listed; the table below has the same links.*",
            "",
            "| Job | Domains |",
            "|---|---|",
            *rows,
            END,
        ]
    )


def png():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MPath

    onto = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = jobs_of(onto)
    counts = defaultdict(set)
    fn = defaultdict(int)
    for i in get_catalog(REPO):
        if i.get("kind", "class") == "function":
            fn[i["domain"]] += 1
        else:
            counts[i["domain"]].add(i["class"])
    order, domains = order_nodes(jobs)
    dh = 2.5
    jh = len(domains) * dh / len(order)
    H = len(domains) * dh + 4
    fig_h = 1.5 + H * 0.1 + 0.4
    fig = plt.figure(figsize=(10.5, fig_h), facecolor=BG)
    fig.text(
        0.5,
        1 - 0.45 / fig_h,
        "THE MAP BY JOB",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        1 - 0.95 / fig_h,
        "what you want to do, and the domains of Dense-Armor that serve it",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#4cc9f0",
    )
    ax = fig.add_axes([0, 0.35 / fig_h, 1, H * 0.1 / fig_h], facecolor=BG)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, H)
    ax.axis("off")
    off_j = (H - len(order) * jh) / 2
    off_d = (H - len(domains) * dh) / 2
    jy = {j: H - off_j - jh * (k + 0.5) for k, j in enumerate(order)}
    dy = {d: H - off_d - dh * (k + 0.5) for k, d in enumerate(domains)}
    color = {j: COLORS[k % 3] for k, j in enumerate(order)}
    for j in order:
        for d in jobs[j]["domains"]:
            p0, p1 = (31, jy[j]), (64, dy[d])
            path = MPath(
                [p0, (p0[0] + 17, p0[1]), (p1[0] - 17, p1[1]), p1],
                [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
            )
            ax.add_patch(
                PathPatch(path, fc="none", ec=color[j], lw=1.1, alpha=0.4, zorder=1)
            )
    for j in order:
        roles = (
            ", ".join(jobs[j]["roles"])
            or ", ".join(jobs[j]["modules"][:1])
            or "by domain"
        )
        ax.add_patch(
            FancyBboxPatch(
                (2, jy[j] - 2.0),
                29,
                4.0,
                boxstyle="round,pad=0.1,rounding_size=0.9",
                fc=CARD,
                ec=color[j],
                lw=1.4,
                zorder=2,
            )
        )
        ax.text(
            3.2,
            jy[j] + 0.75,
            j,
            color=TEXT,
            fontsize=8.6,
            fontweight="bold",
            family="monospace",
            va="center",
            zorder=3,
        )
        ax.text(
            3.2,
            jy[j] - 0.95,
            roles[:36],
            color=MUTED,
            fontsize=6.4,
            va="center",
            zorder=3,
        )
    for d in domains:
        has = page_of(d) is not None
        ax.add_patch(
            FancyBboxPatch(
                (64, dy[d] - 0.95),
                30,
                1.9,
                boxstyle="round,pad=0.05,rounding_size=0.9",
                fc=CARD,
                ec="#4cc9f0" if has else "#334155",
                lw=1.1,
                zorder=2,
            )
        )
        ax.text(
            65.2,
            dy[d],
            d,
            color=TEXT if has else MUTED,
            fontsize=7.4,
            fontweight="bold",
            family="monospace",
            va="center",
            zorder=3,
        )
        ax.text(
            92.8,
            dy[d],
            str(len(counts.get(d, ()))) + (f" + {fn[d]} fn" if fn.get(d) else ""),
            color="#4cc9f0" if has else MUTED,
            fontsize=7,
            ha="right",
            va="center",
            zorder=3,
        )
    ax.text(
        64,
        H - 0.2,
        "DOMAIN  and  classes in the catalog",
        color=PINK,
        fontsize=7.6,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        2,
        H - 0.2,
        "JOB  and  the role it asks for",
        color=PINK,
        fontsize=7.6,
        fontweight="bold",
        va="bottom",
    )
    fig.text(
        0.5,
        0.12 / fig_h,
        "jobs and domains: dense_armor/usable/ontology.json; classes: dense_armor.usable.catalog; drawn by docs/assets/maps/make_usable_map.py",
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )
    fig.savefig(HERE / "jobs.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", (HERE / "jobs.png").relative_to(REPO))


def readme():
    onto = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = jobs_of(onto)
    order, _ = order_nodes(jobs)
    block = readme_block(jobs, order)
    path = REPO / "README.md"
    text = path.read_text(encoding="utf-8")
    CRLF = chr(13) + chr(10)
    nl = CRLF if CRLF in text else chr(10)
    text = text.replace(CRLF, chr(10))
    if START in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        anchor = "Check out [the API](https://tatopenn-cell.github.io/Dense-Armor/api/) for a comprehensive overview."
        assert anchor in text
        text = text.replace(anchor, anchor + chr(10) * 2 + block, 1)
    path.write_bytes(text.replace(chr(10), nl).encode("utf-8"))
    print("README.md updated")


if __name__ == "__main__":
    draw()
    png()
    readme()
