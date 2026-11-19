"""The map of the library, folder by folder, drawn from the code of `dense_armor` as a clickable SVG.

Three columns, which follow the folders of ``dense_armor``:

- left: the big sets, the folders of the package (``core``, ``roles``, ``dynamics``, ``bridges``,
  ``mcp_server``);
- centre: what is directly inside each of them: the sections of ``core`` (``online``, ``signals``,
  ``deep``, ...) and the single files of the others;
- right: what is inside each centre box: the pieces of a section (a sub-folder or a file), or the
  classes of a file.

``core`` is read from the catalog of `usable`; the other folders are read from their files, so that
the roles and the MCP server, which the catalog does not list, are there too. ``ml`` is a folder of
its own in the package, but on the map it is drawn as a section of ``core``. Every box opens a page
of the docs. Run it again when the code changes:

    python docs/assets/maps/make_library_map.py
"""

import ast
from collections import defaultdict
from html import escape
from pathlib import Path

from dense_armor.usable.catalog import get_catalog

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
REPO = DOCS.parent
PKG = REPO / "dense_armor"
OUT = HERE / "library.svg"

BG, CARD, TEXT, MUTED = "#0d131a", "#101a24", "#e0fbfc", "#94a3b8"
CYAN, ORANGE, PINK = "#4cc9f0", "#ff9e00", "#f72585"

# the folders on the left, in order, and their colours
LEFT = ["core", "roles", "dynamics", "bridges", "mcp_server"]
COLORS = {
    "core": CYAN,
    "roles": PINK,
    "dynamics": ORANGE,
    "bridges": CYAN,
    "mcp_server": PINK,
}
# folders that sit outside `core` in the package but are drawn inside it
IN_CORE = ("ml",)

BIG_PAGES = {
    "core": "api/index.md",
    "roles": "roles/index.md",
    "dynamics": "dynamics/index.md",
    "bridges": "bridges/index.md",
    "mcp_server": "mcp.md",
}
SECTION_PAGES = {
    "online": "online/index.md",
    "deep": "neural/overview.md",
    "signals": "preprocessing/index.md",
    "audio": "audio/index.md",
    "evaluation": "metrics/index.md",
    "finance": "finance/index.md",
    "graph": "ontology/index.md",
    "infrastructure": "stream/index.md",
    "ml": "ml/index.md",
    "utility": "shields/engine.md",
    "armor": "shields/armor.md",
    "anomaly_detector": "roles/detectors.md",
    "drift_detector": "roles/detectors.md",
    "transformer": "roles/transformers.md",
}
PIECE_PAGES = {
    "datasets": "stream/index.md",
    "linear": "linear/index.md",
    "evaluate": "metrics/index.md",
    "compose": "stream/index.md",
    "data": "api/data_shared.md",
    "neural": "neural/overview.md",
    "kalman": "timeseries/kalman.md",
    "Armor": "shields/armor.md",
}


def url(rel):
    return "../../" + (
        rel.removesuffix("index.md").removesuffix(".md")
        + ("" if rel.endswith("index.md") else "/")
    )


def page(rel):
    return rel if rel and (DOCS / rel).exists() else None


def count_label(n, functions=False):
    kind = "function" if functions else "class"
    return f"{n} {kind}" + ("" if n == 1 else "s" if functions else "es")


def short(names, limit=40):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def catalog_tree():
    """big set -> section -> piece -> [classes], from the module path of every class in the catalog."""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in get_catalog(REPO):
        if r.get("kind", "class") != "class":
            continue
        p = r["module"].removeprefix("dense_armor.").split(".")
        if p[0] == "core" and len(p) > 2:
            big, sec, rest = "core", p[1], p[2:]
        else:
            big, sec, rest = "dense_armor", p[0], p[1:] or [p[0]]
        piece = r["class"] if len(p) == 1 else rest[0]
        if r["class"] not in t[big][sec][piece]:
            t[big][sec][piece].append(r["class"])
    return t


def file_sections(folder):
    """[(file, [(piece, [names])])] for a folder that holds single files: every public class of a
    file is a piece; a file with no class lists its public functions."""
    out = []
    for f in sorted((PKG / folder).glob("*.py")):
        if f.name.startswith("_"):
            continue
        body = ast.parse(f.read_text(encoding="utf-8")).body
        classes = [
            n.name
            for n in body
            if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
        ]
        funcs = [
            n.name
            for n in body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        ]
        if classes:
            out.append((f.stem, [(c, [c]) for c in classes]))
        elif funcs:
            out.append((f.stem, [(fn, ["function"]) for fn in funcs]))
    return out


def by_size(secs):
    return sorted(secs, key=lambda s: (-sum(len(c) for c in secs[s].values()), s))


def build():
    """The groups, in order: [(name, colour, [(section, [(piece, [names])])])]."""
    t = catalog_tree()
    core = []
    secs = dict(t.get("core", {}))
    for sec in IN_CORE:
        if sec in t.get("dense_armor", {}):
            secs[sec] = t["dense_armor"][sec]
    for sec in by_size(secs):
        core.append(
            (sec, sorted(secs[sec].items(), key=lambda kv: (-len(kv[1]), kv[0])))
        )
    groups = [("core", COLORS["core"], core)]
    for folder in LEFT[1:]:
        groups.append((folder, COLORS[folder], file_sections(folder)))
    rest = {
        s: p
        for s, p in t.get("dense_armor", {}).items()
        if s not in LEFT and s not in IN_CORE
    }
    if "armor" in rest and "utility" in rest:
        rest["utility"].update(rest.pop("armor"))
    groups.append(
        (
            "outside",
            ORANGE,
            [
                (s, sorted(rest[s].items(), key=lambda kv: (-len(kv[1]), kv[0])))
                for s in by_size(rest)
            ],
        )
    )
    return groups


def main():
    groups = build()
    row, gap, top = 19.0, 6.0, 108.0
    y = top
    laid = []
    for name, col, secs in groups:
        blocks = []
        for sec, pieces in secs:
            block = max(row * len(pieces), 40.0)
            off = (block - row * len(pieces)) / 2
            ys = [y + off + row * (k + 0.5) for k in range(len(pieces))]
            blocks.append(
                (sec, [(p, n, yy) for (p, n), yy in zip(pieces, ys)], sum(ys) / len(ys))
            )
            y += block + gap
        y += 10
        laid.append((name, col, blocks))
    H = y + 30
    W = 1000.0
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of the library, folder by folder">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.35;transition:opacity .15s}.net:hover .e{opacity:.07}.net .s:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP OF THE LIBRARY, FOLDER BY FOLDER</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">the big sets, what is inside them and what is inside that; pass over a box in the middle to light its pieces, click to open a page</text>',
        f'<text x="24" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  BIG SET</text>',
        f'<text x="236" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  SECTION OR FILE</text>',
        f'<text x="500" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  PIECES, WITH THEIR CLASSES</text>',
        '<g class="net">',
    ]
    left_y = {}
    for name, col, blocks in laid:
        if name in LEFT and blocks:
            left_y[name] = sum(b[2] for b in blocks) / len(blocks)
    for name, col, blocks in laid:
        for sec, pieces, my in blocks:
            o.append('<g class="s">')
            if name in left_y:
                ly = left_y[name]
                o.append(
                    f'<path class="e" d="M164,{ly:.1f} C200,{ly:.1f} 204,{my:.1f} 238,{my:.1f}" fill="none" stroke="{col}" stroke-width="1.4"/>'
                )
            for _, _, py in pieces:
                o.append(
                    f'<path class="e" d="M388,{my:.1f} C430,{my:.1f} 436,{py:.1f} 490,{py:.1f}" fill="none" stroke="{col}" stroke-width="1.1"/>'
                )
            o.append("</g>")
    o.append("</g>")
    n_pieces = 0
    for name, col, blocks in laid:
        n_all = sum(len(p[1]) for b in blocks for p in b[1])
        if name in left_y:
            ly = left_y[name]
            rel = page(BIG_PAGES.get(name))
            body = (
                f'<rect class="card" x="20" y="{ly - 24:.1f}" width="144" height="48" rx="12" fill="{CARD}" stroke="{col}" stroke-width="1.8"/>'
                f'<text class="dn m" x="34" y="{ly - 4:.1f}" fill="{TEXT}" font-size="14" font-weight="700">{escape(name)}</text>'
                f'<text x="34" y="{ly + 14:.1f}" fill="{MUTED}" font-size="10">{count_label(n_all)}</text>'
            )
            o.append(
                f'<a href="{url(rel)}" target="_top"><title>{escape(name)}: open {escape(rel)}</title>{body}</a>'
                if rel
                else f"<g>{body}</g>"
            )
        for sec, pieces, my in blocks:
            n = sum(len(p[1]) for p in pieces)
            fns = all(p[1] == ["function"] for p in pieces)
            fs = 12.5 if len(sec) <= 17 else 10.5 if len(sec) <= 22 else 8.8
            rel = (
                page(SECTION_PAGES.get(sec))
                or page(f"{sec}/index.md")
                or page(BIG_PAGES.get(name))
            )
            body = (
                f'<rect class="card" x="238" y="{my - 17:.1f}" width="150" height="34" rx="10" fill="{CARD}" stroke="{col}" stroke-width="1.6"/>'
                f'<text class="dn m" x="250" y="{my - 2:.1f}" fill="{TEXT}" font-size="{fs}" font-weight="700">{escape(sec)}</text>'
                f'<text x="250" y="{my + 11:.1f}" fill="{MUTED}" font-size="9.5">{count_label(n, fns)}</text>'
            )
            o.append(
                f'<a href="{url(rel)}" target="_top"><title>{escape(sec)}: open {escape(rel)}</title>{body}</a>'
                if rel
                else f"<g>{body}</g>"
            )
            for piece, names, py in pieces:
                n_pieces += 1
                prel = page(PIECE_PAGES.get(piece)) or page(f"{piece}/index.md") or rel
                single = names == [piece] or names == ["function"]
                pbody = (
                    f'<rect class="card" x="492" y="{py - 8.5:.1f}" width="486" height="17" rx="8.5" fill="{CARD}" stroke="{col}" stroke-width="1.1"/>'
                    f'<text class="dn m" x="504" y="{py + 3.6:.1f}" fill="{TEXT}" font-size="10.5" font-weight="700">{escape(piece)}</text>'
                    + (
                        ""
                        if single
                        else f'<text x="700" y="{py + 3.5:.1f}" fill="{MUTED}" font-size="9.2" class="m">{escape(short(names, 52))}</text>'
                        f'<text x="966" y="{py + 3.5:.1f}" fill="{col}" font-size="9.5" text-anchor="end">{len(names)}</text>'
                    )
                )
                o.append(
                    f'<a href="{url(prel)}" target="_top"><title>{escape(piece)}: open {escape(prel)}</title>{pbody}</a>'
                    if prel
                    else f"<g>{pbody}</g>"
                )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">core: dense_armor.usable.catalog; the other folders: their files; drawn by docs/assets/maps/make_library_map.py</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print(
        "wrote",
        OUT.relative_to(REPO),
        f"{len(laid)} groups, {n_pieces} pieces, height {H:.0f}",
    )
    for name, col, blocks in laid:
        print(
            " ",
            name,
            "left" if name in left_y else "centre only",
            [b[0] for b in blocks][:18],
        )


if __name__ == "__main__":
    main()
