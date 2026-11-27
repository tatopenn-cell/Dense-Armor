"""The interactive map of the Detection section: anomaly detection, drift detection and protection, with every page.

Draws docs/assets/maps/detection.svg as a clickable SVG with three columns, in the drawing of the map of online
learning (make_online_map.py), whose helpers it uses:

- left: Detection, with the number of areas and of classes;
- centre: the three areas, which are the sections of the menu, each with its pages and its classes in the catalog;
- right: every page of the area, from ``mkdocs.yml``.

Run it again when the menu changes:

    python docs/assets/maps/make_detection_map.py
"""

from html import escape
from pathlib import Path

from make_applications_map import BG, CARD, CYAN, HOME, MUTED, ORANGE, PINK, TEXT, count, url
from make_area_maps import DOCS, REPO
from make_online_map import classes, nav_sections

HERE = Path(__file__).resolve().parent
OUT = HERE / "detection.svg"
AREAS = (
    ("anomaly", "Anomaly detection", CYAN),
    ("drift", "Drift detection", PINK),
    ("protect", "Protection", ORANGE),
)


def main():
    sections = nav_sections()
    n_classes = classes()
    row, gap, top = 19.0, 5.0, 112.0
    y = top
    blocks = []
    for area, label, color in AREAS:
        idx, pages = sections[label]
        block = max(row * len(pages), 36.0)
        off = (block - row * len(pages)) / 2
        ys = [y + off + row * (k + 0.5) for k in range(len(pages))]
        blocks.append((label, idx, area, color, [(t, r, yy) for (t, r), yy in zip(pages, ys)], sum(ys) / len(ys)))
        y += block + gap
    H, W = y + 40, 1000.0
    gy = sum(b[5] for b in blocks) / len(blocks)
    total_classes = sum(n_classes.get(b[2], 0) for b in blocks)
    n_pages = sum(len(b[4]) for b in blocks)
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of detection">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.35;transition:opacity .15s}.net:hover .e{opacity:.07}.net .s:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP OF DETECTION</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">what is strange, what has changed, what to keep out of the model; pass over an area to light its pages, click to open</text>',
        f'<a href="{HOME}" target="_top"><title>back to the map of the library</title><rect class="card" x="20" y="14" width="150" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="95" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← map of the library</text></a>',
        f'<text x="24" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  SECTION</text>',
        f'<text x="236" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  AREA</text>',
        f'<text x="500" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">03  PAGES</text>',
        '<g class="net">',
    ]
    for label, idx, area, color, pages, my in blocks:
        o.append('<g class="s">')
        o.append(f'<path class="e" d="M164,{gy:.1f} C200,{gy:.1f} 204,{my:.1f} 238,{my:.1f}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        for _, _, py in pages:
            o.append(f'<path class="e" d="M388,{my:.1f} C430,{my:.1f} 436,{py:.1f} 490,{py:.1f}" fill="none" stroke="{color}" stroke-width="1.1"/>')
        o.append("</g>")
    o.append("</g>")
    o.append(
        f'<a href="../../detection/" target="_top"><title>Detection: open detection/index.md</title>'
        f'<rect class="card" x="20" y="{gy - 24:.1f}" width="144" height="48" rx="12" fill="{CARD}" stroke="{CYAN}" stroke-width="1.8"/>'
        f'<text class="dn m" x="32" y="{gy - 4:.1f}" fill="{TEXT}" font-size="14" font-weight="700">Detection</text>'
        f'<text x="32" y="{gy + 14:.1f}" fill="{MUTED}" font-size="9.5">{count(len(blocks), "area")} · {total_classes} classes</text></a>'
    )
    for label, idx, area, color, pages, my in blocks:
        nc = n_classes.get(area, 0)
        fs = 12.5 if len(label) <= 17 else 10.5
        sub = f"{count(len(pages), 'page')} · {nc} class" + ("" if nc == 1 else "es")
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
        f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">pages: the menu of mkdocs.yml; classes: the catalog of the library (dense_armor.usable.catalog); drawn by docs/assets/maps/make_detection_map.py</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT.relative_to(REPO), f"{len(blocks)} areas, {n_pages} pages, {total_classes} classes, height {H:.0f}")


if __name__ == "__main__":
    main()
