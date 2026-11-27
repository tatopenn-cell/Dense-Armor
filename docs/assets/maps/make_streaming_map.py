"""The interactive map of the Streaming section: two columns, the section and its modules.

Draws docs/assets/maps/streaming.svg as a clickable SVG in the drawing of the map of detection (make_detection_map.py):
- left: Streaming, the section;
- right: one row for each module, with its name once and its page; the last row is the page of Mahalanobis, which is in the
  anomaly section;
- the button at the top goes back to Detection.

Run from the repo root:
    python docs/assets/maps/make_streaming_map.py
"""

from html import escape
from pathlib import Path

from make_applications_map import BG, CARD, CYAN, MUTED, PINK, TEXT, count, url
from make_area_maps import DOCS, REPO

HERE = Path(__file__).resolve().parent
OUT = HERE / "streaming.svg"
MODULES = [
    ("streamDD", "anomaly/streaming/streaming_deviation_detector.md"),
    ("streamDS", "anomaly/streaming/streaming_deviation_scorer.md"),
    ("MultistreamDD", "anomaly/streaming/multi_channel_streaming_deviation_detector.md"),
    ("HampelScorer", "anomaly/streaming/hampel_scorer.md"),
    ("TukeyScorer", "anomaly/streaming/tukey_scorer.md"),
    ("ChauvenetScorer", "anomaly/streaming/chauvenet_scorer.md"),
    ("SigmaClipScorer", "anomaly/streaming/sigma_clip_scorer.md"),
    ("HampelFilter", "anomaly/streaming/hampel_filter.md"),
    ("class_seg_multi", "anomaly/streaming/class_seg_multi.md"),
    ("OnlineRobustMahalanobis", "anomaly/mahalanobis.md"),
]
ROWS = [(t, r) for t, r in MODULES if (DOCS / r).exists()]


def main():
    row, top = 22.0, 112.0
    ys = [top + row * (k + 0.5) for k in range(len(ROWS))]
    gy = sum(ys) / len(ys)
    H, W = top + row * len(ROWS) + 46, 1000.0
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of streaming">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.m{font-family:ui-monospace,Menlo,Consolas,monospace}"
        ".e{opacity:.35;transition:opacity .15s}.net:hover .e{opacity:.07}.net .s:hover .e{opacity:1;stroke-width:2.2}"
        "a:hover .card{fill:#17303f;stroke-width:2.4}a:hover text.dn{fill:#ffffff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="38" fill="{TEXT}" font-size="24" font-weight="700" text-anchor="middle">THE MAP OF STREAMING</text>',
        f'<text x="{W / 2:.0f}" y="62" fill="{CYAN}" font-size="12.5" text-anchor="middle">the modules of the streaming section; pass over a module to light it, click to open its page</text>',
        f'<a href="{url("detection/index.md")}" target="_top"><title>back to Detection</title><rect class="card" x="20" y="14" width="130" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="85" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← Detection</text></a>',
        f'<text x="24" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">01  SECTION</text>',
        f'<text x="290" y="{top - 14:.0f}" fill="{PINK}" font-size="12" font-weight="700">02  MODULES</text>',
        '<g class="net">',
    ]
    for py in ys:
        o.append(f'<g class="s"><path class="e" d="M184,{gy:.1f} C230,{gy:.1f} 240,{py:.1f} 288,{py:.1f}" fill="none" stroke="{CYAN}" stroke-width="1.2"/></g>')
    o.append("</g>")
    o.append(
        f'<a href="{url("anomaly/streaming/index.md")}" target="_top"><title>Streaming: open the index of the section</title>'
        f'<rect class="card" x="20" y="{gy - 26:.1f}" width="164" height="52" rx="12" fill="{CARD}" stroke="{CYAN}" stroke-width="1.8"/>'
        f'<text class="dn" x="34" y="{gy - 4:.1f}" fill="{TEXT}" font-size="16" font-weight="700">Streaming</text>'
        f'<text x="34" y="{gy + 15:.1f}" fill="{MUTED}" font-size="10">{count(len(ROWS), "module")}</text></a>'
    )
    for (title, rel), py in zip(ROWS, ys):
        o.append(
            f'<a href="{url(rel)}" target="_top"><title>{escape(title)}: open {escape(rel)}</title>'
            f'<rect class="card" x="290" y="{py - 9.5:.1f}" width="690" height="19" rx="9.5" fill="{CARD}" stroke="{CYAN}" stroke-width="1.2"/>'
            f'<text class="dn m" x="304" y="{py + 4:.1f}" fill="{TEXT}" font-size="11.5" font-weight="700">{escape(title)}</text>'
            f'<text x="966" y="{py + 3.8:.1f}" fill="{MUTED}" font-size="9.5" class="m" text-anchor="end">{escape(rel)}</text></a>'
        )
    o.append(f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">drawn by docs/assets/maps/make_streaming_map.py</text>')
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT.relative_to(REPO), f"{len(ROWS)} modules, height {H:.0f}")


if __name__ == "__main__":
    main()
