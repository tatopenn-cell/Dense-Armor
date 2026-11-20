"""The interactive map of the Stable variance page: where it is needed, and the four moves that make it work.

Left: the fields where a big number with small wiggles around it shows up, with the numbers measured on the page.
Centre: the function. Right: the four moves of the kernel (the Gamestory of the page). Pass over a card to light its
links; click a card to open its page. The figures are those of the page, from the Kaggle GPU test and the examples
that print them.

Run from the repo root:
    python docs/assets/maps/make_stable_var_map.py
"""

from html import escape

from make_tree_map import BG, CARD, CYAN, DOCS, HERE, MUTED, ORANGE, PINK, TEXT

OUT = HERE / "stable_var.svg"
W, H = 1000.0, 660.0
HOME = "../../#the-library-folder-by-folder"
HUB = (500.0, 290.0)

USES = [
    (
        "Finance",
        "an index near 4500, steps of 0.25",
        "0.0000 → 0.2528",
        "finance/index.md",
        CYAN,
    ),
    (
        "Robots and drones",
        "a joint near 1000, vibration 0.01",
        "wrong by 630 times → exact",
        "dynamics/index.md",
        ORANGE,
    ),
    (
        "Deep networks",
        "1024 layers, GPU, float32",
        "gradient wrong 21 % → 0.006 %",
        "neural/overview.md",
        PINK,
    ),
]
MOVES = [
    (
        "1  Shift",
        "y = x - x[0], exact when values are close",
        "stats/stable_variance.md#gamestory",
        CYAN,
    ),
    (
        "2  Centre",
        "d = y - m, the local centring",
        "stats/stable_variance.md#what-is-different-from-the-paper",
        ORANGE,
    ),
    (
        "3  TwoSum",
        "a + b = s + err, exactly (Knuth)",
        "stats/stable_variance.md#what-is-different-from-the-paper",
        PINK,
    ),
    (
        "4  Kahan sum",
        "sequential, in the paper (Healy 2022)",
        "stats/stable_variance.md#what-is-different-from-the-paper",
        CYAN,
    ),
    (
        "5  Compensated tree",
        "in place of the sequential Kahan sum",
        "stats/stable_variance.md#3-what-the-function-does",
        ORANGE,
    ),
    (
        "6  Exact zero",
        "identical values give 0.0",
        "stats/stable_variance.md#gamestory",
        CYAN,
    ),
]


def url(rel):
    page, _, anchor = rel.partition("#")
    assert (DOCS / page).exists(), page
    path = page.removesuffix("index.md").removesuffix(".md")
    return (
        "../../"
        + path
        + ("" if page.endswith("index.md") else "/")
        + (f"#{anchor}" if anchor else "")
    )


def card(x, y, w, h, col, title, sub, extra, href, side):
    o = [
        f'<a href="{url(href)}" target="_top"><title>{escape(title)}: open {escape(href)}</title>'
    ]
    o.append(
        f'<rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{CARD}" stroke="{col}" stroke-width="1.6"/>'
    )
    o.append(
        f'<text class="dn" x="{x + 16}" y="{y + 28}" fill="{TEXT}" font-size="16" font-weight="700">{escape(title)}</text>'
    )
    o.append(
        f'<text x="{x + 16}" y="{y + 50}" fill="{MUTED}" font-size="12">{escape(sub)}</text>'
    )
    if extra:
        o.append(
            f'<text x="{x + 16}" y="{y + 76}" fill="{col}" font-size="14" font-weight="700">{escape(extra)}</text>'
        )
    o.append("</a>")
    return "\n".join(o)


def curve(x0, y0, x1, y1, col):
    mx = (x0 + x1) / 2
    return f'<path class="e" d="M{x0},{y0} C{mx},{y0} {mx},{y1} {x1},{y1}" fill="none" stroke="{col}" stroke-width="2.2"/>'


def main():
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="The map of stable_var: the fields where it is needed and the moves of the kernel">',
        "<style>text{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}.e{opacity:.4;transition:opacity .15s}"
        ".net:hover .e{opacity:.08}.net .g:hover .e{opacity:1;stroke-width:3}a:hover .card{fill:#17303f;stroke-width:2.6}a:hover text.dn{fill:#fff}</style>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{BG}" rx="14"/>',
        f'<text x="{W / 2:.0f}" y="40" fill="{TEXT}" font-size="26" font-weight="700" text-anchor="middle">THE MAP OF stable_var</text>',
        f'<text x="{W / 2:.0f}" y="66" fill="{CYAN}" font-size="13" text-anchor="middle">a big number with small wiggles around it, in float32: where it appears and how the kernel keeps the wiggles; pass over a card to light it, click to open it</text>',
        f'<text x="24" y="104" fill="{PINK}" font-size="12" font-weight="700">01  WHERE IT IS NEEDED</text>',
        f'<text x="726" y="88" fill="{PINK}" font-size="12" font-weight="700">02  THE MOVES</text>',
        f'<text x="24" y="548" fill="{PINK}" font-size="12" font-weight="700">03  THE JOB AND THE CLASS THAT USE IT</text>',
        '<g class="net">',
    ]
    ys_use = [120 + 144 * k for k in range(3)]
    ys_move = [104 + 90 * k for k in range(6)]
    for (t, s, ex, href, col), y in zip(USES, ys_use):
        o.append(f'<g class="g">{curve(274, y + 48, HUB[0] - 64, HUB[1], col)}')
        o.append(card(24, y, 250, 96, col, t, s, ex, href, "l"))
        o.append("</g>")
    for (t, s, href, col), y in zip(MOVES, ys_move):
        o.append(f'<g class="g">{curve(HUB[0] + 64, HUB[1], 726, y + 38, col)}')
        o.append(card(726, y, 250, 76, col, t, s, "", href, "r"))
        o.append("</g>")
    o.append(f'<g class="g">{curve(274, 592, HUB[0] - 50, HUB[1] + 44, ORANGE)}')
    o.append(
        card(
            24,
            560,
            250,
            64,
            ORANGE,
            "Job: summarize_stream",
            "the job of the ontology that reaches statistics",
            "",
            "stats/index.md",
            "l",
        )
    )
    o.append("</g>")
    o.append(
        f'<g class="g"><path class="e" d="M{HUB[0]},{HUB[1] + 64} L{HUB[0]},560" fill="none" stroke="{PINK}" stroke-width="2.2"/>'
    )
    o.append(
        card(
            375,
            560,
            250,
            64,
            PINK,
            "RunningMoments",
            "the class that calls stable_moments",
            "",
            "stats/moments.md",
            "c",
        )
    )
    o.append("</g>")
    o.append("</g>")
    o.append(
        f'<a class="back" href="{HOME}" target="_top"><title>back to the map of the library</title>'
        f'<rect class="card" x="20" y="14" width="150" height="30" rx="15" fill="{CARD}" stroke="{PINK}" stroke-width="1.6"/>'
        f'<text class="dn" x="95" y="33.5" fill="{TEXT}" font-size="12" font-weight="700" text-anchor="middle">← map of the library</text></a>'
    )
    o.append(
        f'<circle cx="{HUB[0]}" cy="{HUB[1]}" r="64" fill="{CARD}" stroke="{PINK}" stroke-width="2.4"/>'
    )
    o.append(
        f'<text x="{HUB[0]}" y="{HUB[1] - 4}" fill="{TEXT}" font-size="20" font-weight="700" text-anchor="middle">stable_var</text>'
    )
    o.append(
        f'<text x="{HUB[0]}" y="{HUB[1] + 18}" fill="{MUTED}" font-size="11" text-anchor="middle">stable_moments</text>'
    )
    o.append(
        f'<text x="{W / 2:.0f}" y="{H - 14:.0f}" fill="{MUTED}" font-size="10.5" text-anchor="middle">drawn by docs/assets/maps/make_stable_var_map.py; the figures are those of the page</text>'
    )
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT.relative_to(DOCS.parent))


if __name__ == "__main__":
    main()
