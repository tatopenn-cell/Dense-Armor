"""The interactive map of the finance section: the pages, grouped in four sections, with the classes each page uses.

Draws docs/assets/maps/finance.svg with the same drawing as the other section maps (make_tree_map.py). The job is
``finance_controls`` of the ontology. The classes of a page are the names of the catalog of `usable` that the page
writes between backticks or in its code blocks, the most used first; a page that names none shows what it is
for, from the tables of docs/finance/index.md.

Run from the repo root:
    python docs/assets/maps/make_finance_map.py
"""

import re
from collections import Counter

from make_tree_map import CYAN, DOCS, HERE, ORANGE, PINK, REPO, draw

from dense_armor.usable.catalog import get_catalog

SECTIONS = [
    (
        "Markets and risk",
        CYAN,
        [
            ("Surveillance", "surveillance.md"),
            ("Risk", "risk.md"),
            ("Regime", "regime.md"),
            ("Drift vs noise", "drift_vs_noise.md"),
            ("Single ticker", "single_ticker.md"),
            ("Static vs online", "static_vs_online.md"),
            ("Order book", "order_book.md"),
            ("Portfolio", "portfolio.md"),
            ("Stress", "stress.md"),
        ],
    ),
    (
        "Orders and execution",
        ORANGE,
        [
            ("Execution", "execution.md"),
            ("Schedules", "schedule.md"),
            ("Pre-trade limits", "pretrade.md"),
            ("Trading guard", "trading_guard.md"),
            ("FIX gateway", "fix_gateway.md"),
            ("ITCH messages", "itch.md"),
            ("Feed file", "feed.md"),
            ("Synthetic feed", "synthetic.md"),
        ],
    ),
    (
        "Fraud, intrusion and credit",
        PINK,
        [
            ("Fraud detection", "fraud.md"),
            ("Unsupervised fraud", "unsupervised_fraud.md"),
            ("Intrusion detection", "intrusion.md"),
            ("Credit scoring", "credit_scoring.md"),
        ],
    ),
    (
        "Running the desk",
        "#64748b",
        [
            ("Configuration", "config.md"),
            ("Decision log", "decision_log.md"),
            ("Audit table", "audit_db.md"),
            ("Status", "status.md"),
            ("Dashboard", "dashboard.md"),
            ("Red button", "emergency.md"),
            ("Demo", "demo.md"),
            ("Timing", "timing.md"),
            ("Ring buffer", "ring.md"),
            ("Streaming", "streaming.md"),
            ("Primitives map", "PRIMITIVES.md"),
        ],
    ),
]


def short(names, limit=40):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def descriptions():
    """What each page is for, from the tables of docs/finance/index.md: the text of the row, cut to fit a card."""
    out = {}
    for line in (DOCS / "finance" / "index.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \*\*[^|]+\*\* \| ([^|]+) \| \[[^\]]+\]\(([^)]+)\) \|$", line)
        if m:
            d = m.group(1).strip().rstrip(".")
            out[m.group(2)] = d if len(d) <= 52 else d[:49].rstrip() + "..."
    return out


def main():
    names = {i["class"] for i in get_catalog(REPO) if i.get("kind", "class") == "class"}
    pages, summaries, desc = [], {}, descriptions()
    for section, col, rows in SECTIONS:
        for title, rel in rows:
            text = (DOCS / "finance" / rel).read_text(encoding="utf-8")
            code = "\n".join(re.findall(r"```python\n(.*?)```", text, flags=re.S))
            tokens = re.findall(r"`([A-Z][A-Za-z0-9]+)(?:\(|`)", text) + re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", code)
            used = Counter(n for n in tokens if n in names)
            ordered = [n for n, _ in used.most_common()]
            pages.append((title, section, f"finance/{rel}", col, [title]))
            summaries[title] = (
                f"{len(ordered)} class{'' if len(ordered) == 1 else 'es'}: {short(ordered)}"
                if ordered
                else desc.get(rel, "")
            )
    draw(
        pages,
        ("finance",),
        "THE MAP OF FINANCE",
        "the job, the page of each finance pipeline in its section and the classes it uses; pass over a page to light it, click to open it",
        HERE / "finance.svg",
        "The map of finance",
        {},
        "",
        "docs/assets/maps/make_finance_map.py",
        partial=True,
        jobs=["finance_controls"],
        hub="finance",
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
