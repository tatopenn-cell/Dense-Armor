"""Build ``ontology/auto_links.json``: a link concept -> class for every class that has none.

The evidence is the catalog itself: each class belongs to a domain, and the name of the domain is the
concept. Every row has level ``auto``: rank 0.25 (below the manual ranks 0.5 and 1), a ``source`` and a license, and it
never replaces a link written by hand.

    python -m dense_armor.usable.tools.build_class_links --dry-run
    python -m dense_armor.usable.tools.build_class_links
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USABLE = HERE.parent
OUT = USABLE / "ontology" / "auto_links.json"
RANK = 0.25
SOURCE = {"source": "catalog domain of the class", "license": "Dense-Armor"}


def build(repo: Path) -> list[list]:
    from ..catalog import get_catalog

    manual = json.loads((USABLE / "ontology.json").read_text(encoding="utf-8"))
    linked = {cls for targets in manual.get("links", {}).values() for cls in targets}
    domains: dict[str, set[str]] = {}
    for row in get_catalog(repo):
        if row["class"] not in linked:
            domains.setdefault(row["class"], set()).add(row["domain"])
    return [
        [domain, cls, RANK, dict(SOURCE)]
        for cls in sorted(domains)
        for domain in sorted(domains[cls])
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_class_links")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the size, write nothing"
    )
    ap.add_argument("--repo", type=Path, default=USABLE.parents[1])
    a = ap.parse_args(argv)
    rows = build(a.repo)
    classes = {r[1] for r in rows}
    concepts = {r[0] for r in rows}
    print(
        f"{len(rows)} links, {len(classes)} classes, {len(concepts)} concepts (domains); rank {RANK}"
    )
    if a.dry_run:
        return 0
    OUT.write_text(json.dumps({"links": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"written: {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
