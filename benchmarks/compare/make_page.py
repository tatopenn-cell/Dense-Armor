"""Write ``docs/compare/<family>.md``: the machine learning class against the online class.

    python benchmarks/compare/make_page.py FAMILY

The numbers are the JSON written by ``benchmarks/compare/<family>.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(family: str) -> None:
    data = json.loads(
        (ROOT / "benchmarks" / "compare" / f"{family}.json").read_text(encoding="utf-8")
    )
    intro = json.loads(
        (ROOT / "benchmarks" / "ml" / "intro.json").read_text(encoding="utf-8")
    )[family]
    out = [f"# {family.capitalize()}: machine learning against online learning\n"]
    out.append(
        "The machine learning classes read the whole table and the online classes read one row at a time. "
        "They are two different parts of the library; this page runs both on the same synthetic data and "
        "reports how far apart their outputs are.\n"
    )
    out.append(intro + "\n")
    out.append(
        f"## The difference, on synthetic data\n\n{data['seeds']} seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.\n"
    )
    out.append("| class | kind | median | p90 | max | why |\n|---|---|---|---|---|---|")
    for n, r in data["classes"].items():
        out.append(
            f"| `{n}` | {r['kind']} | {r['median']:.3g} | {r['p90']:.3g} | {r['max']:.3g} | {r['why']} |"
        )
    out.append(f"\nWilcoxon and A12: {data['wilcoxon_a12']}.\n")
    out.append(
        f"## Run it\n\n~~~bash\npython benchmarks/compare/{family}.py\n~~~\n\nThe command writes `benchmarks/compare/{family}.json`, the table above.\n"
    )
    dest = ROOT / "docs" / "compare"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{family}.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", dest / f"{family}.md")


if __name__ == "__main__":
    main(sys.argv[1])
