"""Evaluate the selection against ``test/usable/samples/expected.json``.

Item A, check 13 of ``04_SELF_CHECK.md``. The number reported here is
the number of expected classes that appear in the top 3 of each job
for each sample file; it is not a percentage, it is a count that was
written down before the work and is reported as it comes out.

Usage:
    python -m dense_armor.usable.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .catalog import REPO, build_catalog
from .onto_select import select
from .readers import needs_from_profile, profile_csv, profile_wav
from .select import NEED_DOMAIN, NEED_ROLE, rank


def _expected_path() -> Path:
    """``test/usable/samples/expected.json`` next to the repo root.

    ``dense_armor/usable/evaluate.py`` -> ``repo_root``; from there,
    ``test/usable/samples/expected.json``.
    """
    here = Path(__file__).resolve()
    repo = here.parents[2]
    return repo / "test" / "usable" / "samples" / "expected.json"


def main() -> int:
    exp_path = _expected_path()
    if not exp_path.is_file():
        print(f"expected.json not found: {exp_path}", file=sys.stderr)
        return 1
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    cat = build_catalog(REPO)
    tot_hit = tot_exp = 0
    for name, spec in exp.items():
        p = exp_path.parent / name
        if not p.is_file():
            print(f"{name}: SKIP (file not found: {p})", file=sys.stderr)
            continue
        pr = profile_wav(p) if p.suffix == ".wav" else profile_csv(p)
        needs, budget, _notes = needs_from_profile(pr)
        accepted, broken = select(
            REPO,
            cat,
            pr,
            needs,
            budget,
            NEED_ROLE,
            NEED_DOMAIN,
        )
        ranked = rank(accepted, needs, budget)
        shown = {r["name"] for rows in ranked.values() for r in rows[:3]}
        pool = {r["name"] for r in accepted}
        exp_in = [e for e in spec["expected"] if e in pool]
        hits = [e for e in exp_in if e in shown]
        missing = [e for e in exp_in if e not in shown]
        print(f"{name}: needs {needs} budget {budget}")
        print(
            f"   shown {len(shown)} primitives; "
            f"expected {len(exp_in)}/{len(spec['expected'])} in pool; "
            f"found {len(hits)}; not shown: {missing}"
        )
        rejected = {e: broken[e] for e in spec["expected"] if e in broken}
        print(f"   rejected by the ontology: {rejected}")
        tot_hit += len(hits)
        tot_exp += len(exp_in)
    print(f"TOTAL {tot_hit}/{tot_exp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
