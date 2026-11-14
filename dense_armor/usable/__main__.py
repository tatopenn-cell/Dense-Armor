"""The entry point: `card` here, everything else to usable.main."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import card as _card
from .cli import main as _cli_main
from .cli import run as _cli_run


def _run_card(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="usable card")
    ap.add_argument("object", help="a file, a folder, a repo, or an http address")
    ap.add_argument(
        "--base",
        default=None,
        help="the base to compare against (default: Dense-Armor)",
    )
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        c = _card.analyze(a.object, a.base)
    except ValueError as exc:
        print(f"card: {exc}", file=sys.stderr)
        return 1
    problems = _card.validate(c)
    if a.json:
        print(_card.to_json(c))
    else:
        obj = c["object"]
        print(f"{obj['kind']}: {obj.get('path') or obj.get('url') or ''}")
        print(f"units: {len(c['units'])}")
        for u in c["units"][:20]:
            feats = ", ".join(
                f"{k}={v}" for k, v in list(u.get("features", {}).items())[:4]
            )
            print(f"  {u['kind']}/{u['name']}: {feats}")
        ranked = c["summary"].get("classes_ranked") or []
        if ranked:
            print("classes of Dense-Armor called by the code:")
            for e in ranked[:10]:
                files = ", ".join(Path(f).name for f in e["from_files"][:3])
                print(f"  {e['name']:30s} {e['mass']:g} call(s)   {files}")
    if problems:
        print("SCHEMA: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["card"]:
        return _run_card(argv[1:])
    return _cli_main(argv)


if __name__ == "__main__":
    _cli_run(main)
