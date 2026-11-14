"""cli: the argparse commands and the entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import render
from .catalog import REPO, get_catalog
from .readers import (
    fetch,
    is_url,
    needs_from_profile,
    profile,
)
from .select import NEEDS, cmd_select
from .shell import HELP, _Exit, pick_file, shell


def cmd_list(repo: Path, domain=None, role=None, as_json=False, refresh=False) -> int:
    """Phase 1: every primitive."""
    t0 = time.perf_counter()
    cat = get_catalog(repo, refresh)
    rows = [
        r
        for r in cat
        if (not domain or r["domain"] == domain) and (not role or r["role"] == role)
    ]
    if as_json:
        print(json.dumps(rows, indent=1))
        return 0
    print(
        f"{len(rows)} primitives ({(time.perf_counter() - t0) * 1e3:.0f} ms), read from {repo}"
    )
    print(f"{'name':38s} {'domain':12s} {'role':16s} {'budget':>8s}  module")
    for r in sorted(rows, key=lambda r: (r["domain"], r["name"])):
        b = f"{r['budget_s'] * 1e3:g} ms" if r["budget_s"] is not None else "-"
        print(
            f"{r['name']:38s} {r['domain']:12s} {str(r['role'] or '-'):16s} {b:>8s}  {r['module']}"
        )
    return 0


def load_target(target: str, insecure: bool = False):
    """A path or an http address -> (local path, profile, jobs, budget, notes). Raises ValueError with a reason."""
    notes_head: list[str] = []
    if is_url(target):
        try:
            path, note = fetch(target, insecure)
        except Exception as exc:
            raise ValueError(
                f"the address could not be read: {type(exc).__name__}: {exc}"
            ) from exc
        notes_head.append(note)
    else:
        path = Path(target.strip('"').strip("'"))
        if not path.exists():
            raise ValueError(f"file not found: {path}")
    pr = profile(path)
    needs, budget, notes = needs_from_profile(pr)
    return path, pr, needs, budget, notes_head + notes


def _remember_last_analysis(path: Path, pr: dict) -> None:
    """Write ``tmp/last_analysis.json`` (read by ``ontology gaps --last``)."""
    try:
        from .ontology import gaps as gp

        words = [w for w, _c in pr.get("keywords", [])]
        if words:
            gp.write_last_analysis({"file": path.name, "words": words})
    except Exception:
        # the cache is an optimisation, not a requirement: a failure
        # here must not break `analyze`
        pass


def cmd_analyze(
    repo: Path, file: str | None, top=3, as_json=False, why_not=None, insecure=False
) -> int:
    """Give a file or an http address; it is profiled, then phase 2 runs on what the profile says."""
    if not file:
        file = pick_file()
        if not file:
            print("no file chosen")
            return 1
    try:
        path, pr, needs, budget, notes = load_target(file, insecure)
    except ValueError as exc:
        print(str(exc))
        return 1
    if pr["kind"] == "unsupported":
        print(f"{path.name}:")
        for n in notes:
            print("  - " + n)
        return 1
    _remember_last_analysis(path, pr)
    return cmd_select(
        repo, needs, budget, pr, top, as_json, why_not, notes, title=f"{path.name}:"
    )


def cmd_ontology_gaps(
    repo: Path, last: str | None = None, as_json: bool = False
) -> int:
    """Item G: the classes without a link and, with --last, the words without a concept."""
    from .ontology import gaps as gp

    last_path = Path(last) if last else gp.last_analysis_path()
    rep = gp.report(repo, last_path)
    if as_json:
        print(json.dumps(rep, indent=1))
        return 0
    print(f"catalog: {rep['n_rows']} rows, {rep['n_classes']} distinct classes")
    print(f"classes with a link: {rep['n_with_link']}")
    print(f"classes without a link: {rep['n_without_link']}")
    print("  (formula: distinct classes of the catalog - distinct classes with a link)")
    if "last_file" in rep:
        print(f"last analysis: {rep['last_file']}")
        print(
            f"  words: {rep['last_words']}, without a concept: {rep['last_without_concept']}"
        )
        for w in rep["last_without_concept_words"][:20]:
            print(f"    {w}")
    else:
        print("last analysis: (no file; run `usable analyze FILE` first)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo = REPO
    if "--repo" in argv:
        i = argv.index("--repo")
        repo = Path(argv[i + 1])
        del argv[i : i + 2]
    if not argv:
        return shell(repo)
    if argv[0] in ("help", "-h", "--help"):
        print(HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "menu":
        return shell(repo)
    if cmd == "list":
        ap = argparse.ArgumentParser(prog="usable list")
        ap.add_argument("--domain")
        ap.add_argument("--role")
        ap.add_argument("--json", action="store_true")
        ap.add_argument("--refresh", action="store_true")
        a = ap.parse_args(rest)
        return cmd_list(repo, a.domain, a.role, a.json, a.refresh)
    if cmd == "select":
        ap = argparse.ArgumentParser(prog="usable select")
        ap.add_argument("jobs", nargs="+", choices=NEEDS)
        ap.add_argument("--budget-ms", type=float)
        ap.add_argument("--kind", choices=["table", "sound"], default=None)
        ap.add_argument("--top", type=int, default=3)
        ap.add_argument("--json", action="store_true")
        ap.add_argument("--why-not", metavar="NAME")
        a = ap.parse_args(rest)
        kind = a.kind or ("sound" if "audio_features" in a.jobs else "table")
        return cmd_select(
            repo,
            a.jobs,
            a.budget_ms / 1e3 if a.budget_ms else None,
            {"kind": kind, "rows": 0},
            a.top,
            a.json,
            a.why_not,
            title="selection:",
        )
    if cmd == "ontology":
        sub = rest[0] if rest else ""
        if sub != "gaps":
            print("usage: usable ontology gaps [--last FILE] [--json]", file=sys.stderr)
            return 2
        ap = argparse.ArgumentParser(prog="usable ontology gaps")
        ap.add_argument(
            "--last",
            default=None,
            help="a file written by `usable analyze` (default: tmp/last_analysis.json)",
        )
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(rest[1:])
        return cmd_ontology_gaps(repo, a.last, a.json)

    ap = argparse.ArgumentParser(prog="usable analyze")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--why-not", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--domain")
    ap.add_argument("--role")
    ap.add_argument(
        "--weak",
        action="store_true",
        help="show also the classes that share only some words with a definition (not a probability)",
    )
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="download an https address without checking its certificate",
    )
    a = ap.parse_args(rest if cmd == "analyze" else argv)
    if a.list:
        return cmd_list(repo, a.domain, a.role, a.json, a.refresh)
    render.SHOW_WEAK = a.weak
    return cmd_analyze(repo, a.file, a.top, a.json, a.why_not, a.insecure)


def run(entry=None) -> None:
    """The entry point of the installed command: Esc and Ctrl+C leave it quietly."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(  # type: ignore[union-attr]
                errors="replace"
            )  # a character the console cannot write is replaced, not fatal
        except (AttributeError, ValueError):
            pass
    try:
        code = (entry or main)()
    except (KeyboardInterrupt, _Exit):
        print()
        code = 0
    sys.exit(code)
