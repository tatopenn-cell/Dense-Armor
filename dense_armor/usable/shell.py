"""shell: the interactive prompt."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import render
from .catalog import get_catalog
from .render import TITLES, _enable_colors, paint
from .select import NEEDS


def pick_file() -> str:
    """The window to choose a file: the one of Windows (always on top), tkinter as a fallback. '' if closed."""

    if os.name == "nt":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$o = New-Object System.Windows.Forms.Form; $o.TopMost = $true;"
            "$f = New-Object System.Windows.Forms.OpenFileDialog;"
            "$f.Title = 'usable: choose the file to analyse';"
            "$f.Filter = 'Data files|*.csv;*.tsv;*.json;*.jsonl;*.wav;*.txt;*.md;*.html;*.pdf;*.png;*.jpg;*.jpeg;*.mp4;*.avi|All files|*.*';"
            "if ($f.ShowDialog($o) -eq 'OK') { [Console]::Out.Write($f.FileName) }"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if out.stdout.strip():
                return out.stdout.strip()
            if out.returncode == 0:
                return ""
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.update()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            parent=root,
            title="Choose the file to analyse",
            filetypes=[
                ("Data files", "*.csv *.wav *.pdf *.txt *.json"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return chosen
    except Exception:
        return ""


class _Exit(Exception):
    """Esc was pressed: leave the program, without a message."""


def _ask(prompt: str) -> str:
    """One line of input. On a Windows terminal Esc leaves the program at once; Ctrl+C does the same."""
    if os.name == "nt" and sys.stdin.isatty():
        import msvcrt

        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf: list[str] = []
        while True:
            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                return "".join(buf).strip()
            if ch == "\x1b":
                sys.stdout.write("\n")
                raise _Exit
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x08", "\x7f"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()  # type: ignore[attr-defined]
                continue
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


COMMANDS = [
    (
        "list [domain]",
        "phase 1: every primitive (a domain, or:  list role AnomalyDetector)",
    ),
    (
        "select [jobs]",
        "phase 2: the right primitives for the file, or for the jobs you name",
    ),
    (
        "file [path|url]",
        "give the file (no argument: a window); it is read and phase 2 runs on it",
    ),
    (
        "card [path|url]",
        "the card: what the file or the folder is made of, unit by unit",
    ),
    (
        "ontology gaps",
        "the classes without a link and the words of the last file without a concept",
    ),
    (
        "weak",
        "show or hide the matches that share only some words with a definition (hidden at first)",
    ),
    ("help", "this list"),
    ("exit", "leave (or press Esc)"),
]


def banner(repo: Path) -> None:
    n = len(get_catalog(repo))
    print()
    print(paint("  usable", "1;36") + "  which primitives of Dense-Armor fit your data")
    print(paint(f"  {n} primitives, read from {repo}", "2"))
    print()
    show_commands()


def show_commands() -> None:
    print(paint("  commands", "1"))
    for name, text in COMMANDS:
        print("    " + paint(f"{name:17s}", "1;36") + text)
    print(paint("  jobs for select: " + ", ".join(NEEDS), "2"))
    print()


def describe(path: Path, pr: dict, needs: list[str], notes: list[str]) -> None:
    print(f"\n{paint(path.name, '1')}: what I understood")
    for n in notes:
        print("  - " + n)
    if pr["kind"] != "unsupported":
        print("  - jobs: " + ", ".join(TITLES[n] for n in needs))


def shell(repo: Path) -> int:
    """The interactive prompt: type a command."""
    from .__main__ import _run_card
    from .cli import _remember_last_analysis, cmd_list, cmd_ontology_gaps, load_target
    from .readers import is_url
    from .select import cmd_select

    _enable_colors()
    banner(repo)
    state: dict = {}
    while True:
        label = f"usable [{state['path'].name}]> " if state else "usable> "
        line = _ask(paint(label, "1;36"))
        if not line:
            continue
        word, _, rest = line.partition(" ")
        word, rest = word.lower(), rest.strip()
        if word in ("exit", "quit", "q"):
            return 0
        if word in ("help", "?", "h"):
            show_commands()
        elif word == "weak":
            render.SHOW_WEAK = not render.SHOW_WEAK
            print("  weak matches: " + ("shown" if render.SHOW_WEAK else "hidden"))
        elif word == "card":
            if rest:
                _run_card([rest.strip('"').strip("'")])
            else:
                print("  give the object:  card <path|url>")
        elif word == "ontology" or (word == "gaps" and not rest):
            cmd_ontology_gaps(repo)
        elif word in ("1", "list"):
            tokens = rest.split()
            if tokens[:1] == ["role"]:
                cmd_list(repo, None, tokens[1] if len(tokens) > 1 else None)
            else:
                cmd_list(repo, tokens[0] if tokens else None, None)
        elif word in ("2", "select"):
            jobs = [t for t in rest.replace(",", " ").split() if t in NEEDS]
            if jobs:
                kind = "sound" if "audio_features" in jobs else "table"
                cmd_select(
                    repo,
                    jobs,
                    state.get("budget"),
                    state.get("pr") or {"kind": kind, "rows": 0},
                    title="selection:",
                )
            elif state:
                cmd_select(
                    repo,
                    state["needs"],
                    state["budget"],
                    state["pr"],
                    title=f"{state['path'].name}:",
                )
            else:
                print(
                    "  no file yet: give one with  file <path>  or name the jobs:  select classify detect_drift"
                )
        else:
            target = rest if word in ("3", "file", "analyze") else line
            if word in ("3", "file", "analyze") and not target:
                t0 = time.perf_counter()
                target = pick_file()
                if not target:
                    quick = time.perf_counter() - t0 < 2.0
                    print(
                        "  the file window did not open here: drag the file into this window, or paste its path"
                        if quick
                        else "  no file chosen"
                    )
                    continue
            target = target.strip('"').strip("'")
            if (
                word not in ("3", "file", "analyze")
                and not is_url(target)
                and not Path(target).exists()
            ):
                print(f"  unknown command: {word}   (type help)")
                continue
            try:
                path, pr, needs, budget, notes = load_target(target)
            except ValueError as exc:
                print("  " + str(exc))
                continue
            describe(path, pr, needs, notes)
            if pr["kind"] == "unsupported":
                continue
            _remember_last_analysis(path, pr)
            state = {"path": path, "pr": pr, "needs": needs, "budget": budget}
            cmd_select(repo, needs, budget, pr, title=f"{path.name}:")


HELP = (
    """usable: the primitives of Dense-Armor that fit what you are building

  usable                      the menu: it shows the commands and waits
  usable list                 phase 1: every primitive            [--domain D] [--role R] [--json] [--refresh]
  usable select JOB [JOB ...] phase 2: the right ones for jobs     [--budget-ms N] [--kind table|sound] [--top N] [--json]
  usable analyze [FILE]       a file: profile it, then select      [--top N] [--json] [--why-not NAME] [--weak]
  usable card OBJECT          the card of a file, a folder or an address   [--base BASE] [--json]
  usable ontology gaps        classes without a link, words without a concept   [--last FILE] [--json]
  usable FILE                 the same as analyze

jobs: """
    + ", ".join(NEEDS)
    + """
Every command accepts --repo PATH (the Dense-Armor tree to read)."""
)
