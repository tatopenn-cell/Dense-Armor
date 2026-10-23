"""Check that each of the 16 neural classes is in a code block of its page.

Reads ``TABLE`` (16 class names -> page) and, for each class, checks that
the name appears inside a ``~~~python`` block of its page. Executes every
``~~~python`` block of the five pages that items B to F touch and compares
the output with the ``~~~`` block that follows. Prints one line per class
(``OK``/``MISSING``), one line per block (``MATCH``/``MISMATCH`` with the
two outputs) and the counts. Exit code 0 when all classes are ``OK`` and
every block with an expected output is ``MATCH``.
"""

from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

TABLE: dict[str, str] = {
    "Sigmoid": "docs/neural/index.md",
    "Tanh": "docs/neural/index.md",
    "Softmax": "docs/neural/index.md",
    "PRelu": "docs/neural/index.md",
    "SoftmaxCrossEntropy": "docs/neural/index.md",
    "BCEWithLogits": "docs/neural/index.md",
    "Huber": "docs/neural/index.md",
    "MaxPool1D": "docs/neural/vision_layers.md",
    "AvgPool1D": "docs/neural/vision_layers.md",
    "AvgPool2D": "docs/neural/vision_layers.md",
    "Upsample1D": "docs/neural/deep.md",
    "Upsample2D": "docs/neural/deep.md",
    "LastStep": "docs/neural/sequence_layers.md",
    "MeanOverTime": "docs/neural/sequence_layers.md",
    "AddPositions": "docs/neural/sequence_layers.md",
    "OnlineNetClassifier": "docs/neural/continuous.md",
}

BLOCK = re.compile(
    r"~~~python\n(.*?)~~~[ \t]*\n(?:[ \t]*\n)?~~~\n(.*?)\n~~~",
    re.DOTALL,
)
CODE_ONLY = re.compile(r"~~~python\n(.*?)~~~", re.DOTALL)


def extract_blocks(text: str) -> list[tuple[str, str | None]]:
    """Return ``(code, expected)`` for every python block; expected may be None."""
    blocks: list[tuple[str, str | None]] = []
    paired_starts: set[int] = set()
    for m in BLOCK.finditer(text):
        blocks.append((m.group(1), m.group(2).strip()))
        paired_starts.add(m.start(1))
    for m in CODE_ONLY.finditer(text):
        if m.start(1) in paired_starts:
            continue
        blocks.append((m.group(1), None))
    return blocks


def names_in_block(code: str, names: list[str]) -> list[str]:
    """Return the subset of ``names`` whose identifier appears in ``code``."""
    return [n for n in names if re.search(r"\b" + re.escape(n) + r"\b", code)]


def check_page(
    page: Path, names: list[str]
) -> tuple[list[tuple[str, bool]], list[tuple[int, bool, str | None, str | None]]]:
    """Check one page.

    Only the blocks that mention one of ``names`` are executed: the
    other blocks (some of them train a network for minutes) are not
    the subject of the check.

    Returns ``(classes, blocks)``: ``classes`` is a list of ``(name, found)``;
    ``blocks`` is a list of ``(index, ok, expected, got)`` for the
    relevant blocks that carry an expected output.
    """
    if not page.exists():
        return ([(n, False) for n in names], [])
    text = page.read_text(encoding="utf-8")
    blocks = extract_blocks(text)
    found: set[str] = set()
    results: list[tuple[int, bool, str | None, str | None]] = []
    idx = 0
    for code, expected in blocks:
        mentioned = names_in_block(code, names)
        if not mentioned:
            continue
        found.update(mentioned)
        idx += 1
        if expected is None:
            continue
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, {"__name__": "__check__"})
            got = buf.getvalue().strip()
        except Exception as e:  # noqa: BLE001
            got = f"EXC {type(e).__name__}: {e}"
        results.append((idx, got == expected, expected, got))
    classes = [(n, n in found) for n in names]
    return (classes, results)


def main() -> int:
    names_by_page: dict[str, list[str]] = {}
    for name, page in TABLE.items():
        names_by_page.setdefault(page, []).append(name)

    all_classes: list[tuple[str, bool]] = []
    all_blocks: list[tuple[str, int, bool, str | None, str | None]] = []
    for page in sorted(names_by_page):
        classes, blocks = check_page(Path(page), names_by_page[page])
        all_classes.extend(classes)
        for i, ok, exp, got in blocks:
            all_blocks.append((page, i, ok, exp, got))

    print("== classes ==")
    for name, ok in all_classes:
        print(f"  {name:20s} {'OK' if ok else 'MISSING'}  ({TABLE[name]})")

    print("== blocks ==")
    n_match = 0
    n_mismatch = 0
    for page, i, ok, exp, got in all_blocks:
        print(f"  {page} block {i}: {'MATCH' if ok else 'MISMATCH'}")
        if ok:
            n_match += 1
        else:
            n_mismatch += 1
            if exp is not None and got is not None:
                print(f"    expected: {exp!r}")
                print(f"    got:      {got!r}")

    n_ok = sum(1 for _, ok in all_classes if ok)
    total_blocks = n_match + n_mismatch
    print(
        f"\nclasses: {n_ok}/{len(all_classes)} OK; "
        f"blocks: {n_match}/{total_blocks} MATCH"
    )
    if n_ok == len(all_classes) and n_mismatch == 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
