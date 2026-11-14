"""Build ``ontology/dictionary_it_en.tsv`` from the ConceptNet subset.

Reads ``data/conceptnet_subset.tsv`` (built by
``tools/build_conceptnet_subset.py``), keeps only the ``it -> en``
rows with relation ``Synonym`` or ``RelatedTo`` whose English word
is in the catalog vocabulary, and for every Italian word keeps the
English translation with the highest weight.

Output: TSV ``italian \\t english \\t weight``, one row per Italian
word. Local, no download: the subset is the input.

Run once, not at startup:

    python -m dense_armor.usable.tools.build_dictionary --dry-run
    python -m dense_armor.usable.tools.build_dictionary

The filter on the catalog vocabulary is stated, not silent: the
produced dictionary maps Italian words to English words that the
catalog can actually use. A translation to an English word outside
the catalog would never meet a class, so it is not kept.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
ONTO = PKG / "ontology"
SUBSET = DATA / "conceptnet_subset.tsv"
OUTPUT = ONTO / "dictionary_it_en.tsv"
KEEP = {"Synonym", "RelatedTo"}


def _catalog_english_words() -> set[str]:
    """The English words of the catalog and of the ontology (no Italian)."""
    from ..catalog import get_catalog

    cat = get_catalog(Path(__file__).resolve().parents[3])
    words: set[str] = set()
    for r in cat:
        for w in re.findall(r"[A-Z]?[a-z]+", r.get("class", r["name"])):
            if len(w) > 2:
                words.add(w.lower())
        for w in re.findall(r"[A-Za-z]{4,}", r.get("doc") or ""):
            words.add(w.lower())
    onto = json.loads((PKG / "ontology.json").read_text(encoding="utf-8"))
    for a, b in onto.get("synonyms", []) + onto.get("antonyms", []):
        words.add(a.lower())
        words.add(b.lower())
    for c in onto.get("links", {}):
        words.add(c.lower())
    for names in onto.get("evokes", {}).values():
        words.update(n.lower() for n in names)
    return words


def _iter(subset: Path, catalog_en: set[str]):
    with subset.open("r", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            rel, la, a, lb, b, w = parts[:6]
            if rel not in KEEP or la != "it" or lb != "en":
                continue
            if b.lower() not in catalog_en:
                continue
            try:
                weight = float(w)
            except ValueError:
                weight = 1.0
            yield a.lower(), b.lower(), weight


def dry_run() -> int:
    if not SUBSET.is_file():
        print(f"subset missing: {SUBSET}")
        print(
            "build it with: python -m dense_armor.usable.tools.build_conceptnet_subset"
        )
        return 1
    catalog_en = _catalog_english_words()
    words: dict[str, tuple[str, float]] = {}
    n = 0
    for a, b, w in _iter(SUBSET, catalog_en):
        n += 1
        if a not in words or w > words[a][1]:
            words[a] = (b, w)
    print(f"catalog English words: {len(catalog_en)}")
    print(f"filtered it->en rows: {n}")
    print(f"unique Italian words: {len(words)}")
    print(f"output: {OUTPUT}")
    return 0


def build() -> int:
    if not SUBSET.is_file():
        print(f"subset missing: {SUBSET}")
        return 1
    catalog_en = _catalog_english_words()
    words: dict[str, tuple[str, float]] = {}
    for a, b, w in _iter(SUBSET, catalog_en):
        if a not in words or w > words[a][1]:
            words[a] = (b, w)
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        fh.write("# italian\tenglish\tweight\n")
        for it in sorted(words):
            en, w = words[it]
            fh.write(f"{it}\t{en}\t{w}\n")
    print(f"wrote {OUTPUT} ({len(words)} entries)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_dictionary")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return dry_run() if a.dry_run else build()


if __name__ == "__main__":
    sys.exit(main())
