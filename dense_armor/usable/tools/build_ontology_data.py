"""Prepare the ontology data from free sources (item J).

Run once, not at startup: ``ontology/auto.json`` and ``data/LICENZE.md``
are written to disk; ``usable`` reads only the JSON. The sources are
free (WordNet: Princeton license; ConceptNet: CC BY-SA) and each row
carries its ``source`` and license.

How the data is found, in order:

1. ``--dry-run``: measure only. Says how many terms of the catalog are
   in WordNet (if ``nltk`` is installed), whether the ConceptNet subset
   exists (``data/conceptnet_subset.tsv``), what the schema of the
   output would be. Writes nothing.
2. Without ``--dry-run``: writes ``ontology/auto.json`` (synonyms and
   antonyms with level ``auto``) and ``data/LICENZE.md`` (source and
   license for each row). The manual links in ``ontology.json`` win:
   an ``auto`` link never moves a class above one with a manual link.

No download inside this script: if the ConceptNet subset is missing,
the script says the command that builds it
(``tools/build_conceptnet_subset.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ONTO = PKG / "ontology"
DATA = PKG / "data"


def _catalog_terms() -> list[str]:
    """The words of the class names and of the first line of their docstrings."""
    import re

    from ..catalog import get_catalog

    cat = get_catalog(Path(__file__).resolve().parents[3])
    words: set[str] = set()
    for r in cat:
        name = r.get("class", r["name"])
        # split camelCase
        for w in re.findall(r"[A-Z]?[a-z]+", name):
            if len(w) > 2:
                words.add(w.lower())
        doc = (r.get("doc") or "").strip().split("\n")[0]
        for w in re.findall(r"[A-Za-z]{4,}", doc):
            words.add(w.lower())
    return sorted(words)


def dry_run() -> int:
    """Measure only: no file written, no download."""
    terms = _catalog_terms()
    print(f"terms of the catalog: {len(terms)}")
    import importlib.util
    from typing import Any

    wn: Any = None
    if importlib.util.find_spec("nltk") is not None:
        try:
            from nltk.corpus import wordnet as _wn  # type: ignore[import-untyped]

            _wn.synsets("test")
            wn = _wn
        except LookupError:
            print(
                "WordNet: installed nltk, corpus missing; run nltk.download('wordnet')"
            )
    if wn is not None:
        in_wn = [t for t in terms if wn.synsets(t)]
        print(f"WordNet: {len(in_wn)}/{len(terms)} terms have a synset")
    # ConceptNet subset
    sub = DATA / "conceptnet_subset.tsv"
    if sub.is_file():
        n = sum(1 for _ in sub.open("r", encoding="utf-8", errors="ignore"))
        print(f"ConceptNet subset: {sub} ({n} rows)")
    else:
        print(f"ConceptNet subset: MISSING ({sub})")
        print(
            "  build it with: python -m dense_armor.usable.tools.build_conceptnet_subset --output data/conceptnet_subset.tsv"
        )
    # Schema output
    print("\nschema of ontology/auto.json:")
    print('  {"synonyms": [[a, b, {"source": ..., "license": ...}], ...],')
    print('   "antonyms": [[a, b, {"source": ..., "license": ...}], ...]}')
    print("level: 'auto' (proposed, not verified)")
    return 0


def build() -> int:
    """Write auto.json and LICENZE.md from the sources that are present."""
    terms = _catalog_terms()
    ONTO.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    auto: dict[str, list[list]] = {"synonyms": [], "antonyms": []}
    lic = []
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        wn.synsets("test")  # type: ignore[union-attr]
    except Exception:
        wn = None
    if wn is not None:
        terms_set = set(terms)
        for t in terms:
            for syn in wn.synsets(t):
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name == t or name not in terms_set:
                        continue
                    auto["synonyms"].append(
                        [
                            t,
                            name,
                            {
                                "source": "wordnet",
                                "license": "Princeton WordNet (MIT-like)",
                            },
                        ]
                    )
                    for ant in lemma.antonyms():
                        auto["antonyms"].append(
                            [
                                t,
                                ant.name().lower(),
                                {
                                    "source": "wordnet",
                                    "license": "Princeton WordNet (MIT-like)",
                                },
                            ]
                        )
        lic.append("WordNet (Princeton), MIT-like; https://wordnet.princeton.edu/")
    # ConceptNet subset
    sub = DATA / "conceptnet_subset.tsv"
    if sub.is_file():
        terms_set = set(terms)
        with sub.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                rel, a, b = parts[0], parts[1], parts[2]
                a = a.split("/")[-1].lower()
                b = b.split("/")[-1].lower()
                if a not in terms_set or b not in terms_set or a == b:
                    continue
                key = (
                    "synonyms"
                    if rel in ("Synonym", "RelatedTo")
                    else ("antonyms" if rel == "Antonym" else None)
                )
                if key:
                    auto[key].append(
                        [a, b, {"source": "conceptnet", "license": "CC BY-SA 4.0"}]
                    )
        lic.append("ConceptNet 5 (CC BY-SA 4.0); https://conceptnet.io/")
    for k in ("synonyms", "antonyms"):
        seen = set()
        uniq = []
        for row in auto[k]:
            uniq_key = tuple(sorted([row[0], row[1]]))
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            uniq.append(row)
        auto[k] = uniq
    (ONTO / "auto.json").write_text(
        json.dumps(auto, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (DATA / "LICENZE.md").write_text(
        "# Licenses of the data\n\n" + "\n".join(f"- {x}" for x in lic) + "\n",
        encoding="utf-8",
    )
    print(
        f"auto.json: {len(auto['synonyms'])} synonyms, "
        f"{len(auto['antonyms'])} antonyms, level 'auto'"
    )
    print(f"LICENZE.md: {len(lic)} source(s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_ontology_data")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="measure only; write nothing, download nothing",
    )
    a = ap.parse_args(argv)
    return dry_run() if a.dry_run else build()


if __name__ == "__main__":
    sys.exit(main())
