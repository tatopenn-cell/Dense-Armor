"""Build ``data/conceptnet_subset.tsv`` from the ConceptNet 5.7 dump (item J).

The dump is ~9 GB; the script streams it (gzip, one line at a time) and
keeps only the rows that matter: ``Synonym``, ``Antonym``, ``RelatedTo``
between two words of the catalog, English or Italian -> English. It
writes a TSV with four columns: ``relation\tstart\tend\tweight``.

Never called at startup, never called by the tests. Run it once, in
Colab or Kaggle, with ``--limit N`` for a smoke test:

    python -m dense_armor.usable.tools.build_conceptnet_subset --dry-run
    python -m dense_armor.usable.tools.build_conceptnet_subset --limit 5000
    python -m dense_armor.usable.tools.build_conceptnet_subset

``--dry-run`` says: the URL, the expected size, how many words of the
catalog it will keep. ``--limit N`` reads only the first N lines of the
dump (for a smoke test); the full run reads everything.

Dump license: CC BY-SA 4.0 (ConceptNet 5.7, page
https://conceptnet.io/). The produced file contains only the
filtered rows: if published, the same license applies.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
DUMP_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
KEEP = {"Synonym", "Antonym", "RelatedTo"}


def _catalog_words() -> set[str]:
    """The words of the class names and docstrings, lowercased, + ontology words."""
    import json
    import re

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
    for names in onto.get("evokes", {}).values():
        words.update(n.lower() for n in names)
    for c in onto.get("links", {}):
        words.add(c.lower())
    return words


def _iter_edges(limit: int | None, insecure: bool):
    """Stream the dump, yield (relation, lang_start, start, lang_end, end, weight).

    Row format (observed from the 5.7 dump):

        /a/[/r/<rel>/,/c/<l1>/<w1>/<pos>/...]  /r/<rel>  /c/<l1>/<w1>[/pos]  /c/<l2>/<w2>[/pos]  {json}
        0                                      1         2                  3                   4

    The language and the word live in the URI ``/c/<lang>/<word>[/pos]``:
    three segments for the word, four with the part of speech.
    """
    import json
    import ssl

    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(DUMP_URL, headers={"User-Agent": "usable-j/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
            n = 0
            for line in text:
                if limit is not None and n >= limit:
                    return
                n += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                rel_uri = parts[1]
                if not rel_uri.startswith("/r/"):
                    continue
                relation = rel_uri.removeprefix("/r/")
                if relation not in KEEP:
                    continue
                s = parts[2].split("/")
                e = parts[3].split("/")
                if len(s) < 4 or len(e) < 4 or s[1] != "c" or e[1] != "c":
                    continue
                lang_a, word_a = s[2], s[3]
                lang_b, word_b = e[2], e[3]
                if not word_a or not word_b or word_a.lower() == word_b.lower():
                    continue
                try:
                    meta = json.loads(parts[4])
                    weight = float(meta.get("weight", 1.0))
                except Exception:
                    weight = 1.0
                yield relation, lang_a, word_a, lang_b, word_b, weight


def dry_run(limit: int | None, insecure: bool) -> int:
    """Measure only: no file written, no download (only the count)."""
    words = _catalog_words()
    print(f"catalog + ontology words: {len(words)}")
    print(f"dump: {DUMP_URL}")
    print("expected size: ~380 MB (gzip), ~34 million lines, ~9 GB uncompressed")
    print(f"relations kept: {sorted(KEEP)}")
    print(f"limit: {limit if limit is not None else 'none (full run)'}")
    print(f"insecure: {insecure}")
    print("output: data/conceptnet_subset.tsv (relation, start, end, weight)")
    print("licenza: CC BY-SA 4.0 (ConceptNet 5.7)")
    return 0


def build(output: Path, limit: int | None, insecure: bool) -> int:
    """Write ``relation, lang_start, start, lang_end, end, weight``.

    Filter: keep a row only when at least one of the two words is in
    the catalog vocabulary (English) or is English/Italian (needed by
    item K, the glossary). The choice is stated, not silent: the
    produced file does not contain the full dump, only the rows that
    matter. If ``output`` is relative, it is resolved against the
    package directory; absolute paths are used as they are.
    """
    words = _catalog_words()
    kept = 0
    seen = set()
    if not output.is_absolute():
        output = PKG / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        fh.write("# relation\tlang_start\tstart\tlang_end\tend\tweight\n")
        for rel, la, a, lb, b, w in _iter_edges(limit, insecure):
            keep = (
                (la == "en" and a.lower() in words)
                or (lb == "en" and b.lower() in words)
                or (la == "it" and lb == "en")
                or (la == "en" and lb == "it")
            )
            if not keep:
                continue
            key = (rel, la, a.lower(), lb, b.lower())
            if key in seen:
                continue
            seen.add(key)
            fh.write(f"{rel}\t{la}\t{a}\t{lb}\t{b}\t{w}\n")
            kept += 1
            if kept % 500 == 0:
                print(f"  kept {kept}...", file=sys.stderr)
    print(f"wrote {output} ({kept} rows)")
    print("licenza: CC BY-SA 4.0 (ConceptNet 5.7)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_conceptnet_subset")
    ap.add_argument("--output", type=Path, default=DATA / "conceptnet_subset.tsv")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="read only the first N lines of the dump (smoke test)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--insecure", action="store_true", help="skip the certificate check (opt-in)"
    )
    a = ap.parse_args(argv)
    if a.dry_run:
        return dry_run(a.limit, a.insecure)
    return build(a.output, a.limit, a.insecure)


if __name__ == "__main__":
    sys.exit(main())
