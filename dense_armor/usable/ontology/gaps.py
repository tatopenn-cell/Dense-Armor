"""Report the classes without a link and the words without a concept.

The data lives in ``ontology.json``: ``links`` (concept -> class),
``synonyms`` (word -> word), ``evokes`` (concept -> job). A class of
the catalog without a link is a class the graph can never lift; a word
without a concept is a word that no edge can reach. The report names
both, so that the reader can widen ``ontology.json`` (item G of
``ISTRUZIONI.md``).

Choices stated (the original criterion in ``ISTRUZIONI.md`` says
"447 minus the distinct classes in ``ontology.json``"; the 447 is the
number of **rows** of the catalog, not of distinct classes: the
catalog keeps both ``KNNClassifier [core]`` and ``KNNClassifier [ml]``,
so the number of distinct classes is smaller. The formula used here is
the one that has the same unit on both sides: distinct classes of the
catalog minus distinct classes with a link).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..catalog import get_catalog

HERE = Path(__file__).resolve().parent


def _data() -> dict:
    return json.loads((HERE.parent / "ontology.json").read_text(encoding="utf-8"))


def _vocabulary(data: dict) -> set[str]:
    """Every word the graph knows: concepts with a link, synonyms, evokes."""
    words: set[str] = set(data.get("links", {}).keys())
    for a, b in data.get("synonyms", []):
        words.add(a)
        words.add(b)
    for names in data.get("evokes", {}).values():
        words.update(names)
    return words


def classes_without_link(repo: Path) -> list[str]:
    """The distinct classes of the catalog with no ``links`` entry, sorted.

    Distinct: two rows with the same ``class`` are one class.
    """
    cat = get_catalog(repo)
    distinct = {r["class"] for r in cat}
    linked = set()
    for targets in _data().get("links", {}).values():
        linked.update(targets)
    return sorted(distinct - linked)


def words_without_concept(words: list[str], repo: Path) -> list[str]:
    """The words that no edge of the graph can reach, sorted."""
    known = _vocabulary(_data())
    return sorted({w for w in words if w not in known})


def report(repo: Path, last_file: Path | None = None) -> dict:
    """The two numbers, and the lists."""
    cat = get_catalog(repo)
    distinct = {r["class"] for r in cat}
    missing = classes_without_link(repo)
    out = {
        "n_rows": len(cat),
        "n_classes": len(distinct),
        "n_with_link": len(distinct) - len(missing),
        "n_without_link": len(missing),
        "without_link": missing,
    }
    if last_file is not None and last_file.is_file():
        payload = json.loads(last_file.read_text(encoding="utf-8"))
        words = payload.get("words", [])
        no_concept = words_without_concept(words, repo)
        try:
            out["last_file"] = str(last_file.relative_to(HERE.parent))
        except ValueError:
            out["last_file"] = str(last_file)

        out["last_words"] = len(words)
        out["last_without_concept"] = len(no_concept)
        out["last_without_concept_words"] = no_concept
    return out


def write_last_analysis(payload: dict) -> Path:
    """Write ``tmp/last_analysis.json`` next to the package.

    The file is a cache of the last analysis, not data of the original: it
    can be deleted at any time. It is read by ``ontology gaps --last``.
    """
    tmp = HERE.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / "last_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def last_analysis_path() -> Path:
    return HERE.parent / "tmp" / "last_analysis.json"
