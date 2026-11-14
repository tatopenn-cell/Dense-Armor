"""The words of a text against the words of the library.

The keywords of a text are taken to English (the glossary of glossary_it.json, word by word), widened with a few
synonyms and a few opposites, and compared with the words of every primitive: its name split in words and the
first line of its description. A hit is listed with the word that caused it. Nothing is guessed: a word that is
not in the glossary is kept as it is (loanwords such as 'synth' or 'kick' are English already).
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

IT_WORDS = set(
    "il lo la i gli le un una di del della dei delle che e per con non sono come anche questo questa nel nella "
    "sul alla ai al se si piu ma o quando dove sopra sotto tra fra".split()
)
EN_WORDS = set(
    "the of and to in is that for with as on are this by an be it from or at which not".split()
)

SYNONYMS = [
    {"cut", "remove", "attenuate", "reduce", "trim"},
    {"frequency", "pitch", "tone"},
    {"bass", "low"},
    {"high", "treble"},
    {"threshold", "limit", "ceiling"},
    {"compressor", "compression"},
    {"echo", "delay", "reverb"},
    {"noise", "hiss"},
    {"signal", "wave", "waveform"},
    {"detect", "find", "identify"},
    {"anomaly", "outlier", "novelty"},
    {"forecast", "predict", "prediction"},
    {"change", "drift", "shift"},
    {"error", "loss", "residual"},
    {"sample", "sampling"},
    {"band", "range", "bandwidth"},
    {"fraud", "intrusion", "breach"},
    {"image", "picture", "frame"},
]
OPPOSITES = [
    ("low", "high"),
    ("bass", "treble"),
    ("cut", "boost"),
    ("attack", "release"),
    ("increase", "decrease"),
    ("lowpass", "highpass"),
    ("fast", "slow"),
]
GENERIC_LIMIT = 25  # a word that is in more primitives than this says nothing
COMMON = set(
    """from with that this have been acts acting hitting between largest smallest keep keeps kept values value items item each every using used uses first last other
after before while where which when over under into onto upon within without about above below across along among around
because through during since until their there these those this that what whose will would should could might must also
only just very more most much many some such same then than both either neither again once here make makes made take takes
given gives give need needs want wants like likes know known show shown shows find found finds work works working""".split()
)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ]+", text.lower())[:4000]
    it = sum(w in IT_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "it" if it > en else "en"


def _glossary() -> dict:
    path = HERE / "glossary_it.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _to_english(word: str, glossary: dict) -> str | None:
    if word in glossary:
        return glossary[word]
    if len(word) > 4 and word[-1] in "aeio":
        for tail in ("o", "i", "e", "a"):
            if word[:-1] + tail in glossary:
                return glossary[word[:-1] + tail]
    return None


def _words_of(row: dict) -> set[str]:
    name = re.sub(r"\[.*?\]", "", row.get("class", row["name"]))
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    words = {p.lower() for p in parts if len(p) >= 3}
    words |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", row.get("doc", ""))}
    return words


def _neighbours(term: str) -> list[tuple[str, str]]:
    out = []
    for group in SYNONYMS:
        if term in group:
            out += [(w, "synonym") for w in sorted(group - {term})]
    for a, b in OPPOSITES:
        if term == a:
            out.append((b, "opposite"))
        elif term == b:
            out.append((a, "opposite"))
    return out


def _stem(w: str) -> str:
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def related(keywords: list, lang: str, rows: list[dict], top: int = 8) -> dict:
    """Primitives whose words meet the keywords of the text, with the words that caused each meeting."""
    glossary = _glossary() if lang == "it" else {}
    index: dict[str, set[str]] = {}
    for r in rows:
        for w in _words_of(r):
            index.setdefault(_stem(w), set()).add(r["name"])
    by_name = {r["name"]: r for r in rows}

    def hits(term: str) -> set[str]:
        t = _stem(term)
        found = set(index.get(t, ()))
        if len(t) >= 5:
            for w, names in index.items():
                if len(w) >= 5 and (w.startswith(t) or t.startswith(w)):
                    found |= names
        return found

    translated = matched = 0
    per: dict[str, list[tuple[str, str, str]]] = {}
    for word, _count in keywords:
        if word in COMMON:
            continue
        en = _to_english(word, glossary) if lang == "it" else word
        if en is None:
            term_list = [(word, "as written")]
        else:
            translated += 1
            term_list = [(en, "direct")] + _neighbours(en)
        used = False
        for term, kind in term_list:
            found = hits(term)
            if not found or len(found) > GENERIC_LIMIT:
                continue
            used = True
            label = f"{word}->{term}" if term != word else word
            for name in found:
                per.setdefault(name, []).append((word, kind, label))
        matched += used
    items = []
    for name, hs in per.items():
        direct = {w for w, k, _ in hs if k in ("direct", "as written", "synonym")}
        opposite = {w for w, k, _ in hs if k == "opposite"} - direct
        items.append(
            {
                "name": name,
                "domain": by_name[name]["domain"],
                "direct": len(direct),
                "opposite": len(opposite),
                "because": sorted(
                    {
                        (
                            f"{lab} (opposite)"
                            if k == "opposite"
                            else f"{lab} (synonym)"
                            if k == "synonym"
                            else lab
                        )
                        for _w, k, lab in hs
                    }
                ),
            }
        )
    items.sort(key=lambda x: (-x["direct"], -x["opposite"], x["name"]))
    return {
        "language": lang,
        "keywords": len(keywords),
        "translated": translated,
        "matched": matched,
        "items": items[:top],
    }
