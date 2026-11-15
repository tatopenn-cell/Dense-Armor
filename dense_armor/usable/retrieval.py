"""Ask the library: every class of Dense-Armor is a document, every definition of a file is a question.

Documents: the words of the class name (weight 3), of its description, of its methods and of its module.
Questions: the words of a definition (its name and the first line of its docstring), widened with the synonyms of
ontology.json. The search is the cosine similarity between TF-IDF vectors (a class and a question are vectors of words
weighted by how rare each word is in the library). The links of the ontology (concept -> class) add to the similarity of
the classes they name. A class answers when it shares two words of the question and reaches MIN_SIMILARITY (one word is never enough); p is the share of the similarity among
the answers shown. If no class is similar enough, nothing is answered.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

from . import keywords as K

HERE = Path(__file__).resolve().parent
EXTRA_STOP = set(
    """self return returns given using used uses use based specialized cached built lazily lazy bare module level
function functions class classes object objects method methods value values parameter parameters argument arguments
default optional list dict none true false number numbers array arrays input output inputs outputs type types""".split()
)
MIN_SIMILARITY = 0.2  # and at least two words of the question in common
LINK_WEIGHT = 0.5
STOP = K.COMMON | EXTRA_STOP | K.IT_WORDS | K.EN_WORDS


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
    out = []
    for w in re.findall(r"[A-Za-z]{3,}", text):
        w = K._stem(w.lower())
        if w not in STOP:
            out.append(w)
    return out


def _synonyms() -> dict[str, list[str]]:
    data = json.loads((HERE / "ontology.json").read_text(encoding="utf-8"))
    syn: dict[str, list[str]] = {}
    for a, b in data["synonyms"]:
        syn.setdefault(K._stem(a), []).append(K._stem(b))
        syn.setdefault(K._stem(b), []).append(K._stem(a))
    return syn


def terms_of(name: str, doc: str) -> dict[str, float]:
    """The question of a definition: the words of its name (weight 2) and of its description (weight 1)."""
    terms: dict[str, float] = {}
    for w in tokens(name):
        terms[w] = max(terms.get(w, 0.0), 2.0)
    for w in tokens(doc):
        terms[w] = max(terms.get(w, 0.0), 1.0)
    return terms


def terms_from_keywords(keywords: list, lang: str, limit: int = 15) -> dict[str, float]:
    """The question of a text: its most frequent words, taken to English with the glossary."""
    glossary = K._glossary() if lang == "it" else {}
    top = max((c for _w, c in keywords), default=1) or 1
    terms: dict[str, float] = {}
    for w, c in keywords[: limit * 2]:
        en = K._to_english(w, glossary) if lang == "it" else w
        for t in tokens(en or w):
            terms[t] = max(terms.get(t, 0.0), 0.5 + c / top)
    return dict(sorted(terms.items(), key=lambda kv: -kv[1])[:limit])


class Index:
    def __init__(self, rows: list[dict]):
        self.docs: dict[str, Counter] = {}
        self.rows = {r["name"]: r for r in rows}
        for r in rows:
            c: Counter = Counter()
            for w in tokens(r.get("class", r["name"])):
                c[w] += 3
            for w in tokens(r.get("doc_full") or r.get("doc", "")):
                c[w] += 1
            for w in tokens(" ".join(r.get("methods", []))):
                c[w] += 1
            for w in tokens(r["module"].replace(".", " ") + " " + r["domain"]):
                c[w] += 1
            self.docs[r["name"]] = c
        self.n = len(self.docs)
        self.df: Counter = Counter()
        for c in self.docs.values():
            self.df.update(c.keys())
        self.avg = sum(sum(c.values()) for c in self.docs.values()) / max(self.n, 1)
        self.syn = _synonyms()
        self.vec: dict[str, dict[str, float]] = {}
        self.norm: dict[str, float] = {}
        for name, c in self.docs.items():
            v = {t: (1.0 + math.log(tf)) * self.idf(t) for t, tf in c.items()}
            self.vec[name] = v
            self.norm[name] = math.sqrt(sum(x * x for x in v.values())) or 1.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(
        self,
        terms: dict[str, float],
        links: dict[str, float] | None = None,
        top: int = 5,
    ) -> list[dict]:
        """The classes most similar to the question (cosine of the TF-IDF vectors), with p = their share."""
        wide = dict(terms)
        for t, w in terms.items():
            for syn in self.syn.get(t, []):
                wide[syn] = max(wide.get(syn, 0.0), 0.7 * w)
        q = {
            t: w * self.idf(t)
            for t, w in wide.items()
            if 0 < self.df.get(t, 0) <= 0.25 * self.n
        }
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        out = []
        for name, v in self.vec.items():
            dot, matched = 0.0, []
            for t, w in q.items():
                if t in v:
                    dot += w * v[t]
                    matched.append(t)
            sim = dot / (qn * self.norm[name])
            cls = self.rows[name].get("class", name)
            link = (links or {}).get(cls, 0.0)
            own = sum(1 for t in matched if t in terms)
            if link <= 0 and not (own >= 2 and sim >= MIN_SIMILARITY):
                continue
            out.append(
                {
                    "name": name,
                    "domain": self.rows[name]["domain"],
                    "similarity": sim,
                    "score": sim + LINK_WEIGHT * link,
                    "matched": matched,
                    "link": link,
                }
            )
        out.sort(key=lambda r: -r["score"])
        out = out[:top]
        total = sum(r["score"] for r in out) or 1.0
        for r in out:
            r["p"] = r["score"] / total
        return out
