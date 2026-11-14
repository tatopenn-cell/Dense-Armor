"""The words of a text against the words of the library.

The keywords of a text are taken to English with the glossary of glossary_it.json, word by word; a word that is not in
the glossary is kept as it is (loanwords such as 'synth' or 'kick' are English already). The synonyms and the opposites
are in ontology.json, not here.
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


def _stem(w: str) -> str:
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    return w[:-1] if len(w) > 4 and w.endswith("s") else w
