"""Translate the keywords of a text to English (item K).

The ontology is English only. The words of an incoming text are
translated at the entrance, on the keywords only (never on the
full text). A word that the backend cannot translate stays as it
is, and the number of translated words goes into the card.

Backends, selected by name (constructor argument or the
``USABLE_TRANSLATOR`` environment variable):

- ``glossary`` (default): the manual Italian-to-English glossary in
  ``glossary_it.json`` (222 entries). Local, no download, always
  available.
- ``dictionary``: an Italian-to-English dictionary built once from
  the ConceptNet 5.7 it->en edges (``ontology/dictionary_it_en.tsv``,
  built by ``tools/build_dictionary.py``). Manual entries in the
  glossary win over dictionary entries. Local, no download at
  runtime.
- ``argos``: the offline Argos Translate package. Optional: the
  model is downloaded once into ``data/`` and never at startup or
  in the tests. Missing package -> ``available=False`` and a clear
  message, never a crash.

No backend calls a network service. No backend translates the
full text: only the keywords of the card.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from . import keywords as K

HERE = Path(__file__).resolve().parent
DICTIONARY_PATH = HERE / "ontology" / "dictionary_it_en.tsv"
BLOCKLIST_PATH = HERE / "data" / "dictionary_blocklist.tsv"
SUPPORTED = ("glossary", "dictionary", "argos")
DEFAULT = "glossary"


class Translator:
    """Translate the keywords of a text to English.

    Args:
        backend: one of ``"glossary"``, ``"dictionary"``,
            ``"argos"``. ``None`` reads the ``USABLE_TRANSLATOR``
            environment variable, then ``DEFAULT``.

    Raises:
        ValueError: if ``backend`` is not one of ``SUPPORTED``.
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = os.environ.get("USABLE_TRANSLATOR", DEFAULT)
        if backend not in SUPPORTED:
            raise ValueError(f"unknown backend {backend!r}: choose one of {SUPPORTED}")
        self.backend = backend
        self._dictionary: dict[str, str] | None = None
        self._argos = None
        self._blocklist: set[str] | None = None
        self._available = self._probe()

    def _probe(self) -> bool:
        if self.backend == "glossary":
            return True
        if self.backend == "dictionary":
            return DICTIONARY_PATH.is_file()
        if self.backend == "argos":
            return importlib.util.find_spec("argostranslate") is not None
        return False

    @property
    def available(self) -> bool:
        """True if the backend can run here."""
        return self._available

    def why_not_available(self) -> str:
        """The reason the backend is not available; empty if it is."""
        if self._available:
            return ""
        if self.backend == "dictionary":
            return (
                f"dictionary backend: {DICTIONARY_PATH.name} is missing. "
                "Build it with: python -m dense_armor.usable.tools.build_dictionary"
            )
        if self.backend == "argos":
            return (
                "argos backend: argostranslate is not installed. "
                "Install it once with: pip install argostranslate "
                "(the model is downloaded into data/ the first time)."
            )
        return ""

    def translate(self, words: list[str], lang: str) -> dict[str, str | None]:
        """Translate ``words`` from ``lang`` to English.

        Args:
            words: the keywords to translate.
            lang: ``"it"``, ``"en"``, or another tag. ``"en"`` returns
                the words unchanged. A tag the backends do not
                support returns ``None`` for every word.

        Returns:
            A dict ``word -> english | None``. ``None`` means the
            word was not translated and stays as it is.
        """
        if lang == "en":
            return {w: w for w in words}
        if lang != "it":
            return {w: None for w in words}
        if self.backend == "glossary":
            return self._glossary(words)
        if self.backend == "dictionary":
            return self._dictionary_backend(words)
        if self.backend == "argos":
            return self._argos_backend(words)
        return {w: None for w in words}

    def _glossary(self, words: list[str]) -> dict[str, str | None]:
        glossary = K._glossary()
        return {w: K._to_english(w, glossary) for w in words}

    def _dictionary_backend(self, words: list[str]) -> dict[str, str | None]:
        if self._dictionary is None:
            self._dictionary = self._load_dictionary()
        glossary = K._glossary()
        out: dict[str, str | None] = {}
        for w in words:
            en = K._to_english(w, glossary)
            if en is not None:
                out[w] = en
                continue
            if self._blocklist is None:
                self._blocklist = self._load_blocklist()
            if w.lower() in self._blocklist:
                out[w] = None
                continue
            out[w] = self._dictionary.get(w.lower())
        return out

    def _argos_backend(self, words: list[str]) -> dict[str, str | None]:
        if not self._available:
            return {w: None for w in words}
        tr = self._argos_translation()
        if tr is None:
            return {w: None for w in words}
        out: dict[str, str | None] = {}
        for w in words:
            try:
                en = tr.translate(w).strip().lower() or None
            except Exception:
                en = None
            out[w] = en
        return out

    @staticmethod
    def _load_dictionary() -> dict[str, str]:
        if not DICTIONARY_PATH.is_file():
            return {}
        out: dict[str, str] = {}
        with DICTIONARY_PATH.open("r", encoding="utf-8") as fh:
            next(fh, None)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                it, en = parts[0], parts[1]
                if it not in out:
                    out[it] = en
        return out

    def _load_blocklist(self) -> set[str]:
        """The entries that the ``dictionary`` backend must not use.

        Each line is ``italian <TAB> reason``. A comment starts with ``#``.
        The blocklist is a manual choice: entries of the dictionary
        (ConceptNet) that were checked and found wrong. The manual
        glossary always wins, and for the entries the glossary does not
        cover the blocklist returns them as ``None`` (the word stays as written).
        """
        if not BLOCKLIST_PATH.is_file():
            return set()
        out: set[str] = set()
        with BLOCKLIST_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if parts:
                    out.add(parts[0].lower())
        return out

    def _argos_translation(self):
        if self._argos is not None:
            return self._argos
        try:
            import argostranslate.translate as _argos_tr  # type: ignore[import-not-found, import-untyped]

            languages = _argos_tr.get_installed_languages()
            it = next((lang for lang in languages if lang.code == "it"), None)
            en = next((lang for lang in languages if lang.code == "en"), None)
            if it is None or en is None:
                self._argos = None
                return None
            self._argos = it.get_translation(en)
            return self._argos
        except Exception:
            self._argos = None
            return None
