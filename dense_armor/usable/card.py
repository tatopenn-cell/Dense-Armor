"""The card: one analyser and one card for every object.

The card of an object (file, folder, repo, url) is the same shape for every
kind: ``{schema_version, object, units, summary}``. For an audio file the
units are the properties of the sound; for a text the units are the
keywords; for a table the units are the columns; for a code file the units
are the definitions; for a folder or a repo there is one card per file and
a summary with the classes ranked by mass.

The analyser uses the primitives of Dense-Armor, not copies of them:
``AudioFeatures``, ``mel_spectrogram``, ``mfcc``, ``spectral_*``,
``onset_detect``, ``yin_f0`` for a sound; ``Tokenizer``, ``TFIDF``,
``TextVectorizer`` for a text; ``FrameFeatures`` for an image. The tables
have no profiler in the library: the prototype's ``_profile_table`` is kept
and its choices are declared in the delivery. The code files use ``ast``
through ``readers.profile_code``, which is the shape code2vec asks for.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from . import readers
from .launch import ensure_real

SCHEMA_VERSION = "1.0"
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}
MAX_FILES = 200
MAX_BYTES = 8 * 1024 * 1024


def _base_path(base) -> Path:
    """Resolve the base: an explicit path, ``USABLE_REPO``, or the package.

    No path that exists on one machine is written here (check 5 of
    ``04_SELF_CHECK.md``). The location of the base is a parameter.
    """
    import importlib.util
    import os

    if base is not None:
        p = Path(base) if not isinstance(base, Path) else base
        if not p.exists():
            raise ValueError(f"base not found: {p}")
        return p
    env = os.environ.get("USABLE_REPO")
    if env:
        p = Path(env)
        if not p.is_dir():
            raise ValueError(f"USABLE_REPO is not a directory: {p}")
        return p
    spec = importlib.util.find_spec("dense_armor")
    if spec and spec.origin:
        p = Path(spec.origin).resolve().parent.parent
        if (p / "dense_armor").is_dir():
            return p
    raise ValueError(
        "base is None and Dense-Armor is not importable: set USABLE_REPO or pass --base"
    )


def _audio_units(path: Path) -> list[dict]:
    """Properties of a sound, computed with Dense-Armor primitives.

    The energy bands come from ``AudioFeatures``, which takes an
    ``Audio`` object. The F0 is computed only when the file is long
    enough (at least 1024 samples), because YIN needs a window.
    """
    ensure_real()
    from dense_armor.core.audio import (
        AudioFeatures,
        mel_spectrogram,
        mfcc,
        onset_detect,
        read_wav,
        spectral_centroid,
        spectral_flatness,
        spectral_flux,
        yin_f0,
    )

    units: list[dict] = []
    a = read_wav(path)
    units.append(
        {
            "kind": "sound",
            "name": "waveform",
            "features": {
                "rate_hz": int(a.sr),
                "channels": int(a.x.ndim > 1 and a.x.shape[1] or 1),
                "n_samples": int(a.x.shape[0]),
                "duration_s": round(float(a.x.shape[0] / a.sr), 6),
            },
            "evidence": f"read_wav({path.name})",
        }
    )
    feats = AudioFeatures(n_bands=8).transform(a)
    units.append(
        {
            "kind": "sound",
            "name": "energy_bands",
            "features": {k: round(float(v), 6) for k, v in feats.items()},
            "evidence": "AudioFeatures(n_bands=8)",
        }
    )
    # Spettrogramma mel
    ms = mel_spectrogram(a.x, a.sr)
    units.append(
        {
            "kind": "sound",
            "name": "mel_spectrogram",
            "features": {
                "n_frames": int(ms.shape[0]),
                "n_mels": int(ms.shape[1]),
            },
            "evidence": "mel_spectrogram(x, sr)",
        }
    )
    # MFCC
    mf = mfcc(a.x, a.sr)
    units.append(
        {
            "kind": "sound",
            "name": "mfcc",
            "features": {
                "n_frames": int(mf.shape[0]),
                "n_coeffs": int(mf.shape[1]),
            },
            "evidence": "mfcc(x, sr)",
        }
    )
    # Spettrali
    sc = float(spectral_centroid(a.x, a.sr).mean())
    sf = float(spectral_flatness(a.x).mean())
    sfl = float(spectral_flux(a.x).mean())
    units.append(
        {
            "kind": "sound",
            "name": "spectral",
            "features": {
                "centroid_hz_mean": round(sc, 3),
                "flatness_mean": round(sf, 6),
                "flux_mean": round(sfl, 6),
            },
            "evidence": "spectral_centroid / spectral_flatness / spectral_flux",
        }
    )
    # Onset
    on = onset_detect(a.x, a.sr)
    units.append(
        {
            "kind": "sound",
            "name": "onsets",
            "features": {
                "n_onsets": int(len(on)),
                "first_s": round(float(on[0]), 4) if len(on) else None,
            },
            "evidence": "onset_detect(x, sr)",
        }
    )
    if a.x.shape[0] >= 1024:
        try:
            f0 = float(yin_f0(a.x[:1024], a.sr, 80.0, 0.1))
            units.append(
                {
                    "kind": "sound",
                    "name": "f0",
                    "features": {"f0_hz": round(f0, 2) if f0 == f0 else None},
                    "evidence": "yin_f0(first 1024 samples, sr, 80.0, 0.1)",
                }
            )
        except Exception as exc:
            units.append(
                {
                    "kind": "sound",
                    "name": "f0",
                    "features": {"error": type(exc).__name__},
                    "evidence": "yin_f0 (failed, not a number)",
                }
            )
    return units


def _text_units(path: Path) -> list[dict]:
    """Tokens and frequencies; TF-IDF only when the text has a corpus inside.

    Choices stated explicitly:
        - ``Tokenizer`` for the tokens (Dense-Armor primitive).
        - Frequencies (count, not TF-IDF) for the top terms: on a single
          short document TF-IDF is zero by construction, so a frequency
          is the honest number. This replaces the prototype's ``STOP``
          stoplist with the library's tokenizer.
        - TF-IDF (via ``TextVectorizer``) only when the text has at least
          ``MIN_SENTENCES_FOR_TFIDF`` sentences. The sentences are the
          documents of the internal corpus, and the vectorizer is fitted
          on them. On a shorter text the TF-IDF key is absent and the
          note says why.
        - Sentences are split with the same rule the prototype used.
        - The TF-IDF score of a term is the sum of its TF-IDF over
          the sentences, so a term that appears in many sentences
          ranks high.
    """
    from collections import Counter

    ensure_real()
    from dense_armor.core.signals.preprocessing.text import Tokenizer
    from dense_armor.core.signals.preprocessing.vectorizer import TextVectorizer

    from . import keywords as K
    from .translate import Translator

    MIN_SENTENCES_FOR_TFIDF = 3

    units: list[dict] = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    tok = Tokenizer()
    out = tok.transform_one({"text": raw})
    tokens = out.get("tokens", [])
    units.append(
        {
            "kind": "text",
            "name": "tokens",
            "features": {
                "n_chars": len(raw),
                "n_tokens": len(tokens),
                "n_unique": len(set(tokens)),
            },
            "evidence": "Tokenizer().transform_one",
        }
    )

    freq = Counter(tokens)
    top = freq.most_common(10)
    units.append(
        {
            "kind": "text",
            "name": "frequent_terms",
            "features": {term: int(count) for term, count in top},
            "evidence": "Tokenizer().transform_one + Counter",
        }
    )

    # item K: translate the keywords only, never the full text
    lang = K.detect_language(raw)
    if lang == "it" and top:
        words = [w for w, _c in top]
        tr = Translator()
        mapping = tr.translate(words, lang)
        translated = sum(1 for v in mapping.values() if v is not None)
        total = len(words)
        units.append(
            {
                "kind": "text",
                "name": "translation",
                "features": {
                    "backend": tr.backend,
                    "available": tr.available,
                    "lang": lang,
                    "translated": translated,
                    "total": total,
                    "mapping": mapping,
                    "why_not_available": tr.why_not_available(),
                },
                "evidence": f"Translator(backend={tr.backend!r})",
            }
        )

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", " ".join(raw.split()))
        if len(s.split()) >= 4
    ]
    if len(sentences) >= MIN_SENTENCES_FOR_TFIDF:
        tv = TextVectorizer()
        tv.fit(sentences)
        agg: dict[str, float] = {}
        for s in sentences:
            for term, score in tv.transform_one({"text": s}).get("scores", {}).items():
                agg[term] = agg.get(term, 0.0) + float(score)
        top_tfidf = sorted(agg.items(), key=lambda kv: -kv[1])[:10]
        units.append(
            {
                "kind": "text",
                "name": "tfidf_top",
                "features": {term: round(float(score), 6) for term, score in top_tfidf},
                "evidence": (
                    f"TextVectorizer().fit({len(sentences)} sentences) + "
                    "transform_one per sentence, TF-IDF summed over sentences"
                ),
            }
        )
    else:
        units.append(
            {
                "kind": "text",
                "name": "tfidf_top",
                "features": {
                    "skipped": f"only {len(sentences)} sentences, need {MIN_SENTENCES_FOR_TFIDF}"
                },
                "evidence": "TextVectorizer skipped: no internal corpus (choice stated)",
            }
        )

    return units


def _vision_units(path: Path) -> list[dict]:
    """Image properties, computed with Dense-Armor primitives."""
    ensure_real()
    from dense_armor.core.signals.vision import FrameFeatures

    units: list[dict] = []
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
            arr = np.array(im.convert("L"))
    except Exception as exc:
        return [
            {
                "kind": "image",
                "name": "image",
                "features": {"error": f"{type(exc).__name__}: {exc}"},
                "evidence": "PIL not available or unreadable file",
            }
        ]
    units.append(
        {
            "kind": "image",
            "name": "image",
            "features": {
                "width": int(width),
                "height": int(height),
                "mode": mode,
            },
            "evidence": f"PIL Image({path.name}).size",
        }
    )
    try:
        ff = FrameFeatures(n_bins=9, cells=(3, 4), gray=True)
        feats = ff.transform_one({"frame": arr})
        n_hog = sum(1 for k in feats if k.startswith("hog_"))
        units.append(
            {
                "kind": "image",
                "name": "hog",
                "features": {
                    "n_bins": int(n_hog),
                    "mean": round(float(feats.get("mean", 0.0)), 6),
                    "contrast": round(float(feats.get("contrast", 0.0)), 6),
                },
                "evidence": "FrameFeatures(n_bins=9, cells=(3, 4), gray=True)",
            }
        )
    except Exception as exc:
        units.append(
            {
                "kind": "image",
                "name": "hog",
                "features": {"error": f"{type(exc).__name__}: {exc}"},
                "evidence": "FrameFeatures (failed)",
            }
        )
    return units


def _table_units(profile: dict) -> list[dict]:
    """One unit per column, with type, semantic, range, top3.

    The profiler is the prototype's, extended in ``readers._profile_table``
    with the fields below. Dense-Armor has no table profiler; the paper
    reference is Cocoon (arXiv:2404.12552), the code is the prototype's.
    """
    units: list[dict] = []
    for col in profile.get("columns", []):
        units.append(
            {
                "kind": "table",
                "name": str(col.get("name", "?")),
                "features": {
                    "type": col.get("type", "unknown"),
                    "semantic": col.get("semantic", "unknown"),
                    "numeric": bool(col.get("numeric")),
                    "date": bool(col.get("date")),
                    "missing_pct": round(float(col.get("missing", 0.0)) * 100, 3),
                    "unique": int(col.get("unique", 0)),
                    "unique_ratio": float(col.get("unique_ratio", 0.0)),
                    "monotonic": bool(col.get("monotonic")),
                    "dt_s": col.get("dt"),
                    "range": col.get("range"),
                    "range_t": col.get("range_t"),
                    "lengths": col.get("lengths"),
                    "top3": col.get("top3", []),
                },
                "evidence": f"_profile_table: column {col.get('name')}",
            }
        )
    return units


def _code_units(path: Path) -> list[dict]:
    """One unit per definition; the shape code2vec asks for (name + calls)."""
    pr = readers.profile_code(path)
    units: list[dict] = []
    for d in pr.get("defs", []):
        units.append(
            {
                "kind": "code",
                "name": str(d.get("name", "?")),
                "features": {
                    "kind": d.get("kind"),
                    "line": int(d.get("line", 0)),
                    "n_args": len(d.get("args", [])),
                    "calls": list(d.get("calls", []))[:6],
                },
                "evidence": f"ast.parse({path.name})",
            }
        )
    return units


def analyze(object_path: str, base: str | None = None) -> dict:
    """The card of one object (a file, a folder, a repo, a url).

    For a folder or a repo: one card per file (up to ``MAX_FILES``), plus a
    summary. Raises ``ValueError`` with a reason on a missing or unreadable
    object.
    """
    target = str(object_path)
    is_url = readers.is_url(target)
    notes: list[str] = []
    if is_url:
        try:
            local, note = readers.fetch(target)
        except Exception as exc:
            raise ValueError(f"url not readable: {type(exc).__name__}: {exc}") from exc
        path = local
        notes.append(note)
    else:
        path = Path(target.strip().strip('"').strip("'"))
        if not path.exists():
            raise ValueError(f"object not found: {path}")

    if path.is_dir():
        return _analyze_folder(path, base, notes)
    return _analyze_file(path, base, notes)


def _analyze_file(path: Path, base, notes: list[str]) -> dict:
    pr = readers.profile(path)
    kind = pr.get("kind", "unsupported")
    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "object": {
            "kind": "file",
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "note": " | ".join(notes) if notes else "",
        },
        "units": [],
        "summary": {"n_files": 1, "n_units": 0, "classes_ranked": []},
    }
    if kind == "unsupported":
        card["object"]["note"] = pr.get("reason", "unsupported")
        return card
    if kind == "sound":
        card["object"]["kind"] = "sound"
        card["units"] = _audio_units(path)
    elif kind == "text":
        card["object"]["kind"] = "text"
        card["units"] = _text_units(path)
    elif kind == "vision":
        card["object"]["kind"] = "image"
        card["units"] = _vision_units(path)
    elif kind == "table":
        card["object"]["kind"] = "table"
        card["units"] = _table_units(pr)
    elif kind == "code":
        card["object"]["kind"] = "code"
        card["units"] = _code_units(path)
    card["summary"]["n_units"] = len(card["units"])
    for u in card["units"]:
        if u.get("name") == "translation":
            card["summary"]["translation"] = {
                "backend": u["features"].get("backend", ""),
                "available": bool(u["features"].get("available", False)),
                "lang": u["features"].get("lang", ""),
                "translated": int(u["features"].get("translated", 0)),
                "total": int(u["features"].get("total", 0)),
            }
            break
    return card


def _analyze_folder(path: Path, base, notes: list[str]) -> dict:
    """Card of a folder: one card per file, up to ``MAX_FILES``.

    The card declares how many files were skipped because of the bound
    (``MAX_FILES``) and because of the size limit (``MAX_BYTES``); the
    reader can see the bound at work without reading the code.
    """
    seen = 0
    files: list[Path] = []
    skipped_by_count = 0
    skipped_by_size = 0
    for p in sorted(path.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        seen += 1
        if p.stat().st_size > MAX_BYTES:
            skipped_by_size += 1
            continue
        if len(files) >= MAX_FILES:
            skipped_by_count += 1
            continue
        files.append(p)
    cards = [_analyze_file(f, base, []) for f in files]
    n_units = sum(c["summary"]["n_units"] for c in cards)
    note_parts = list(notes)
    if skipped_by_count:
        note_parts.append(
            f"skipped {skipped_by_count} files over MAX_FILES={MAX_FILES}"
        )
    if skipped_by_size:
        note_parts.append(f"skipped {skipped_by_size} files over MAX_BYTES={MAX_BYTES}")
    return {
        "schema_version": SCHEMA_VERSION,
        "object": {
            "kind": "folder",
            "path": str(path),
            "size_bytes": sum(f.stat().st_size for f in files),
            "note": " | ".join(note_parts) if note_parts else "",
        },
        "units": [
            {
                "kind": "file",
                "name": str(f),
                "features": c["object"],
                "evidence": "one card per file",
            }
            for f, c in zip(files, cards)
        ],
        "summary": {
            "n_files": len(files),
            "n_files_seen": seen,
            "n_files_skipped": skipped_by_count + skipped_by_size,
            "n_files_skipped_by_count": skipped_by_count,
            "n_files_skipped_by_size": skipped_by_size,
            "n_units": n_units,
            "classes_ranked": _rank_classes(cards, base),
        },
    }


def _foreign_imports(path: str) -> set[str]:
    """The names a file imports from outside Dense-Armor (``Counter``, ``Path``...): not the library's classes."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if (n.module or "").startswith("dense_armor"):
                continue
            out.update(a.asname or a.name for a in n.names)
        elif isinstance(n, ast.Import):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
    return out


def _rank_classes(cards: list[dict], base) -> list[dict]:
    """The classes of the base that the code of the cards calls by name, most called first, with the files that call them."""
    try:
        from .catalog import get_catalog

        repo = _base_path(base)
        names = {r.get("class", r["name"]) for r in get_catalog(repo)}
    except Exception:
        return []
    counts: dict[str, dict] = {}
    for c in cards:
        path = c["object"].get("path") or ""
        units = [u for u in c["units"] if u.get("kind") == "code"]
        own = {u["name"] for u in units}
        foreign = _foreign_imports(path) if path else set()
        for u in units:
            for call in u.get("features", {}).get("calls", []):
                last = call.rsplit(".", 1)[-1]
                if last not in names or last in own or last in foreign:
                    continue
                entry = counts.setdefault(
                    last, {"name": last, "mass": 0.0, "from_files": []}
                )
                entry["mass"] += 1.0
                if path and path not in entry["from_files"]:
                    entry["from_files"].append(path)
    return sorted(counts.values(), key=lambda e: (-e["mass"], e["name"]))[:20]


def to_json(card: dict) -> str:
    return json.dumps(card, indent=2, ensure_ascii=False)


def validate(card: dict) -> list[str]:
    """Check the card against card.schema.json (light: no jsonschema dependency).

    Returns the list of problems; empty if the card is valid enough for the
    task.
    """
    problems: list[str] = []
    if not isinstance(card, dict):
        return ["card is not a dict"]
    if card.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {card.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    obj = card.get("object")
    if not isinstance(obj, dict) or "kind" not in obj:
        problems.append("object.kind is missing")
    if not isinstance(card.get("units"), list):
        problems.append("units is not a list")
    else:
        for i, u in enumerate(card["units"]):
            if not isinstance(u, dict) or "kind" not in u or "name" not in u:
                problems.append(f"units[{i}] misses kind or name")
    if not isinstance(card.get("summary"), dict):
        problems.append("summary is missing")
    return problems
