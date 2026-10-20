"""Retrieval benchmark: TextVectorizer against scikit-learn's TfidfVectorizer.

The corpus is the Markdown pages under docs/, split into chunks of about
1000 characters with an overlap of 200. The query of a page is its first
heading and its first sentence; the relevant chunks are the ones of that
page. Both systems rank the chunks and the metric is the reciprocal rank
of the first relevant chunk.

scikit-learn is used only here and in tests, never in the library.

Run from the repository root:

    python benchmarks/text_retrieval.py
    python benchmarks/text_retrieval.py --smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dense_armor.utility.preprocessing.vectorizer import TextVectorizer
from dense_armor.utility.stats.similarity import (
    SparseCosineIndex,
    cosine_similarity_many,
)

CHUNK = 1000
OVERLAP = 200
SMOKE_QUERIES = 20
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def chunk_text(text: str, chunk: int = CHUNK, overlap: int = OVERLAP) -> list[str]:
    """Split a text into chunks of about ``chunk`` characters."""
    if chunk <= overlap:
        raise ValueError(f"chunk must be > overlap, got {chunk} <= {overlap}")
    step = chunk - overlap
    return [text[i:i + chunk] for i in range(0, max(len(text), 1), step)]


def first_query(text: str) -> str | None:
    """The first heading plus the first sentence after it."""
    m = _HEADING.search(text)
    if m is None:
        return None
    after = text[m.end():].strip()
    sentence = re.split(r"(?<=[.!?])\s", after, maxsplit=1)[0]
    return f"{m.group(1).strip()} {sentence[:200]}".strip()


def build_corpus(docs_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (chunks, page_of_chunk, queries) over the Markdown pages."""
    chunks, pages, queries = [], [], []
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        q = first_query(text)
        if q is None:
            continue
        page_id = str(path.relative_to(docs_dir))
        queries.append(q)
        pages.append(page_id)
        chunks.extend(chunk_text(text))
    return chunks, [p for p in pages for _ in chunk_text(
        (docs_dir / p).read_text(encoding="utf-8", errors="replace"))], queries




def _rr(ranked: np.ndarray, relevant: set[int]) -> float:
    for i, idx in enumerate(ranked):
        if int(idx) in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def run(docs_dir: Path, max_queries: int | None = None,
        use_sklearn: bool = True) -> dict:
    chunks, page_of_chunk, queries = build_corpus(docs_dir)
    pages = sorted(set(page_of_chunk))
    if max_queries is not None:
        pages = pages[:max_queries]
        queries = queries[:len(pages)]
        keep = set(pages)
        idx = [i for i, c in enumerate(page_of_chunk) if c in keep]
        chunks = [chunks[i] for i in idx]
        page_of_chunk = [page_of_chunk[i] for i in idx]
    page_index = {p: i for i, p in enumerate(pages)}
    chunk_page = np.array([page_index[p] for p in page_of_chunk])
    relevant_of_query = {}
    for qi, page in enumerate(sorted(set(page_of_chunk))):
        relevant_of_query[page] = set(np.flatnonzero(chunk_page == page_index[page]).tolist())

    stop = None
    if use_sklearn:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore[import-untyped]
            stop = sorted(ENGLISH_STOP_WORDS)
        except ImportError:
            stop = None

    t0 = time.perf_counter()
    vec = TextVectorizer(stop_words=stop, ngram_range=(1, 2), norm="l2")
    vec.fit(chunks)
    doc_scores = [vec.transform_one(c)["scores"] for c in chunks]
    build_ours = time.perf_counter() - t0

    q_scores = [vec.transform_one(q)["scores"] for q in queries]

    index = SparseCosineIndex(doc_scores, max_terms_per_document=50,
                                dtype=np.float32)

    rrs = []
    t0 = time.perf_counter()
    for qs, page in zip(q_scores, pages):
        rel = relevant_of_query[page]
        sims = index.query(qs)
        ranked = np.argsort(-sims)
        rrs.append(_rr(ranked, rel))
    query_ours = (time.perf_counter() - t0) / max(len(queries), 1)
    rrs = np.array(rrs)

    rrs_loop = []
    t0 = time.perf_counter()
    for qs, page in zip(q_scores, pages):
        rel = relevant_of_query[page]
        sims = cosine_similarity_many(qs, doc_scores)
        ranked = np.argsort(-sims)
        rrs_loop.append(_rr(ranked, rel))
    query_loop = (time.perf_counter() - t0) / max(len(queries), 1)
    rrs_loop = np.array(rrs_loop)

    out: dict = {
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "ours": {"rr_median": float(np.median(rrs)),
                 "rr_iqr": float(np.percentile(rrs, 75) - np.percentile(rrs, 25)),
                 "build_s": build_ours, "query_s": query_ours,
                 "method": "sparse"},
        "ours_loop": {"rr_median": float(np.median(rrs_loop)),
                      "rr_iqr": float(np.percentile(rrs_loop, 75)
                                      - np.percentile(rrs_loop, 25)),
                      "query_s": query_loop, "method": "loop"},
    }

    if use_sklearn:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
            from sklearn.metrics.pairwise import cosine_similarity as sk_cos  # type: ignore[import-untyped]
            t0 = time.perf_counter()
            sk = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 max_features=20000)
            X = sk.fit_transform(chunks)
            build_sk = time.perf_counter() - t0
            t0 = time.perf_counter()
            rrs_sk = []
            for qi, page in enumerate(pages):
                qv = sk.transform([queries[qi]])
                sims = sk_cos(qv, X).ravel()
                ranked = np.argsort(-sims)
                rrs_sk.append(_rr(ranked, relevant_of_query[page]))
            query_sk = (time.perf_counter() - t0) / max(len(pages), 1)
            rrs_sk = np.array(rrs_sk)
            out["sklearn"] = {
                "rr_median": float(np.median(rrs_sk)),
                "rr_iqr": float(np.percentile(rrs_sk, 75) - np.percentile(rrs_sk, 25)),
                "build_s": build_sk, "query_s": query_sk,
            }
            if np.allclose(rrs, rrs_sk):
                out["wilcoxon_p"] = 1.0
                out["a12"] = 0.5
                out["wilcoxon_note"] = (
                    "all reciprocal ranks equal: the two systems rank "
                    "the same chunk first for every query, the test "
                    "has no pair to compare")
            else:
                out["wilcoxon_p"] = _wilcoxon(rrs, rrs_sk)
                out["a12"] = _a12(rrs, rrs_sk)
        except (ImportError, ValueError, KeyError) as e:
            out["sklearn_error"] = f"{type(e).__name__}: {e}"
    else:
        out["sklearn_error"] = "not asked"

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="the first 20 queries")
    ap.add_argument("--docs", type=Path, default=ROOT / "docs")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "text_retrieval.json")
    args = ap.parse_args()
    r = run(args.docs, max_queries=SMOKE_QUERIES if args.smoke else None)
    print(f"chunks {r['n_chunks']}, queries {r['n_queries']}")
    print(f"ours:      RR median {r['ours']['rr_median']:.4f} "
          f"IQR {r['ours']['rr_iqr']:.4f}  build {r['ours']['build_s']:.3f}s "
          f"query {r['ours']['query_s']:.4f}s (sparse)")
    print(f"ours loop: RR median {r['ours_loop']['rr_median']:.4f} "
          f"IQR {r['ours_loop']['rr_iqr']:.4f}  query {r['ours_loop']['query_s']:.4f}s")
    if "sklearn" in r:
        s = r["sklearn"]
        print(f"sklearn:   RR median {s['rr_median']:.4f} IQR {s['rr_iqr']:.4f} "
              f" build {s['build_s']:.3f}s query {s['query_s']:.4f}s")
        print(f"Wilcoxon p {r['wilcoxon_p']:.4f}  A12 {r['a12']:.3f}")
    else:
        print(f"sklearn: not computed ({r.get('sklearn_error', '?')})")
    args.out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
