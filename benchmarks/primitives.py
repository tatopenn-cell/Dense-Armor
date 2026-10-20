"""Measure the ranking primitive against the full product.

Writes benchmarks/primitives.json. Each number here comes from this
script, not from a notebook: the page cites this file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_retrieval import build_corpus

from dense_armor.utility.preprocessing.vectorizer import TextVectorizer
from dense_armor.utility.stats.similarity import SparseCosineIndex

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "primitives.json"


def main() -> int:
    chunks, _, queries = build_corpus(ROOT.parent / "docs")
    vec = TextVectorizer(ngram_range=(1, 2), norm="l2", smooth_idf=False)
    vec.fit(chunks)
    doc_scores = [vec.transform_one(c)["scores"] for c in chunks]
    q_scores = [vec.transform_one(q)["scores"] for q in queries]
    idx = SparseCosineIndex(doc_scores, dtype=np.float32)

    t0 = time.perf_counter()
    S = idx.query_many(q_scores)
    full_s = (time.perf_counter() - t0) / max(len(q_scores), 1)
    top_full = [set(np.argsort(-S[:, j])[:5].tolist())
                for j in range(S.shape[1])]

    def per_query(fn, n=3):
        runs = []
        for _ in range(n):
            t0 = time.perf_counter()
            for qd in q_scores:
                fn(qd, k=5)
            runs.append(time.perf_counter() - t0)
        return float(np.median(runs)) / max(len(q_scores), 1)

    fagin_s = per_query(idx.top_k_threshold)
    fagin_same = sum(
        int({d for d, _ in idx.top_k_threshold(qd, k=5)} == top_full[j])
        for j, qd in enumerate(q_scores))

    payload = {
        "n_documents": len(chunks),
        "n_queries": len(q_scores),
        "full_matmul_s_per_query": full_s,
        "fagin_s_per_query": fagin_s,
        "fagin_top5_identical": fagin_same,
        "fagin_total": len(q_scores),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
