"""Cosine similarity between score dicts.

A document is a dict ``{term: score}``; the cosine similarity of two
documents is the dot product of their score vectors divided by the
product of their norms. A zero vector gives 0.0, never ``nan``.

Reference: the cosine of two vectors is the standard measure used to
rank a query against a document index (scikit-learn's
``cosine_similarity`` is the batch version). The paper of this folder
that the TF-IDF follows (Silajev 2026, arXiv:2609.14016) gives the TF-IDF
scores; the cosine is the ranking on those scores, not a formula of the
paper, and it is "my choice".
"""

from __future__ import annotations

import math

import numpy as np


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity of two ``{term: score}`` dicts.

    Args:
        a: first document, as ``{term: score}``.
        b: second document, as ``{term: score}``.

    Returns:
        The cosine, in ``[-1, 1]``; 0.0 if either document is all zero.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        w = b.get(k)
        if w is not None:
            dot += float(v) * float(w)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    denom = na * nb
    if denom == 0.0:
        return 0.0
    return dot / denom


def cosine_similarity_many(query: dict, matrix: list[dict]) -> np.ndarray:
    """Cosine similarity of one document against a list of documents.

    Args:
        query: the query, as ``{term: score}``.
        matrix: the documents, one ``{term: score}`` dict each.

    Returns:
        A numpy array, one cosine per document, in the order of
        ``matrix``.
    """
    return np.array([cosine_similarity(query, d) for d in matrix],
                    dtype=float)
