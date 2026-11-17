"""Tests for PageRank, RWR, anomaly score."""

import networkx as nx
import numpy as np
import pytest

from dense_armor.core.graph import (
    Graph,
    graph_anomaly_score,
    page_rank,
    random_walk_restart,
)


def _star(n=10):
    edges = [(0, i, 1.0) for i in range(1, n)]
    return Graph(n, edges)


def _path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def test_page_rank_matches_networkx():
    g = _path(6)
    G_nx = g.to_networkx()
    ours = page_rank(g)
    ref = nx.pagerank(G_nx)
    for n in range(6):
        assert abs(ours[n] - ref[n]) < 1e-12


def test_page_rank_center_max_star():
    g = _star(10)
    pr = page_rank(g)
    vals = np.array([pr[i] for i in range(10)])
    assert np.argmax(vals) == 0


def test_rwr_seed_gets_mass():
    g = _path(10)
    rwr = random_walk_restart(g, seed=0, alpha=0.85)
    # la massa è concentrata vicino al seed, non necessariamente > 0.8
    top3 = sum(rwr[i] for i in range(3))
    assert top3 > 0.7
    # il seed ha più massa dell'estremo opposto
    assert rwr[0] > rwr[9]
    assert abs(sum(rwr.values()) - 1.0) < 1e-12


def test_rwr_unknown_seed():
    with pytest.raises(ValueError):
        random_walk_restart(_path(4), seed=99)


def test_graph_anomaly_score_isolates_outlier():
    g = _star(10)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(10) * 0.1
    sig[4] = 5.0
    scores = graph_anomaly_score(g, sig)
    pr = page_rank(g)
    assert max(pr, key=pr.get) == 0
    assert max(scores, key=scores.get) == 4


def test_graph_anomaly_score_uniform_is_page_rank():
    g = _path(5)
    scores = graph_anomaly_score(g, np.ones(5))
    pr = page_rank(g)
    for i in range(5):
        assert abs(scores[i] - pr[i]) < 1e-12
