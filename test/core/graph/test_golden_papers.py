"""Tests against the golden files of the paper primitives."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import gen_golden_papers as gen  # noqa: E402

GOLD = Path(__file__).parent / "golden"


def test_wavelet_matches_the_golden_file():
    ref = np.load(GOLD / "papers_wavelet_ring8.npz")
    got = gen.wavelet_arrays()
    for name in ref.files:
        assert np.allclose(got[name], ref[name], atol=1e-10), name


def test_recovery_matches_the_golden_file():
    ref = np.load(GOLD / "papers_recovery_denoised.npy")
    assert np.allclose(gen.recovery_array(), ref, atol=1e-6)


def test_learned_weights_match_the_golden_file():
    ref = np.load(GOLD / "papers_learning_weights.npy")
    assert np.allclose(gen.learning_array(), ref, atol=1e-6)


def test_adacgp_filters_match_the_golden_file():
    ref = np.load(GOLD / "papers_adacgp_filters.npy")
    assert np.allclose(gen.adacgp_array(), ref, atol=1e-8)


def test_structural_change_matches_the_golden_file():
    ref = json.loads((GOLD / "papers_change.json").read_text())
    got = gen.change_dict()
    assert got["added"] == ref["added"] and got["removed"] == ref["removed"]
    assert got["reweighted"] == ref["reweighted"]
    assert np.isclose(got["topology_change"], ref["topology_change"], atol=1e-12)
    assert np.isclose(got["weight_change"], ref["weight_change"], atol=1e-12)


def test_dataflow_matches_the_golden_file():
    assert gen.dataflow_dict() == json.loads(
        (GOLD / "papers_dataflow.json").read_text()
    )
