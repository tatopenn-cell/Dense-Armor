"""Tests for synthetic_itch: determinism, types, invariants, refusals."""

from collections import Counter

import pytest

from dense_armor.utility.finance.feed import iter_itch
from dense_armor.utility.finance.synthetic import synthetic_itch


def test_1_same_seed_same_bytes():
    assert synthetic_itch(50, 7) == synthetic_itch(50, 7)
    assert synthetic_itch(10, 0) == synthetic_itch(10, 0)


def test_2_different_seeds_differ():
    assert synthetic_itch(50, 0) != synthetic_itch(50, 1)


def test_3_written_and_read_back(tmp_path):
    data = synthetic_itch(2000, 0)
    path = tmp_path / "feed.bin"
    path.write_bytes(data)
    msgs = list(iter_itch(path))
    assert len(msgs) == 2000
    assert msgs[0]["type"] == "A"
    counts = Counter(m["type"] for m in msgs)
    for t in ("A", "F", "E", "C", "X", "D", "U"):
        assert counts[t] > 0, f"type {t} missing: {counts}"


def test_4_invariants(tmp_path):
    path = tmp_path / "feed.bin"
    path.write_bytes(synthetic_itch(1000, 3))
    msgs = list(iter_itch(path))

    ts_prev = None
    live: set[int] = set()
    all_refs: set[int] = set()
    for m in msgs:
        assert ts_prev is None or m["timestamp"] > ts_prev
        ts_prev = m["timestamp"]
        t = m["type"]
        if t in ("A", "F"):
            live.add(m["order_ref"])
            all_refs.add(m["order_ref"])
        elif t in ("E", "C", "X"):
            assert m["order_ref"] in live, f"{t} ref {m['order_ref']} not live"
        elif t == "D":
            assert m["order_ref"] in live
            live.discard(m["order_ref"])
        elif t == "U":
            assert m["original_ref"] in live
            assert m["new_ref"] not in all_refs
            live.discard(m["original_ref"])
            live.add(m["new_ref"])
            all_refs.add(m["new_ref"])


def test_5_refusals():
    with pytest.raises(ValueError):
        synthetic_itch(-1, 0)
    with pytest.raises(ValueError):
        synthetic_itch(2.5, 0)
    with pytest.raises(ValueError):
        synthetic_itch(True, 0)
    with pytest.raises(ValueError):
        synthetic_itch(5, -1)
    with pytest.raises(ValueError):
        synthetic_itch(5, True)
    with pytest.raises(ValueError):
        synthetic_itch(5, "0")
    assert synthetic_itch(0, 0) == b""
