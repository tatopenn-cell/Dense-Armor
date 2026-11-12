"""DecisionLog.append_many: a batch, verified once, equal to n calls of append."""

from __future__ import annotations

import pytest

from dense_armor.utility.finance.decision_log import GENESIS_HASH, DecisionLog

ITEMS = [
    (
        float(i),
        {"order_ref": i, "approved": bool(i % 2), "reason": "ok" if i % 2 else "size"},
    )
    for i in range(25)
]


def test_the_batch_writes_the_same_bytes_as_one_call_each(tmp_path):
    a, b = DecisionLog(tmp_path / "a.jsonl"), DecisionLog(tmp_path / "b.jsonl")
    one_by_one = [a.append(t, p) for t, p in ITEMS]
    batch = b.append_many(ITEMS)
    assert batch == one_by_one
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len(b) == 25 and b.last_hash == a.last_hash and b.verify()


def test_a_batch_continues_a_chain_and_a_chain_can_continue_a_batch(tmp_path):
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append(0.0, {"x": 0})
    hs = log.append_many(ITEMS[:3])
    h = log.append(9.0, {"x": 9})
    assert len(log) == 5 and log.verify() and h == log.last_hash and hs[-1] != h
    assert DecisionLog(tmp_path / "log.jsonl").last_hash == h


def test_no_items_writes_nothing_and_changes_nothing(tmp_path):
    log = DecisionLog(tmp_path / "log.jsonl")
    assert log.append_many([]) == [] and len(log) == 0
    assert not (tmp_path / "log.jsonl").exists() and log.last_hash == GENESIS_HASH


@pytest.mark.parametrize("bad", [(float("nan"), {}), (1.0, [1]), ("x", {})])
def test_one_refused_record_leaves_the_file_unchanged(tmp_path, bad):
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append(0.0, {"x": 0})
    before = (tmp_path / "log.jsonl").read_bytes()
    with pytest.raises(ValueError):
        log.append_many([(1.0, {"x": 1}), bad, (2.0, {"x": 2})])
    assert (tmp_path / "log.jsonl").read_bytes() == before and len(log) == 1


def test_a_broken_chain_is_not_extended_by_a_batch(tmp_path):
    p = tmp_path / "log.jsonl"
    log = DecisionLog(p)
    log.append_many(ITEMS[:3])
    p.write_bytes(p.read_bytes().replace(b'"approved":false', b'"approved":true', 1))
    before = p.read_bytes()
    with pytest.raises(ValueError, match="broken"):
        log.append_many(ITEMS[3:5])
    assert p.read_bytes() == before


def test_a_big_batch_is_fast_enough_to_check_in_a_test(tmp_path):
    big = [(float(i), {"order_ref": i}) for i in range(5000)]
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append_many(big)
    assert len(log) == 5000 and log.verify()
