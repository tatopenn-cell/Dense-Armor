"""Tests for DecisionLog: canonical hash, chain, tampering, limits."""

import hashlib
import json
from pathlib import Path

import pytest

from dense_armor.utility.finance.decision_log import GENESIS_HASH, DecisionLog

CANONICAL_FIRST = '{"payload":{"a":1},"prev":"' + "0" * 64 + '","seq":0,"t":1.5}'


def _three_records(path: Path) -> DecisionLog:
    log = DecisionLog(path)
    log.append(1.0, {"a": 1})
    log.append(2.0, {"a": 2})
    log.append(3.0, {"a": 3})
    return log


def test_1_first_hash_equals_hand_computed(tmp_path):
    expected = hashlib.sha256(CANONICAL_FIRST.encode("ascii")).hexdigest()
    log = DecisionLog(tmp_path / "log.jsonl")
    h = log.append(1.5, {"a": 1})
    assert h == expected
    assert len(GENESIS_HASH) == 64


def test_2_chain_prev_seq_len_last_hash(tmp_path):
    path = tmp_path / "log.jsonl"
    log = DecisionLog(path)
    h0 = log.append(1.0, {"a": 1})
    h1 = log.append(2.0, {"a": 2})
    h2 = log.append(3.0, {"a": 3})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["seq"] == 0
    assert json.loads(lines[1])["seq"] == 1
    assert json.loads(lines[2])["seq"] == 2
    assert json.loads(lines[0])["prev"] == GENESIS_HASH
    assert json.loads(lines[1])["prev"] == h0
    assert json.loads(lines[2])["prev"] == h1
    assert json.loads(lines[0])["hash"] == h0
    assert json.loads(lines[1])["hash"] == h1
    assert json.loads(lines[2])["hash"] == h2
    assert log.verify() is True
    assert len(log) == 3
    assert log.last_hash == h2


@pytest.mark.parametrize(
    "tamper", ["delete_middle", "edit_payload", "swap", "garbage", "cut_last"]
)
def test_3_tampering_detected(tmp_path, tamper):
    path = tmp_path / "log.jsonl"
    _three_records(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    if tamper == "delete_middle":
        new = lines[:1] + lines[2:]
    elif tamper == "edit_payload":
        new = [lines[0], lines[1].replace('"a":2', '"a":99'), lines[2]]
    elif tamper == "swap":
        new = [lines[1], lines[0], lines[2]]
    elif tamper == "garbage":
        new = [lines[0], "not json\n", lines[1], lines[2]]
    else:
        new = lines[:-1] + [lines[-1][:-10]]

    path.write_text("".join(new), encoding="utf-8", newline="\n")
    assert DecisionLog(path).verify() is False


def test_4_new_instance_continues_chain(tmp_path):
    path = tmp_path / "log.jsonl"
    log1 = DecisionLog(path)
    log1.append(1.0, {"a": 1})
    h1 = log1.append(2.0, {"a": 2})

    log2 = DecisionLog(path)
    assert log2.last_hash == h1
    assert len(log2) == 2

    h2 = log2.append(3.0, {"a": 3})
    log3 = DecisionLog(path)
    assert len(log3) == 3
    assert log3.verify() is True
    assert log3.last_hash == h2
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[2])["seq"] == 2


def test_5_append_on_broken_log_raises_and_does_not_write(tmp_path):
    path = tmp_path / "log.jsonl"
    _three_records(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        lines[0] + lines[1].replace('"a":2', '"a":99') + lines[2],
        encoding="utf-8",
        newline="\n",
    )
    before = path.read_bytes()

    log = DecisionLog(path)
    with pytest.raises(ValueError):
        log.append(4.0, {"a": 4})

    assert path.read_bytes() == before


def test_6_refused_input_writes_nothing(tmp_path):
    path = tmp_path / "log.jsonl"

    log = DecisionLog(path)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            log.append(bad, {"a": 1})
    assert not path.exists()

    log.append(1.0, {"a": 1})
    before = path.read_bytes()
    bad_calls = [
        (float("nan"), {"a": 2}),
        (2.0, {"a": float("nan")}),
        (2.0, [1, 2, 3]),
        (2.0, {"a": object()}),
    ]
    for t, payload in bad_calls:
        with pytest.raises(ValueError):
            log.append(t, payload)
    assert path.read_bytes() == before


def test_7_missing_and_empty_file(tmp_path):
    missing = tmp_path / "missing.jsonl"
    log = DecisionLog(missing)
    assert log.verify() is True
    assert len(log) == 0
    assert log.last_hash == GENESIS_HASH

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    log = DecisionLog(empty)
    assert log.verify() is True
    assert len(log) == 0
    assert log.last_hash == GENESIS_HASH


def test_8_deleting_tail_is_not_detected(tmp_path):
    path = tmp_path / "log.jsonl"
    log = _three_records(path)
    kept = log.last_hash
    assert len(log) == 3

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8", newline="\n")

    truncated = DecisionLog(path)
    assert truncated.verify() is True
    assert len(truncated) == 2
    assert truncated.last_hash != kept


def test_9_read_state_tolerates_binary_file(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"\xff\xfe\x00\x01")
    log = DecisionLog(path)
    assert log.last_hash == GENESIS_HASH
    assert log.verify() is False


def test_10_is_finite_number_bool_and_str():
    from dense_armor.utility.finance.decision_log import _is_finite_number

    assert _is_finite_number(True) is False
    assert _is_finite_number("1.0") is False
    assert _is_finite_number(None) is False
    assert _is_finite_number(1.5) is True
    assert _is_finite_number(2) is True


def test_11_parse_line_rejects_bad_fields(tmp_path):
    from dense_armor.utility.finance.decision_log import _parse_line

    good_prev = "0" * 64
    good_hash = "a" * 64
    with pytest.raises(ValueError, match="wrong keys"):
        _parse_line('{"seq":0}', 0)
    with pytest.raises(ValueError, match="non-integer seq"):
        _parse_line(
            f'{{"seq":"0","t":1.0,"payload":{{}},"prev":"{good_prev}","hash":"{good_hash}"}}',
            0,
        )
    with pytest.raises(ValueError, match="non-integer seq"):
        _parse_line(
            f'{{"seq":true,"t":1.0,"payload":{{}},"prev":"{good_prev}","hash":"{good_hash}"}}',
            0,
        )
    with pytest.raises(ValueError, match="non-finite t"):
        _parse_line(
            f'{{"seq":0,"t":"x","payload":{{}},"prev":"{good_prev}","hash":"{good_hash}"}}',
            0,
        )
    with pytest.raises(ValueError, match="non-dict payload"):
        _parse_line(
            f'{{"seq":0,"t":1.0,"payload":[1],"prev":"{good_prev}","hash":"{good_hash}"}}',
            0,
        )
    with pytest.raises(ValueError, match="invalid prev"):
        _parse_line(
            '{"seq":0,"t":1.0,"payload":{},"prev":"short","hash":"' + good_hash + '"}',
            0,
        )
    with pytest.raises(ValueError, match="invalid hash"):
        _parse_line(
            '{"seq":0,"t":1.0,"payload":{},"prev":"' + good_prev + '","hash":"ZZZ"}',
            0,
        )


def test_12_prev_check_isolated(tmp_path):
    """A record with the correct hash but the wrong prev: only the prev check
    can catch it, because the hash is recomputed on the wrong content."""
    import hashlib

    path = tmp_path / "log.jsonl"
    payload0 = {"a": 1}
    prev0 = "0" * 64
    canon0 = json.dumps(
        {"payload": payload0, "prev": prev0, "seq": 0, "t": 1.0},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    hash0 = hashlib.sha256(canon0.encode("ascii")).hexdigest()
    line0 = (
        json.dumps(
            {"hash": hash0, "payload": payload0, "prev": prev0, "seq": 0, "t": 1.0},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )

    payload1 = {"a": 2}
    prev1_wrong = "f" * 64
    canon1 = json.dumps(
        {"payload": payload1, "prev": prev1_wrong, "seq": 1, "t": 2.0},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    hash1 = hashlib.sha256(canon1.encode("ascii")).hexdigest()
    line1 = (
        json.dumps(
            {
                "hash": hash1,
                "payload": payload1,
                "prev": prev1_wrong,
                "seq": 1,
                "t": 2.0,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )

    path.write_text(line0 + line1, encoding="utf-8", newline="\n")
    log = DecisionLog(path)
    assert log.verify() is False


def test_13_seq_check_isolated(tmp_path):
    """A record with the correct hash but a wrong seq: only the seq check
    can catch it, because the hash is recomputed on the wrong content."""
    import hashlib

    path = tmp_path / "log.jsonl"
    payload0 = {"a": 1}
    prev0 = "0" * 64
    canon0 = json.dumps(
        {"payload": payload0, "prev": prev0, "seq": 0, "t": 1.0},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    hash0 = hashlib.sha256(canon0.encode("ascii")).hexdigest()
    line0 = (
        json.dumps(
            {"hash": hash0, "payload": payload0, "prev": prev0, "seq": 0, "t": 1.0},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )

    payload1 = {"a": 2}
    bad_seq = 5
    canon1 = json.dumps(
        {"payload": payload1, "prev": hash0, "seq": bad_seq, "t": 2.0},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    hash1 = hashlib.sha256(canon1.encode("ascii")).hexdigest()
    line1 = (
        json.dumps(
            {
                "hash": hash1,
                "payload": payload1,
                "prev": hash0,
                "seq": bad_seq,
                "t": 2.0,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )

    path.write_text(line0 + line1, encoding="utf-8", newline="\n")
    log = DecisionLog(path)
    assert log.verify() is False


def _write_one_line(path, obj):
    import json as _json

    path.write_text(
        _json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_14_nan_in_payload_returns_false(tmp_path):
    """json.loads accepts NaN; _hash with allow_nan=False raises; verify() must return False."""
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"hash":"'
        + "a" * 64
        + '","payload":{"a":NaN},"prev":"'
        + "0" * 64
        + '","seq":0,"t":1.0}\n',
        encoding="utf-8",
        newline="\n",
    )
    assert DecisionLog(path).verify() is False


def test_15_infinity_in_payload_returns_false(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"hash":"'
        + "a" * 64
        + '","payload":{"a":Infinity},"prev":"'
        + "0" * 64
        + '","seq":0,"t":1.0}\n',
        encoding="utf-8",
        newline="\n",
    )
    assert DecisionLog(path).verify() is False


def test_16_huge_int_in_t_returns_false(tmp_path):
    """float(10**400) raises OverflowError in _is_finite_number."""
    huge = 10**400
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"hash":"'
        + "a" * 64
        + '","payload":{},"prev":"'
        + "0" * 64
        + '","seq":0,"t":'
        + str(huge)
        + "}\n",
        encoding="utf-8",
        newline="\n",
    )
    assert DecisionLog(path).verify() is False


def test_17_deep_nesting_returns_false(tmp_path):
    """Deep nesting can raise RecursionError while reading or hashing a line."""
    deep = "[" * 5000 + "]" * 5000
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"hash":"'
        + "a" * 64
        + '","payload":{"x":'
        + deep
        + '},"prev":"'
        + "0" * 64
        + '","seq":0,"t":1.0}\n',
        encoding="utf-8",
        newline="\n",
    )
    assert DecisionLog(path).verify() is False
