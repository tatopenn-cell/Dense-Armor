"""Tests for frame and iter_itch: hand-built ITCH messages."""

import pytest

from dense_armor.utility.finance.feed import frame, iter_itch


def _A(order_ref: int, side: str, shares: int, stock: str, price_raw: int) -> bytes:
    b = bytearray(36)
    b[0] = ord("A")
    b[1:3] = (1).to_bytes(2, "big")
    b[3:5] = (2).to_bytes(2, "big")
    b[5:11] = (3).to_bytes(6, "big")
    b[11:19] = order_ref.to_bytes(8, "big")
    b[19] = ord(side)
    b[20:24] = shares.to_bytes(4, "big")
    b[24:32] = stock.encode("ascii").ljust(8)
    b[32:36] = price_raw.to_bytes(4, "big")
    return bytes(b)


def _D(order_ref: int) -> bytes:
    b = bytearray(19)
    b[0] = ord("D")
    b[11:19] = order_ref.to_bytes(8, "big")
    return bytes(b)


def _U(original_ref: int, new_ref: int, shares: int, price_raw: int) -> bytes:
    b = bytearray(35)
    b[0] = ord("U")
    b[11:19] = original_ref.to_bytes(8, "big")
    b[19:27] = new_ref.to_bytes(8, "big")
    b[27:31] = shares.to_bytes(4, "big")
    b[31:35] = price_raw.to_bytes(4, "big")
    return bytes(b)


def _record(m: bytes) -> bytes:
    return len(m).to_bytes(2, "big") + m


def test_1_three_messages_decoded(tmp_path):
    path = tmp_path / "feed.bin"
    a = _A(11, "B", 100, "AAPL", 1234567)
    d = _D(11)
    u = _U(11, 22, 200, 2345678)
    path.write_bytes(_record(a) + _record(d) + _record(u))
    msgs = list(iter_itch(path))
    assert len(msgs) == 3
    assert msgs[0]["type"] == "A"
    assert msgs[0]["order_ref"] == 11
    assert msgs[0]["side"] == "B"
    assert msgs[0]["shares"] == 100
    assert msgs[0]["stock"] == "AAPL"
    assert msgs[0]["price_raw"] == 1234567
    assert msgs[0]["price"] == 123.4567
    assert msgs[1]["type"] == "D"
    assert msgs[1]["order_ref"] == 11
    assert msgs[2]["type"] == "U"
    assert msgs[2]["original_ref"] == 11
    assert msgs[2]["new_ref"] == 22
    assert msgs[2]["shares"] == 200
    assert msgs[2]["price_raw"] == 2345678


def test_2_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert list(iter_itch(path)) == []


def test_3_length_prefix_one_byte(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match=r"offset 0.*1 byte"):
        list(iter_itch(path))


def test_4_body_shorter_than_length(tmp_path):
    path = tmp_path / "short.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + (40).to_bytes(2, "big") + b"\x00" * 10)
    with pytest.raises(ValueError, match=r"offset 38"):
        list(iter_itch(path))


def test_5_unknown_type_reports_record_offset(tmp_path):
    path = tmp_path / "unknown.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    bad = b"Z" + b"\x00" * 35
    path.write_bytes(_record(a) + _record(bad))
    with pytest.raises(ValueError, match=r"offset 38"):
        list(iter_itch(path))


def test_6_length_zero(tmp_path):
    path = tmp_path / "zero.bin"
    path.write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match=r"length 0"):
        list(iter_itch(path))


def test_7_generator_is_lazy(tmp_path):
    path = tmp_path / "lazy.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + b"\x00")
    it = iter_itch(path)
    first = next(it)
    assert first["type"] == "A"
    assert first["order_ref"] == 1
    with pytest.raises(ValueError):
        next(it)


def test_8_frame_bytes_and_refusals():
    assert frame(b"abc") == b"\x00\x03abc"
    assert frame(bytearray(b"abc")) == b"\x00\x03abc"
    assert frame(memoryview(b"abc")) == b"\x00\x03abc"
    with pytest.raises(ValueError, match="empty"):
        frame(b"")
    with pytest.raises(ValueError, match="65535"):
        frame(b"x" * 65536)
    with pytest.raises(TypeError):
        frame("abc")  # type: ignore[arg-type]


def test_3b_prefix_one_byte_after_yielded(tmp_path):
    """The count after yielded is in the message, not just offset 0."""
    path = tmp_path / "bad_after.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + b"\x00")
    with pytest.raises(ValueError, match=r"offset 38.*after 1 messages"):
        list(iter_itch(path))


def test_4b_truncated_body_count(tmp_path):
    """The expected length and the yielded count are in the message."""
    path = tmp_path / "short.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + (40).to_bytes(2, "big") + b"\x00" * 10)
    with pytest.raises(
        ValueError, match=r"offset 38.*body is 10 bytes.*expected 40.*after 1 messages"
    ):
        list(iter_itch(path))


def test_5b_known_type_with_wrong_length_reports_offset(tmp_path):
    path = tmp_path / "wrong_length.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + _record(b"A" + bytes(9)))
    with pytest.raises(ValueError, match=r"offset 38.*after 1 messages"):
        list(iter_itch(path))


def test_9_messages_before_a_bad_record_were_yielded(tmp_path):
    path = tmp_path / "partial.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    d = _D(1)
    path.write_bytes(_record(a) + _record(d) + (40).to_bytes(2, "big") + bytes(10))
    seen = []
    with pytest.raises(ValueError, match="offset 59"):
        for msg in iter_itch(path):
            seen.append(msg["type"])
    assert seen == ["A", "D"]


def test_10_file_is_closed_when_the_generator_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "two.bin"
    a = _A(1, "B", 10, "AAPL", 100)
    path.write_bytes(_record(a) + _record(_D(1)))
    opened = []
    real_open = type(path).open

    def spy(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(type(path), "open", spy)
    it = iter_itch(path)
    next(it)
    assert len(opened) == 1
    assert opened[0].closed is False
    it.close()
    assert opened[0].closed is True
