"""Tests for the ITCH 5.0 decoder."""

import pytest

from dense_armor.utility.finance.itch import decode_message


def test_a_add_order_no_attribution():
    m = (
        b"A"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + b"B"
        + (100).to_bytes(4, "big")
        + b"AAPL    "
        + (1234567).to_bytes(4, "big")
    )
    d = decode_message(m)
    assert d["type"] == "A"
    assert d["locate"] == 1
    assert d["tracking"] == 2
    assert d["order_ref"] == 7
    assert d["side"] == "B"
    assert d["shares"] == 100
    assert d["stock"] == "AAPL"
    assert d["price_raw"] == 1234567
    assert abs(d["price"] - 123.4567) < 1e-9


def test_f_add_order_with_attribution():
    m = (
        b"F"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + b"S"
        + (200).to_bytes(4, "big")
        + b"MSFT    "
        + (4200000).to_bytes(4, "big")
        + b"NSDQ"
    )
    d = decode_message(m)
    assert d["type"] == "F"
    assert d["side"] == "S"
    assert d["stock"] == "MSFT"
    assert d["attribution"] == "NSDQ"
    assert abs(d["price"] - 420.0) < 1e-9


def test_e_order_executed():
    m = (
        b"E"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + (50).to_bytes(4, "big")
        + (12345).to_bytes(8, "big")
    )
    d = decode_message(m)
    assert d["type"] == "E"
    assert d["order_ref"] == 7
    assert d["executed_shares"] == 50
    assert d["match_number"] == 12345


def test_c_order_executed_with_price():
    m = (
        b"C"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + (50).to_bytes(4, "big")
        + (12345).to_bytes(8, "big")
        + b"Y"
        + (1234567).to_bytes(4, "big")
    )
    d = decode_message(m)
    assert d["type"] == "C"
    assert d["printable"] == "Y"
    assert abs(d["price"] - 123.4567) < 1e-9


def test_x_order_cancel():
    m = (
        b"X"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + (25).to_bytes(4, "big")
    )
    d = decode_message(m)
    assert d["type"] == "X"
    assert d["cancelled_shares"] == 25


def test_d_order_delete():
    m = (
        b"D"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
    )
    d = decode_message(m)
    assert d["type"] == "D"
    assert d["order_ref"] == 7


def test_u_order_replace():
    m = (
        b"U"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + (8).to_bytes(8, "big")
        + (500).to_bytes(4, "big")
        + (4000000).to_bytes(4, "big")
    )
    d = decode_message(m)
    assert d["type"] == "U"
    assert d["original_ref"] == 7
    assert d["new_ref"] == 8
    assert d["shares"] == 500
    assert abs(d["price"] - 400.0) < 1e-9


def test_timestamp_is_big_endian():
    # 00 00 00 00 01 00 = 256 ns
    m = (
        b"D"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + bytes.fromhex("000000000100")
        + (7).to_bytes(8, "big")
    )
    assert decode_message(m)["timestamp"] == 256
    # 01 00 00 00 00 00 = 1099511627776 ns
    m = (
        b"D"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + bytes.fromhex("010000000000")
        + (7).to_bytes(8, "big")
    )
    assert decode_message(m)["timestamp"] == 1099511627776


def test_stock_padding_is_stripped():
    m = (
        b"A"
        + (0).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (0).to_bytes(8, "big")
        + b"B"
        + (1).to_bytes(4, "big")
        + b"X       "
        + (0).to_bytes(4, "big")
    )
    assert decode_message(m)["stock"] == "X"


def test_type_and_length_errors():
    with pytest.raises(ValueError):
        decode_message(b"")
    with pytest.raises(ValueError):
        decode_message(b"Z" + b"\x00" * 35)
    with pytest.raises(ValueError):
        decode_message(b"A" + b"\x00" * 34)
    with pytest.raises(TypeError):
        decode_message("A not bytes")
