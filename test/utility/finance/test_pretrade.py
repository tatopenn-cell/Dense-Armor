"""Tests for PreTradeChecker."""

import pytest

from dense_armor.utility.finance.pretrade import Decision, PreTradeChecker


def _checker(**kw):
    base = dict(
        max_notional=1e6,
        max_size=10.0,
        price_band=0.01,
        max_orders=5,
        window_s=1.0,
        duplicate_window_s=0.1,
    )
    base.update(kw)
    return PreTradeChecker(**base)


def test_approved_at_limit():
    c = _checker(max_size=10.0)
    assert c.check("a", "buy", 10.0, 100.0, 100.0, 0.0) == Decision(True, "ok")


def test_size_rejects_one_tick_above():
    c = _checker(max_size=10.0)
    d = c.check("a", "buy", 10.01, 100.0, 100.0, 0.0)
    assert d == Decision(False, "size")


def test_price_band_rejects_one_tick_above():
    c = _checker(price_band=0.01)
    d = c.check("a", "buy", 1.0, 101.01, 100.0, 0.0)
    assert d == Decision(False, "price_band")


def test_duplicate_rejects_same_order_in_window():
    c = _checker(duplicate_window_s=0.1)
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.0).approved
    d = c.check("a", "buy", 1.0, 100.0, 100.0, 0.05)
    assert d == Decision(False, "duplicate")


def test_duplicate_window_expires():
    c = _checker(duplicate_window_s=0.1)
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.0).approved
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.15).approved


def test_rate_rejects_after_max_orders():
    c = _checker(max_orders=2, window_s=1.0, price_band=0.05)
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.0).approved
    assert c.check("a", "buy", 1.0, 101.0, 100.0, 0.1).approved
    d = c.check("a", "buy", 1.0, 102.0, 100.0, 0.2)
    assert d == Decision(False, "rate")


def test_rate_window_expires():
    c = _checker(max_orders=1, window_s=0.1, price_band=0.05)
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.0).approved
    assert c.check("a", "buy", 1.0, 101.0, 100.0, 0.2).approved


def test_notional_rejects_above_cap():
    c = _checker(max_notional=1000.0, max_size=100.0)
    assert c.check("a", "buy", 5.0, 100.0, 100.0, 0.0).approved
    d = c.check("a", "buy", 6.0, 100.0, 100.0, 0.1)
    assert d == Decision(False, "notional")


def test_notional_exactly_at_limit_is_approved():
    c = _checker(max_notional=1000.0, max_size=100.0)
    assert c.check("a", "buy", 10.0, 100.0, 100.0, 0.0).approved


def test_first_failing_control_gives_reason():
    c = _checker(max_size=1.0, price_band=0.01)
    d = c.check("a", "buy", 100.0, 200.0, 100.0, 0.0)
    assert d.reason == "size"


def test_rejected_leaves_state_unchanged():
    c = _checker(max_notional=1e6, max_size=10.0)
    before = c.open_notional("a")
    d = c.check("a", "buy", 100.0, 100.0, 100.0, 0.0)
    assert d.reason == "size"
    assert c.open_notional("a") == before
    assert c.check("a", "buy", 1.0, 100.0, 100.0, 0.1).approved


def test_accounts_are_independent():
    c = _checker(max_notional=1000.0, max_size=100.0)
    assert c.check("a", "buy", 5.0, 100.0, 100.0, 0.0).approved
    assert c.check("b", "buy", 5.0, 100.0, 100.0, 0.1).approved
    assert c.open_notional("a") == 500.0
    assert c.open_notional("b") == 500.0


def test_release_and_error():
    c = _checker(max_notional=1000.0)
    assert c.check("a", "buy", 5.0, 100.0, 100.0, 0.0).approved
    c.release("a", 300.0)
    assert c.open_notional("a") == 200.0
    with pytest.raises(ValueError):
        c.release("a", 1000.0)


def test_constructor_rejects_non_positive_limits():
    for kw in (
        "max_notional",
        "max_size",
        "price_band",
        "window_s",
        "duplicate_window_s",
    ):
        with pytest.raises(ValueError):
            _checker(**{kw: 0.0})


def test_constructor_rejects_zero_max_orders():
    with pytest.raises(ValueError):
        _checker(max_orders=0)


def test_check_rejects_bad_inputs():
    c = _checker()
    with pytest.raises(ValueError):
        c.check("a", "hold", 1.0, 100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        c.check("a", "buy", 0.0, 100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        c.check("a", "buy", 1.0, 0.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        c.check("a", "buy", 1.0, 100.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        c.check("a", "buy", 1.0, 100.0, 100.0, 1.0)
        c.check("a", "buy", 1.0, 100.0, 100.0, 0.5)
