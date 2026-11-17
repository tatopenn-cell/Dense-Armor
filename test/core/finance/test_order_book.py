"""Tests for the order book pipeline."""

import numpy as np
import pytest

from pipelines.finance.order_book import (
    event_contribution,
    mid_price_changes,
    order_flow_imbalance,
    read_book_csv,
    regress,
    synthetic_book,
)


def test_event_contribution_bid_size_increase():
    e = event_contribution(100.0, 10.0, 100.01, 10.0,
                           100.0, 15.0, 100.01, 10.0)
    assert e == pytest.approx(5.0)


def test_event_contribution_bid_size_decrease():
    e = event_contribution(100.0, 15.0, 100.01, 10.0,
                           100.0, 10.0, 100.01, 10.0)
    assert e == pytest.approx(-5.0)


def test_event_contribution_ask_size_increase():
    e = event_contribution(100.0, 10.0, 100.01, 10.0,
                           100.0, 10.0, 100.01, 15.0)
    assert e == pytest.approx(-5.0)


def test_event_contribution_ask_size_decrease():
    e = event_contribution(100.0, 10.0, 100.01, 15.0,
                           100.0, 10.0, 100.01, 10.0)
    assert e == pytest.approx(5.0)


def test_event_contribution_bid_price_improvement():
    e = event_contribution(100.0, 10.0, 100.02, 10.0,
                           100.01, 7.0, 100.02, 10.0)
    assert e == pytest.approx(7.0)


def test_ofi_hand_made_sequence():
    book = np.array([
        [100.0, 10.0, 100.01, 10.0],
        [100.0, 15.0, 100.01, 10.0],
        [100.0, 15.0, 100.01, 12.0],
    ])
    ofi = order_flow_imbalance(book)
    assert len(ofi) == 2
    assert ofi[0] == pytest.approx(5.0)
    assert ofi[1] == pytest.approx(-2.0)


def test_ofi_rejects_bad_shape():
    with pytest.raises(ValueError, match="n by 4"):
        order_flow_imbalance(np.zeros((5, 3)))


def test_mid_price_changes():
    book = np.array([
        [100.0, 1.0, 100.02, 1.0],
        [100.01, 1.0, 100.03, 1.0],
    ])
    dp = mid_price_changes(book, tick=0.01)
    assert dp[0] == pytest.approx(1.0)


def test_regress_recovers_hand_made_slope():
    bid_sz = np.array([10.0, 14.0, 20.0, 26.0])
    ofi = np.diff(bid_sz)
    d = 3e-3 * ofi
    mid = 100.01 + np.concatenate([[0.0], np.cumsum(d)])
    book = np.column_stack([
        mid - 0.01, bid_sz, mid + 0.01, np.full(len(mid), 10.0)])
    res = regress(book, tick=0.01)
    assert np.isfinite(res.r2)
    assert res.slope != 0.0


def test_synthetic_book_recovers_sign_and_r2():
    book, _ofi = synthetic_book(n=2000, seed=0, slope=2e-4, noise=1e-3)
    res = regress(book, true_slope=2e-4)
    assert res.n_events == 2000
    assert res.slope > 0.0
    assert res.r2 > 0.5


def test_synthetic_book_is_deterministic():
    a, _ = synthetic_book(n=50, seed=1)
    b, _ = synthetic_book(n=50, seed=1)
    np.testing.assert_array_equal(a, b)


def test_read_book_csv(tmp_path):
    p = tmp_path / "book.csv"
    p.write_text("t,bid_px,bid_sz,ask_px,ask_sz\n"
                 "0,100.0,10,100.01,10\n"
                 "1,100.0,15,100.01,10\n")
    b = read_book_csv(p)
    assert b.shape == (2, 4)
    assert b[1, 1] == 15.0


def test_read_book_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("t,bid_px,bid_sz,ask_px\n0,100.0,10,100.01\n")
    with pytest.raises(ValueError, match="missing a column"):
        read_book_csv(p)


def test_regress_needs_three_states():
    with pytest.raises(ValueError, match="at least three"):
        regress(np.zeros((2, 4)))


def test_mid_price_changes_rejects_bad_shape():
    with pytest.raises(ValueError, match="n by 4"):
        mid_price_changes(np.zeros((5, 3)))
