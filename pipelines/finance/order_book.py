"""Order book: does the book say where the price goes next?

The order flow imbalance of Cont, Kukanov and Stoikov (2010) is the net
order flow at the best bid and ask between two observations. The paper
defines the contribution of the n-th event (page 4)

    e_n = 1{PB_n >= PB_n-1} qB_n - 1{PB_n <= PB_n-1} qB_n-1
        - 1{PA_n <= PA_n-1} qA_n + 1{PA_n >= PA_n-1} qA_n-1

and the imbalance over an interval as the sum of the e_n (page 4). The
price impact is the regression (page 8, equation 4)

    dP_k = alpha + beta OFI_k + eps_k

with dP_k = (P_k - P_k-1) / delta, P the mid price and delta the tick
size. The paper reports a linear relation with an average R^2 of 65%
(abstract, page 1).

Reference: Cont, R., Kukanov, A. and Stoikov, S. (2010). *The Price
Impact of Order Book Events.* arXiv:1011.6402. The e_n, the OFI and the
mid-price change are on page 4; the regression is equation 4, page 8.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


def event_contribution(
    bid_px_prev: float, bid_sz_prev: float, ask_px_prev: float,
    ask_sz_prev: float, bid_px: float, bid_sz: float,
    ask_px: float, ask_sz: float,
) -> float:
    """The e_n of the paper, page 4, one event."""
    e = 0.0
    if bid_px >= bid_px_prev:
        e += bid_sz
    if bid_px <= bid_px_prev:
        e -= bid_sz_prev
    if ask_px <= ask_px_prev:
        e -= ask_sz
    if ask_px >= ask_px_prev:
        e += ask_sz_prev
    return float(e)


def order_flow_imbalance(book: np.ndarray) -> np.ndarray:
    """OFI for a sequence of book states.

    Args:
        book: an ``n`` by 4 array, rows ``(bid_px, bid_sz, ask_px,
            ask_sz)`` in time order.

    Returns:
        An array of length ``n - 1``: the OFI between ``i`` and ``i+1``.
    """
    b = np.asarray(book, dtype=float)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"book must be n by 4, got {b.shape}")
    out = np.empty(len(b) - 1, dtype=float)
    for i in range(1, len(b)):
        out[i - 1] = event_contribution(
            b[i - 1, 0], b[i - 1, 1], b[i - 1, 2], b[i - 1, 3],
            b[i, 0], b[i, 1], b[i, 2], b[i, 3])
    return out


def mid_price_changes(book: np.ndarray, tick: float = 0.01) -> np.ndarray:
    """dP_k = (P_k - P_k-1) / tick, with P the mid price (page 4)."""
    b = np.asarray(book, dtype=float)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"book must be n by 4, got {b.shape}")
    mid = 0.5 * (b[:, 0] + b[:, 2])
    return np.diff(mid) / float(tick)


def read_book_csv(path) -> np.ndarray:
    """Read a CSV with columns ``t, bid_px, bid_sz, ask_px, ask_sz``."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        needed = ("t", "bid_px", "bid_sz", "ask_px", "ask_sz")
        if any(c not in header for c in needed):
            raise ValueError(
                f"{path}: missing a column of {needed}, header is {header}")
        idx = [header.index(c) for c in needed]
        rows = [[float(r[i]) for i in idx] for r in reader if len(r) > max(idx)]
    a = np.asarray(rows, dtype=float)
    return a[:, 1:]


def synthetic_book(
    n: int = 2000, seed: int = 0, slope: float = 2e-4,
    noise: float = 1e-3, depth: float = 100.0, tick: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic labelled book with a planted linear relation.

    The book walks with an OFI drawn from a normal, and the mid-price
    moves by ``slope * OFI`` plus noise. The book is synthetic.

    Returns:
        ``(book, true_ofi)``: the book states, ``n`` by 4, and the OFI
        that generated each step, length ``n - 1``.
    """
    rng = np.random.default_rng(seed)
    mid = 100.0
    rows = np.empty((n, 4), dtype=float)
    ofi = rng.normal(scale=depth / 3.0, size=n - 1)
    rows[0] = [mid - tick / 2.0, depth, mid + tick / 2.0, depth]
    for i in range(1, n):
        mid = mid + slope * ofi[i - 1] + noise * rng.normal()
        rows[i] = [mid - tick / 2.0, depth, mid + tick / 2.0, depth]
    return rows, ofi


@dataclass
class OrderBookResult:
    """Result of the regression of the mid-price change on the OFI.

    Attributes:
        n_events: number of book states.
        slope: the estimated beta.
        intercept: the estimated alpha.
        r2: R^2 of the regression.
        true_slope: the planted slope, or ``None`` for a real book.
    """

    n_events: int
    slope: float
    intercept: float
    r2: float
    true_slope: float | None


def regress(book: np.ndarray, tick: float = 0.01,
            true_slope: float | None = None) -> OrderBookResult:
    """Regress the mid-price change on the OFI (equation 4, page 8)."""
    ofi = order_flow_imbalance(book)
    dp = mid_price_changes(book, tick)
    if ofi.size < 2:
        raise ValueError("at least three book states")
    a = np.column_stack([np.ones_like(ofi), ofi])
    coef, *_ = np.linalg.lstsq(a, dp, rcond=None)
    pred = a @ coef
    ss_res = float(np.sum((dp - pred) ** 2))
    ss_tot = float(np.sum((dp - dp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return OrderBookResult(
        n_events=len(book), slope=float(coef[1]),
        intercept=float(coef[0]), r2=r2, true_slope=true_slope)
