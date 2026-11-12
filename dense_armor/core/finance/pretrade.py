"""Pre-trade limits for one order, from SEC Rule 15c3-5 (c)(1)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One decision on one order."""

    approved: bool
    reason: str


class PreTradeChecker:
    """Accept or reject one order with the five controls of (c)(1).

    The controls run in this order and the first one that fails gives
    the reason; an approved order has ``reason == "ok"``:

    1. ``"size"``: ``size > max_size`` (SEC 15c3-5 (c)(1)(ii), size
       parameters, PDF page 128).
    2. ``"price_band"``: ``abs(price / ref_price - 1) > price_band``
       ((c)(1)(ii), price parameters, PDF page 128).
    3. ``"duplicate"``: an approved order with the same ``account``,
       ``side``, ``size`` and ``price`` was approved in the last
       ``duplicate_window_s`` seconds ((c)(1)(ii), duplicative orders, PDF
       page 128).
    4. ``"rate"``: ``max_orders`` or more orders of the account were
       approved in the last ``window_s`` seconds ((c)(1)(ii), over a short
       period of time, PDF page 128).
    5. ``"notional"``: the open notional of the account plus
       ``size * price`` is above ``max_notional`` ((c)(1)(i), pre-set
       credit or capital thresholds, in the aggregate for each customer,
       PDF page 128).

    A rejected order does not change the open notional, the rate window,
    the duplicate window or the time of the last approved sample. An
    approved order records its timestamp as the new ``_last_t``, adds its
    entry to the rate and duplicate windows, adds ``size * price`` to the
    open notional, and returns ``Decision(True, "ok")``. A ``t`` smaller
    than the last **approved** ``t`` raises ``ValueError`` before the five
    controls run: the sequence of approved timestamps is monotone, but the
    sequence of timestamps of all orders, rejected ones included, is not. The windows are
    filtered to the alive entries on every ``check`` call, so an entry
    that is older than the window is never counted and never written
    back. Every approved order calls ``_prune``, which drops the entries
    of every account that are older than the two windows; a rejected
    order never calls it. The cost of an approved order is
    O(accounts with entries in the rate or duplicate lists, plus the
    entries that survive); a rejected one costs O(entries of its own
    account). The open notional map is never pruned: an
    account that stops trading keeps its last notional until ``release``
    brings it back to zero. The value exactly at a limit is
    approved; one tick above is rejected. Only approved
    orders count for the rate and for the duplicates: that is our choice,
    the rule does not say it.

    The ``notional`` control is per account, not aggregated across
    accounts for the broker or dealer. Rule 15c3-5 (c)(1)(i) requires the
    aggregate limit for each customer **and** the broker or dealer; the
    broker-wide aggregate is not implemented. This is a declared limit,
    not a deviation from the rule: the class does not replace the
    broker's own controls.

    Args:
        max_notional: the largest open notional an account can carry, in
            the currency of the prices; strictly positive. This is the
            per-account limit; the aggregate for the broker is not
            implemented (see above).
        max_size: the largest size of one order, in shares; strictly
            positive.
        price_band: the largest relative distance of the price from the
            reference price, as a fraction; strictly positive.
        max_orders: the number of approved orders in ``window_s`` after
            which the account is rejected; at least 1.
        window_s: the length of the rate window, in seconds; strictly
            positive.
        duplicate_window_s: the length of the duplicate window, in
            seconds; strictly positive.

    Raises:
        ValueError: if a limit is not strictly positive.

    Examples:
        >>> c = PreTradeChecker(max_notional=1e6, max_size=10.0,
        ...                     price_band=0.01, max_orders=5,
        ...                     window_s=1.0, duplicate_window_s=0.1)
        >>> c.check("a", "buy", 10.0, 100.0, 100.0, t=0.0)
        Decision(approved=True, reason='ok')
        >>> c.check("a", "buy", 10.01, 100.0, 100.0, t=0.1).reason
        'size'
        >>> c.open_notional("a")
        1000.0

    References:
        SEC Release 34-63241, Rule 15c3-5 (c)(1)(i) and (c)(1)(ii), PDF
        page 128: pre-set credit or capital thresholds in the aggregate for
        each customer; price or size parameters, over a short period of
        time, duplicative orders.
    """

    def __init__(
        self,
        max_notional: float,
        max_size: float,
        price_band: float,
        max_orders: int,
        window_s: float,
        duplicate_window_s: float,
    ) -> None:
        for name, v in (
            ("max_notional", max_notional),
            ("max_size", max_size),
            ("price_band", price_band),
            ("window_s", window_s),
            ("duplicate_window_s", duplicate_window_s),
        ):
            fv = float(v)
            if not (math.isfinite(fv) and fv > 0.0):
                raise ValueError(
                    f"{name} must be a finite number strictly positive, got {v}"
                )
        if int(max_orders) < 1:
            raise ValueError(f"max_orders must be >= 1, got {max_orders}")
        self.max_notional = float(max_notional)
        self.max_size = float(max_size)
        self.price_band = float(price_band)
        self.max_orders = int(max_orders)
        self.window_s = float(window_s)
        self.duplicate_window_s = float(duplicate_window_s)
        self._open: dict[str, float] = {}
        self._rate: dict[str, list[float]] = {}
        self._dup: dict[str, list[tuple[float, float, float, float, str]]] = {}
        self._last_t = -float("inf")

    def _prune(self, t: float) -> None:
        """Drop the entries of every account that are older than the windows."""
        lo = t - self.window_s
        dlo = t - self.duplicate_window_s
        for k in list(self._rate.keys()):
            self._rate[k] = [u for u in self._rate[k] if u >= lo]
            if not self._rate[k]:
                del self._rate[k]
        for k in list(self._dup.keys()):
            self._dup[k] = [r for r in self._dup[k] if r[0] >= dlo]
            if not self._dup[k]:
                del self._dup[k]

    def check(
        self,
        account: str,
        side: str,
        size: float,
        price: float,
        ref_price: float,
        t: float,
    ) -> Decision:
        """Decide one order. A rejected order leaves the state unchanged."""
        if side not in ("buy", "sell"):
            raise ValueError(f'side must be "buy" or "sell", got {side!r}')
        sz = float(size)
        px = float(price)
        rpx = float(ref_price)
        tf = float(t)
        for name, v in (("size", sz), ("price", px), ("ref_price", rpx), ("t", tf)):
            if not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v}")
        if not sz > 0.0:
            raise ValueError(f"size must be positive, got {size}")
        if not px > 0.0:
            raise ValueError(f"price must be positive, got {price}")
        if not rpx > 0.0:
            raise ValueError(f"ref_price must be positive, got {ref_price}")
        if tf < self._last_t:
            raise ValueError(f"t must be >= the previous t ({self._last_t}), got {t}")
        rate_alive = [u for u in self._rate.get(account, []) if u >= tf - self.window_s]
        dup_alive = [
            r
            for r in self._dup.get(account, [])
            if r[0] >= tf - self.duplicate_window_s
        ]
        if sz > self.max_size:
            return Decision(False, "size")
        rel = abs(px / rpx - 1.0)
        if rel > self.price_band:
            return Decision(False, "price_band")
        for r in dup_alive:
            if r[1] == sz and r[2] == px and r[4] == side:
                return Decision(False, "duplicate")
        if len(rate_alive) >= self.max_orders:
            return Decision(False, "rate")
        open_now = self._open.get(account, 0.0)
        add = sz * px
        if open_now + add > self.max_notional:
            return Decision(False, "notional")
        self._last_t = tf
        self._open[account] = open_now + add
        self._prune(tf)
        rate_alive.append(tf)
        dup_alive.append((tf, sz, px, rpx, side))
        self._rate[account] = rate_alive
        self._dup[account] = dup_alive
        return Decision(True, "ok")

    def release(self, account: str, notional: float) -> None:
        """Subtract a notional after a fill or a cancel.

        Args:
            account: the account whose open notional is reduced.
            notional: the notional to release, a finite number ``> 0`` and
                ``<=`` the current open notional of the account.

        Raises:
            ValueError: if ``notional`` is not finite, if it is not ``> 0``,
                or if it is larger than the open notional of the account by
                more than ``1e-9`` of the open notional (or ``1e-9`` in
                absolute value, whichever is larger). This tolerance covers
                the floating point residue of a sequence of approvals and
                releases that cancel each other; an exact equality in float
                would raise a spurious ``ValueError`` after such a sequence.
        """
        cur = self._open.get(account, 0.0)
        n = float(notional)
        tol = 1e-9 * max(abs(cur), 1.0)
        if not math.isfinite(n) or not (0.0 < n <= cur + tol):
            raise ValueError(
                f"release of {notional} is not allowed: open notional is {cur}"
            )
        self._open[account] = max(cur - n, 0.0)

    def open_notional(self, account: str) -> float:
        """The open notional of the account."""
        return float(self._open.get(account, 0.0))
